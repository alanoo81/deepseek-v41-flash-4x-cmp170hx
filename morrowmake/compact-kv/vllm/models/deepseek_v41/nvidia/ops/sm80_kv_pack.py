# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless packed rows for the sm_80 compressed-KV and index-key caches.

The reference rounds what it caches (``sm80_attn_core``): the compressed latent
through FP4 (16, E4M3 scale) and the index keys through FP4 (32, E8M0). The
BF16 rows the plain caches hold are therefore exactly ``bf16(q * s)`` with
``q`` on the E2M1 grid and ``s`` an E4M3 or power-of-two scale. A packed row
keeps ``q`` as 4-bit codes and ``s`` as one byte per block, and unpacking
recomputes ``bf16(q * s)``: the same bits as the plain cache.

Row layout (``D`` values, block ``B``): ``D // 2`` code bytes (value ``2j`` in
the low nibble of byte ``j``, ``2j + 1`` in the high nibble; E2M1 codes, sign in
bit 3, -0.0 kept), then ``D // B`` scale bytes (E4M3 without sign, or the
biased E8M0 exponent).

``VLLM_SM80_COMPACT_KV=1`` selects the packed caches. A kv source's
compressed-KV cache then holds rows of 512 bytes: the packed latent (bytes
0-287), the packed index key of the same compressed position (bytes 288-355)
and padding. The index-key cache declares no memory of its own; its rows are
the view :func:`index_key_rows` of the compressed-KV cache, addressed through
that cache's block table and slots. Writers of the latent touch bytes 0-287
only (:func:`latent_rows`), writers of the key bytes 288-415.

The torch functions are the definition, the ``*_tl`` helpers are their Triton
twins.
"""

import os

import torch

from vllm.models.deepseek_v41.nvidia.ops.sm80_act_quant import (
    FP4_MAX,
    _pow2_ceil_tl,
    fp4_scale,
    round_e2m1,
    round_e2m1_tl,
    round_e4m3_tl,
)
from vllm.triton_utils import tl, triton

# (values, block, scale format, packed bytes) of the two packed rows.
COMP_D, COMP_BLOCK, COMP_SCALE, COMP_PACKED_BYTES = 512, 16, "e4m3", 288
IDXK_D, IDXK_BLOCK, IDXK_SCALE, IDXK_ROW_BYTES = 128, 32, "e8m0", 128
# The compressed-KV cache row, and where the index key sits in it.
COMP_ROW_BYTES = 512
IDXK_OFFSET = COMP_PACKED_BYTES


def compact_kv_enabled() -> bool:
    return os.environ.get("VLLM_SM80_COMPACT_KV", "0") == "1"


def is_packed(cache: torch.Tensor | None) -> bool:
    return cache is not None and cache.dtype == torch.uint8


def latent_rows(comp: torch.Tensor) -> torch.Tensor:
    """The packed-latent bytes of a packed compressed-KV cache's rows."""
    assert comp.shape[-1] == COMP_ROW_BYTES
    return comp[..., :COMP_PACKED_BYTES]


def index_key_rows(comp: torch.Tensor) -> torch.Tensor:
    """The index-key rows ([..., IDXK_ROW_BYTES] uint8 view) of a packed
    compressed-KV cache: same pages, block table and slots."""
    assert comp.shape[-1] == COMP_ROW_BYTES
    return comp[..., IDXK_OFFSET:IDXK_OFFSET + IDXK_ROW_BYTES]


# ---------------------------------------------------------------------------
# torch definition
# ---------------------------------------------------------------------------


def _e2m1_codes(q: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (int32, 0..15) of fp32 values on the E2M1 grid."""
    bits = q.view(torch.int32)
    sign = (bits >> 31) & 1
    a = q.abs()
    exp = ((a.view(torch.int32) >> 23) & 0xFF) - 127
    mant = (a.view(torch.int32) >> 22) & 1
    mag = torch.where(a < 1.0, (a * 2).to(torch.int32), ((exp + 1) << 1) | mant)
    return (sign << 3) | mag


def _e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """fp32 values of E2M1 codes (int32, 0..15), -0.0 kept."""
    mag = codes & 7
    e = mag >> 1
    m = mag & 1
    normal = ((e - 1 + 127) << 23) | (m << 22)
    half = torch.where(m == 1, torch.full_like(m, 126 << 23), torch.zeros_like(m))
    bits = torch.where(e == 0, half, normal) | ((codes >> 3) << 31)
    return bits.view(torch.float32)


def _scale_bytes(s: torch.Tensor, scale: str) -> torch.Tensor:
    """One byte (int32) per fp32 scale: E4M3 (positive) or biased E8M0."""
    bits = s.view(torch.int32)
    e32 = (bits >> 23) & 0xFF
    if scale == "e8m0":
        return e32
    denormal = (s / 2.0**-9).to(torch.int32)
    normal = ((e32 - 120) << 3) | ((bits >> 20) & 7)
    return torch.where(e32 < 121, denormal, normal)


def _scale_values(b: torch.Tensor, scale: str) -> torch.Tensor:
    """fp32 scales of the bytes (int32) :func:`_scale_bytes` writes."""
    if scale == "e8m0":
        return (b << 23).view(torch.float32)
    e = b >> 3
    m = b & 7
    normal = (((e - 7 + 127) << 23) | (m << 20)).view(torch.float32)
    return torch.where(e == 0, m.to(torch.float32) * 2.0**-9, normal)


def pack_rows(x: torch.Tensor, block: int, scale: str, row_bytes: int) -> torch.Tensor:
    """``[N, D]`` values -> ``[N, row_bytes]`` uint8 packed rows of the FP4
    quantization the reference applies (``fp4_qdq(x, block, scale)``)."""
    n, d = x.shape
    xf = x.float().unflatten(-1, (d // block, block))
    s = fp4_scale(xf.abs().amax(-1, keepdim=True), scale)
    q = round_e2m1((xf / s).clamp(-FP4_MAX, FP4_MAX)).flatten(-2)
    codes = _e2m1_codes(q.contiguous())
    out = torch.zeros((n, row_bytes), dtype=torch.uint8, device=x.device)
    out[:, : d // 2] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    out[:, d // 2 : d // 2 + d // block] = _scale_bytes(
        s.squeeze(-1).contiguous(), scale
    ).to(torch.uint8)
    return out


def unpack_rows(rows: torch.Tensor, d: int, block: int, scale: str) -> torch.Tensor:
    """``[..., row_bytes]`` uint8 packed rows -> ``[..., d]`` BF16 values."""
    b = rows.to(torch.int32)
    codes = torch.stack((b[..., : d // 2] & 0xF, b[..., : d // 2] >> 4), dim=-1)
    q = _e2m1_values(codes.flatten(-2))
    s = _scale_values(b[..., d // 2 : d // 2 + d // block], scale)
    return (q.unflatten(-1, (d // block, block)) * s.unsqueeze(-1)).flatten(-2).to(
        torch.bfloat16
    )


def pack_comp(x: torch.Tensor) -> torch.Tensor:
    """Packed latents ([N, COMP_PACKED_BYTES]; stored into :func:`latent_rows`)."""
    return pack_rows(x, COMP_BLOCK, COMP_SCALE, COMP_PACKED_BYTES)


def pack_idxk(x: torch.Tensor) -> torch.Tensor:
    return pack_rows(x, IDXK_BLOCK, IDXK_SCALE, IDXK_ROW_BYTES)


def unpack_like(cache: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """BF16 view of rows gathered from ``cache`` (identity for a BF16 cache)."""
    if not is_packed(cache):
        return rows
    if cache.shape[-1] == COMP_ROW_BYTES:
        return unpack_rows(rows, COMP_D, COMP_BLOCK, COMP_SCALE)
    assert cache.shape[-1] == IDXK_ROW_BYTES
    return unpack_rows(rows, IDXK_D, IDXK_BLOCK, IDXK_SCALE)


# ---------------------------------------------------------------------------
# Triton
# ---------------------------------------------------------------------------


@triton.jit
def e2m1_values_tl(codes):
    """fp32 values of E2M1 codes (int32 tile), -0.0 kept."""
    mag = codes & 7
    e = mag >> 1
    m = mag & 1
    normal = ((e - 1 + 127) << 23) | (m << 22)
    half = tl.where(m == 1, 126 << 23, 0)
    bits = tl.where(e == 0, half, normal) | ((codes >> 3) << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def scale_values_tl(b, E4M3: tl.constexpr):
    """fp32 scales of scale bytes (int32 tile)."""
    if E4M3:
        e = b >> 3
        m = b & 7
        normal = (((e - 7 + 127) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        return tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)  # 2**-9
    else:
        return (b << 23).to(tl.float32, bitcast=True)


@triton.jit
def load_packed_rows(ptr, off, ok, D: tl.constexpr, BLOCK: tl.constexpr,
                     E4M3: tl.constexpr):
    """``[N, D]`` BF16 values of the packed rows at byte offsets ``off`` [N]
    (int64, 16-byte aligned) of the uint8 cache ``ptr``; rows with ``~ok``
    read as zeros. Every code and scale byte is loaded once (contiguous,
    vectorised loads); the two nibbles of a byte are interleaved back with
    ``tl.join`` and each block is multiplied by its broadcast scale.

    Decoding feeds ``tl.dot`` exactly in the index scores, but not in the
    attention kernels (one output element in ~40k came out one BF16 ulp
    apart): those attend over :func:`unpack_selected` instead."""
    n: tl.constexpr = off.shape[0]
    off = tl.multiple_of(off, 16)
    b = tl.load(ptr + off[:, None] + tl.arange(0, D // 2)[None, :], mask=ok[:, None],
                other=0).to(tl.int32)
    sb = tl.load(ptr + off[:, None] + (D // 2 + tl.arange(0, D // BLOCK))[None, :],
                 mask=ok[:, None], other=0).to(tl.int32)
    q = tl.join(e2m1_values_tl(b & 0xF), e2m1_values_tl(b >> 4))  # [n, D // 2, 2]
    v = tl.reshape(q, (n, D // BLOCK, BLOCK)) * scale_values_tl(sb, E4M3)[:, :, None]
    return tl.reshape(v, (n, D)).to(tl.bfloat16)


@triton.jit(do_not_specialize=["k_sel"])
def _unpack_selected_kernel(cache_ptr, bt_ptr, local_ptr, req_ptr, out_ptr, out_local_ptr,
                            ident_ptr, k_sel, stride_cb, stride_ce, stride_bt, stride_li,
                            E: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                            E4M3: tl.constexpr, ROWS: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 0:
        tl.store(ident_ptr + t, t.to(tl.int32))
    r = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    inside = r < k_sel
    j = tl.load(local_ptr + t * stride_li + r, mask=inside, other=-1)
    ok = j >= 0
    j64 = tl.maximum(j, 0).to(tl.int64)
    req = tl.load(req_ptr + t).to(tl.int64)
    blk = tl.load(bt_ptr + req * stride_bt + j64 // E, mask=ok, other=0)
    v = load_packed_rows(cache_ptr, blk.to(tl.int64) * stride_cb + (j64 % E) * stride_ce, ok,
                         D, BLOCK, E4M3)
    row = t * k_sel + r
    tl.store(out_ptr + row[:, None] * D + tl.arange(0, D)[None, :], v, mask=inside[:, None])
    tl.store(out_local_ptr + row, tl.where(ok, r, -1), mask=inside)


def unpack_selected(cache: torch.Tensor, block_table: torch.Tensor, local: torch.Tensor,
                    req: torch.Tensor):
    """The selected rows of a packed compressed-KV cache as a BF16 cache of
    their own, for the attention kernels' plain path.

    ``local`` [n, K] are each query's compressed positions (``-1`` unused) in
    request ``req[t]``'s pages. Returns ``(rows, table, local', req')``: query
    ``t``'s K rows decoded into page ``t`` of ``rows`` [n, K, D] BF16, an
    identity block table and request map (``arange(n)``), and ``local'`` the
    slot of each used entry (``-1`` kept). The kernels then load the very bits
    a BF16 cache would hold, through their unchanged loads."""
    n, k_sel = local.shape
    assert k_sel > 0
    rows = torch.empty((n, k_sel, COMP_D), dtype=torch.bfloat16, device=cache.device)
    out_local = torch.empty((n, k_sel), dtype=torch.int32, device=cache.device)
    ident = torch.empty(n, dtype=torch.int32, device=cache.device)
    if n:
        rows_per_program = 8
        _unpack_selected_kernel[(n, triton.cdiv(k_sel, rows_per_program))](
            cache, block_table, local, req, rows, out_local, ident, k_sel,
            cache.stride(0), cache.stride(1), block_table.stride(0), local.stride(0),
            E=cache.shape[1], D=COMP_D, BLOCK=COMP_BLOCK, E4M3=True,
            ROWS=rows_per_program, num_warps=4,
        )
    return rows, ident[:, None], out_local, ident


@triton.jit
def _e2m1_codes_tl(q):
    bits = q.to(tl.int32, bitcast=True)
    sign = (bits >> 31) & 1
    a = tl.abs(q)
    abits = a.to(tl.int32, bitcast=True)
    exp = ((abits >> 23) & 0xFF) - 127
    mant = (abits >> 22) & 1
    mag = tl.where(a < 1.0, (a * 2.0).to(tl.int32), ((exp + 1) << 1) | mant)
    return (sign << 3) | mag


@triton.jit
def _scale_bytes_tl(s, E4M3: tl.constexpr):
    bits = s.to(tl.int32, bitcast=True)
    e32 = (bits >> 23) & 0xFF
    if E4M3:
        denormal = (s * 512.0).to(tl.int32)
        normal = ((e32 - 120) << 3) | ((bits >> 20) & 7)
        return tl.where(e32 < 121, denormal, normal)
    else:
        return e32


@triton.jit
def store_packed_row(ptr, off, x, D: tl.constexpr, BLOCK: tl.constexpr,
                     E4M3: tl.constexpr):
    """Quantize the fp32 row ``x`` [D] exactly as ``qdq_fp4_tile`` and store
    its packed form at byte offset ``off`` of the uint8 cache ``ptr``."""
    xb = tl.reshape(x, (D // BLOCK, BLOCK))
    amax = tl.max(tl.abs(xb), axis=1)
    if E4M3:
        a = tl.div_rn(tl.maximum(amax, 0.01171875), 6.0)
        s = round_e4m3_tl(tl.minimum(a, 448.0))
    else:
        s = _pow2_ceil_tl(tl.maximum(amax, 7.052966104933725e-38) * 0.1666666716337204)
    v = tl.minimum(tl.maximum(tl.div_rn(xb, s[:, None]), -6.0), 6.0)
    codes = tl.reshape(_e2m1_codes_tl(round_e2m1_tl(v)), (D // 2, 2))
    lo, hi = tl.split(codes)
    j = tl.arange(0, D // 2)
    tl.store(ptr + off + j, (lo | (hi << 4)).to(tl.uint8))
    k = tl.arange(0, D // BLOCK)
    tl.store(ptr + off + D // 2 + k, _scale_bytes_tl(s, E4M3).to(tl.uint8))
