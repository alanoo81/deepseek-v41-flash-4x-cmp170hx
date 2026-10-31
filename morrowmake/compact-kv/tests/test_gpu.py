"""GPU check of the packed caches: every kernel that reads or writes them gives the
same bits as with the BF16 caches holding the reference's rounded values."""
import torch
from vllm.models.deepseek_v41.nvidia.ops import sm80_kv_pack as kp
from vllm.models.deepseek_v41.nvidia.ops.sm80_act_quant import qdq_fp4, qdq_fp8
from vllm.models.deepseek_v41.nvidia.ops import sm80_sparse_attn, sm80_attn_fused, sm80_index_scores
from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core
from vllm.triton_utils import tl, triton

dev = "cuda"
torch.manual_seed(0)
ok_all = True


def same(name, a, b):
    global ok_all
    a, b = a.contiguous(), b.contiguous()
    eq = a.view(torch.uint8).equal(b.view(torch.uint8))
    ok_all &= eq
    print(f"{name:42s} {'IDENTICAL' if eq else 'DIFFERENT'}"
          + ("" if eq else f"  max|d|={(a.float()-b.float()).abs().max().item():.3g}"))


def raw(*shape):
    x = torch.randn(*shape) * 10 ** torch.empty(*shape[:-1], 1).uniform_(-3, 1)
    return x.to(torch.bfloat16)


# 1. store kernel (Triton) bytes == torch pack bytes
@triton.jit
def _store(x_ptr, cache_ptr, ROW: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, E4M3: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    x = tl.load(x_ptr + t * D + tl.arange(0, D)).to(tl.float32)
    kp.store_packed_row(cache_ptr, t * ROW, x, D, BLOCK, E4M3)


for d, blk, sc, rb in ((512, 16, "e4m3", 288), (128, 32, "e8m0", 128)):
    x = raw(4096, d)
    x[0] = 0
    x[1, ::2] = -0.0
    c = torch.zeros(4096, rb, dtype=torch.uint8, device=dev)
    _store[(4096,)](x.to(dev), c, rb, d, blk, sc == "e4m3")
    same(f"Triton store {d} {sc} == torch", c.cpu(), kp.pack_rows(x, blk, sc, rb))
    same(f"unpack == reference qdq {d}", kp.unpack_rows(c, d, blk, sc).cpu(), qdq_fp4(x, blk, sc))

# 1c. latent and index key written by Triton into the same 512-byte rows, in both orders
@triton.jit
def _store_both(lat_ptr, key_ptr, cache_ptr, KEY_FIRST: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    lat = tl.load(lat_ptr + t * 512 + tl.arange(0, 512)).to(tl.float32)
    key = tl.load(key_ptr + t * 128 + tl.arange(0, 128)).to(tl.float32)
    if KEY_FIRST:
        kp.store_packed_row(cache_ptr + kp.IDXK_OFFSET, t * 512, key, 128, 32, False)
    kp.store_packed_row(cache_ptr, t * 512, lat, 512, 16, True)
    if not KEY_FIRST:
        kp.store_packed_row(cache_ptr + kp.IDXK_OFFSET, t * 512, key, 128, 32, False)


lat, key = raw(1024, 512), raw(1024, 128)
want = torch.zeros(1024, 512, dtype=torch.uint8)
kp.latent_rows(want)[:] = kp.pack_comp(lat)
kp.index_key_rows(want)[:] = kp.pack_idxk(key)
for key_first in (True, False):
    c = torch.full((1024, 512), 0, dtype=torch.uint8, device=dev)
    _store_both[(1024,)](lat.to(dev), key.to(dev), c, key_first)
    same(f"shared latent+key row (key first={key_first})", c.cpu(), want)

# caches holding the same values, plain and packed (packed keys inside the latent rows)
B, E = 6, 64
comp_raw = raw(B * E, 512)
comp_bf16 = qdq_fp4(comp_raw, 16, "e4m3").view(B, E, 512).to(dev)
comp_pk = torch.zeros(B, E, 512, dtype=torch.uint8)
kp.latent_rows(comp_pk)[:] = kp.pack_comp(comp_raw).view(B, E, -1)
kp.index_key_rows(comp_pk)[:] = kp.pack_idxk(raw(B * E, 128)).view(B, E, -1)
comp_pk = comp_pk.to(dev)
swa = qdq_fp8(raw(4 * 32, 512), 32).view(4, 32, 512).to(dev)
n, H = 5, 16
q = raw(n, H, 512).to(dev)
pos = torch.tensor([100, 150, 191, 30, 7], dtype=torch.int64, device=dev)
freqs = (1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))).to(dev)
req = torch.tensor([0, 0, 1, 1, 0], dtype=torch.int32, device=dev)
bt = torch.tensor([[3, 0, 5], [1, 4, 2]], dtype=torch.int32, device=dev)
swa_idx = torch.full((n, 128), -1, dtype=torch.int32, device=dev)
for i in range(n):
    k = 20 + 10 * i
    swa_idx[i, :k] = torch.randperm(128)[:k].to(dev)
comp_local = torch.full((n, 512), -1, dtype=torch.int32, device=dev)
for i in range(n):
    k = [60, 150, 192, 10, 0][i]
    comp_local[i, :k] = torch.sort(torch.randperm(192)[:k]).values.to(dev)
sink = torch.randn(H, device=dev)
scale = 512 ** -0.5

# 1b. Triton load path on GPU == plain cache
@triton.jit
def _loadk(cache_ptr, out_ptr, N, ROW: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, E4M3: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0) * BN + tl.arange(0, BN)
    ok = r < N
    v = kp.load_packed_rows(cache_ptr, r.to(tl.int64) * ROW, ok, D, BLOCK, E4M3)
    tl.store(out_ptr + r[:, None] * D + tl.arange(0, D)[None, :], v, mask=ok[:, None])

out = torch.empty(B * E, 512, dtype=torch.bfloat16, device=dev)
_loadk[(triton.cdiv(B * E, 16),)](comp_pk.view(-1, 512), out, B * E, 512, 512, 16, True, 16)
same("Triton GPU load comp == bf16 cache", out, comp_bf16.view(-1, 512))

# determinism controls (same bf16 inputs twice)
a1 = sm80_sparse_attn.sparse_attention_triton(q, swa, swa_idx, comp_bf16, bt, comp_local, req, sink, scale)
a2 = sm80_sparse_attn.sparse_attention_triton(q, swa, swa_idx, comp_bf16, bt, comp_local, req, sink, scale)
same("A/A control sm80_sparse_attn bf16", a1, a2)
a1 = sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_bf16, bt, comp_local, req, sink, scale)
a2 = sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_bf16, bt, comp_local, req, sink, scale)
same("A/A control sm80_attn_fused bf16", a1, a2)
a3 = sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_bf16.clone(), bt, comp_local, req, sink, scale)
same("A/A control sm80_attn_fused bf16 (copy)", a1, a3)

# 2. sparse attention (core path)
a = sm80_sparse_attn.sparse_attention_triton(q, swa, swa_idx, comp_bf16, bt, comp_local, req, sink, scale)
b = sm80_sparse_attn.sparse_attention_triton(q, swa, swa_idx, comp_pk, bt, comp_local, req, sink, scale)
same("sm80_sparse_attn", a, b)

# 3. fused attention (decode path), pre- and post-RoPE
for post in (False, True):
    a = sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_bf16, bt, comp_local,
                                                req, sink, scale, post_rope=post)
    b = sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_pk, bt, comp_local,
                                                req, sink, scale, post_rope=post)
    same(f"sm80_attn_fused post_rope={post}", a, b)
oa = torch.empty(n + 3, H, 512, dtype=torch.bfloat16, device=dev)
ob = torch.empty_like(oa)
sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_bf16, bt, comp_local, req,
                                        sink, scale, post_rope=True, final_out=oa[:n])
sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_pk, bt, comp_local, req,
                                        sink, scale, post_rope=True, final_out=ob[:n])
same("sm80_attn_fused final_out epilogue", oa[:n], ob[:n])

# 4. index scores
EI = 64
idxk_raw = raw(B * EI, 128)
idxk_bf16 = qdq_fp4(idxk_raw, 32, "e8m0").view(B, EI, 128).to(dev)
rows = torch.zeros(B, EI, 512, dtype=torch.uint8)
kp.latent_rows(rows)[:] = kp.pack_comp(raw(B * EI, 512)).view(B, EI, -1)
kp.index_key_rows(rows)[:] = kp.pack_idxk(idxk_raw).view(B, EI, -1)
idxk_pk = kp.index_key_rows(rows.to(dev))
iq = qdq_fp4(raw(n, 32, 128), 32, "e8m0").to(dev)
iw = raw(n, 32).to(dev)
nvis = torch.tensor([100, 192, 1, 150, 0], dtype=torch.int64, device=dev)
a = sm80_index_scores.index_scores_triton(iq, iw, nvis, req, idxk_bf16, bt, 192, 8)
b = sm80_index_scores.index_scores_triton(iq, iw, nvis, req, idxk_pk, bt, 192, 8)
same("sm80_index_scores", a, b)

# 5. torch fallbacks (gather unpacks)
a = core.sparse_attention_torch(q.cpu(), swa.cpu(), swa_idx.cpu(), comp_bf16.cpu(), bt.cpu(),
                                comp_local.cpu(), req.cpu(), sink.cpu(), scale)
b = core.sparse_attention_torch(q.cpu(), swa.cpu(), swa_idx.cpu(), comp_pk.cpu(), bt.cpu(),
                                comp_local.cpu(), req.cpu(), sink.cpu(), scale)
same("torch fallback sparse_attention", a, b)
a = core.index_scores_torch(iq.cpu(), iw.cpu(), nvis.cpu(), req.cpu(), idxk_bf16.cpu(), bt.cpu(), 192, 8)
b = core.index_scores_torch(iq.cpu(), iw.cpu(), nvis.cpu(), req.cpu(), idxk_pk.cpu(), bt.cpu(), 192, 8)
same("torch fallback index_scores", a, b)
print("RESULT", "OK" if ok_all else "FAILED")
