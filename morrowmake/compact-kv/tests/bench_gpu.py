"""Kernel timings, BF16 caches vs packed caches, at served shapes (one GPU, 16 local heads)."""
import sys

import torch
from vllm_sm80.models.deepseek_v41.ops import sm80_kv_pack as kp
from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import qdq_fp4, qdq_fp8
from vllm_sm80.models.deepseek_v41.ops import sm80_sparse_attn, sm80_attn_fused, sm80_index_scores

dev = "cuda"
torch.manual_seed(0)
H, D, ID, IH = 16, 512, 128, 32


def timeit(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000  # µs


def raw(*shape):
    return (torch.randn(*shape, device=dev) * 0.5).to(torch.bfloat16)


rows = []


def report(name, fa, fb):
    ta, tb = timeit(fa), timeit(fb)
    rows.append((name, ta, tb))
    print(f"{name:44s} bf16 {ta:9.1f} µs   compact {tb:9.1f} µs   {100 * (tb / ta - 1):+6.1f} %",
          flush=True)


# ---- attention (comp rows: 64 entries / 32 KiB page in both layouts)
n_pos = 131072
xc = raw(n_pos, D)
comp_bf16 = torch.zeros(n_pos // 32 + 1, 32, D, dtype=torch.bfloat16, device=dev)
comp_bf16[1:] = qdq_fp4(xc, 16, "e4m3").view(-1, 32, D)
comp_pk = torch.zeros(n_pos // 64 + 1, 64, 512, dtype=torch.uint8, device=dev)
kp.latent_rows(comp_pk)[1:] = kp.pack_comp(xc).view(-1, 64, kp.COMP_PACKED_BYTES)
cbt_bf = (torch.arange(n_pos // 32, device=dev, dtype=torch.int32) + 1)[None]
cbt_pk = (torch.arange(n_pos // 64, device=dev, dtype=torch.int32) + 1)[None]
swa = qdq_fp8(raw(8 * 32, D), 32).view(8, 32, D)
freqs = (1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))).to(dev)
sink = torch.randn(H, device=dev)
scale = D ** -0.5

for n in (1, 8, 64):
    q = raw(n, H, D)
    pos = torch.full((n,), n_pos * 2 - 1, dtype=torch.int64, device=dev)
    req = torch.zeros(n, dtype=torch.int32, device=dev)
    swa_idx = torch.arange(128, device=dev, dtype=torch.int32).repeat(n, 1) % (8 * 32)
    local = torch.stack([torch.sort(torch.randperm(n_pos, device=dev)[:512]).values
                         for _ in range(n)]).to(torch.int32)
    out_a = torch.empty(n, H, D, dtype=torch.bfloat16, device=dev)
    out_b = torch.empty_like(out_a)
    report(f"fused attention (decode) n={n}",
           lambda: sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_bf16, cbt_bf,
                                                           local, req, sink, scale, post_rope=True,
                                                           final_out=out_a),
           lambda: sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_pk, cbt_pk,
                                                           local, req, sink, scale, post_rope=True,
                                                           final_out=out_b))
    a = sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_bf16, cbt_bf, local, req,
                                                sink, scale, post_rope=True)
    b = sm80_attn_fused.sparse_attention_triton(q, pos, freqs, swa, swa_idx, comp_pk, cbt_pk, local, req,
                                                sink, scale, post_rope=True)
    print("   fused identical:", a.view(torch.int16).equal(b.view(torch.int16)))

for n in (256, 2048):
    q = raw(n, H, D)
    req = torch.zeros(n, dtype=torch.int32, device=dev)
    swa_idx = torch.arange(128, device=dev, dtype=torch.int32).repeat(n, 1) % (8 * 32)
    local = torch.sort(torch.randint(0, n_pos, (n, 512), device=dev), dim=1).values.to(torch.int32)
    report(f"sparse attention (prefill) n={n}",
           lambda: sm80_sparse_attn.sparse_attention_triton(q, swa, swa_idx, comp_bf16, cbt_bf, local, req,
                                                            sink, scale),
           lambda: sm80_sparse_attn.sparse_attention_triton(q, swa, swa_idx, comp_pk, cbt_pk, local, req,
                                                            sink, scale))
    a = sm80_sparse_attn.sparse_attention_triton(q, swa, swa_idx, comp_bf16, cbt_bf, local, req, sink, scale)
    b = sm80_sparse_attn.sparse_attention_triton(q, swa, swa_idx, comp_pk, cbt_pk, local, req, sink, scale)
    print("   sparse identical:", a.view(torch.int16).equal(b.view(torch.int16)))

# ---- index scores (idxk rows: 128 entries bf16 page, 256 packed)
for n, n_cols in ((1, 131072), (8, 131072), (1, 16384), (2048, 16384)):
    k_bf = torch.zeros(n_cols // 128 + 1, 128, ID, dtype=torch.bfloat16, device=dev)
    xk = raw(n_cols, ID)
    k_bf[1:] = qdq_fp4(xk, 32, "e8m0").view(-1, 128, ID)
    kc = torch.zeros(n_cols // 64 + 1, 64, 512, dtype=torch.uint8, device=dev)
    k_pk = kp.index_key_rows(kc)
    k_pk[1:] = kp.pack_idxk(xk).view(-1, 64, 128)
    bt_bf = (torch.arange(n_cols // 128, device=dev, dtype=torch.int32) + 1)[None]
    bt_pk = (torch.arange(n_cols // 64, device=dev, dtype=torch.int32) + 1)[None]
    iq = qdq_fp4(raw(n, IH, ID), 32, "e8m0")
    iw = raw(n, IH)
    nvis = torch.full((n,), n_cols, dtype=torch.int64, device=dev)
    req = torch.zeros(n, dtype=torch.int32, device=dev)
    report(f"index scores n={n} keys={n_cols}",
           lambda: sm80_index_scores.index_scores_triton(iq, iw, nvis, req, k_bf, bt_bf, n_cols, 8),
           lambda: sm80_index_scores.index_scores_triton(iq, iw, nvis, req, k_pk, bt_pk, n_cols, 8))
    a = sm80_index_scores.index_scores_triton(iq, iw, nvis, req, k_bf, bt_bf, n_cols, 8)
    b = sm80_index_scores.index_scores_triton(iq, iw, nvis, req, k_pk, bt_pk, n_cols, 8)
    print("   index identical:", a.equal(b))
print("done", sys.argv[1:] if len(sys.argv) > 1 else "")
