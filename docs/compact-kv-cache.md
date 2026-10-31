# A lossless compact KV cache for Morrowmake's sm_80 engine

Morrowmake's engine keeps the compressed attention latent and the indexer keys of DeepSeek-V4.1 as BF16 rows.
The values in those rows have already been rounded by the model's reference code to 4-bit floats with one
scale per small block, so most of each BF16 row is redundant. This change stores the 4-bit codes and the
scale bytes instead and decodes them back to the **same bits** when a kernel reads them.

| | BF16 (Morrowmake 1.0.0) | Compact v1 | **Compact v1.5 (served)** |
|---|---:|---:|---:|
| Compressed latent row | 1,024 B | 512 B | **512 B, shared** |
| Index-key row | 256 B | 128 B | **inside the latent row** |
| KV pool, no DSpark, 262,144-token context | 1,604,950 tokens | 2,738,502 | **3,189,000 (x1.99)** |
| KV pool with DSpark | (not measured) | 1,166,235 | **1,360,109** |
| Prompt log-probabilities vs BF16, 1.39 M tokens | reference | bit-identical | **bit-identical** |
| Single-stream decode vs BF16 | reference | -5 to -9 % | **-2 to -5 %** |
| Prefill vs BF16 | reference | -10 to -35 % | **-6 to -10 %** |

The patch is [`morrowmake/compact-kv/`](../morrowmake/compact-kv/). It changes nothing unless the
environment sets `VLLM_SM80_COMPACT_KV=1`.

## 1. Why it can be lossless

DeepSeek-V4.1's reference code rounds what it caches (Morrowmake reproduces this in `sm80_attn_core.py`):

- the sliding-window rows: FP8 E4M3, one UE8M0 scale per 32 values;
- the **compressed latent** (512 values, RoPE applied): FP4 E2M1, one **E4M3** scale per 16 values (the NVFP4 layout);
- the **indexer keys** (128 values, RoPE applied): FP4 E2M1, one **E8M0** (power-of-two) scale per 32 values (MXFP4).

Morrowmake keeps the rounded values in BF16 rows. Each value is `bf16(q x s)` with `q` on the E2M1 grid
(0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives) and `s` an E4M3 or E8M0 number. The product has at most
5 significant bits, so it is exact in FP32 and exactly representable in BF16. Storing `q` as a 4-bit code and
`s` as one byte, then recomputing `q x s`, gives back the BF16 row bit for bit, including `-0.0`.

The packing itself (`pack_rows` in torch, `store_packed_row` in Triton) uses the same arithmetic as the
reference's `fp4_qdq`: block maximum, scale, division, clamp, rounding to E2M1. The codes are then exactly the
ones the BF16 path would have multiplied back.

The sliding-window cache (FP8 values, 128 tokens per request) is left as it is: it is small and bounded.

## 2. Layout (v1.5)

The kv-source layers (2, 8, 14 and 20) own both a compressed-latent cache and an index-key cache. Both have
the same compression ratio, so they cover the same compressed positions. v1.5 puts both in one row:

```
compressed-latent row, 512 bytes
  0 ..255   latent: 512 E2M1 codes, value 2j in the low nibble of byte j, sign in bit 3
256 ..287   latent: 32 E4M3 scale bytes (one per 16 values, no sign bit)
288 ..351   index key: 128 E2M1 codes
352 ..355   index key: 4 E8M0 scale bytes (one per 32 values)
356 ..511   free
```

- The index-key cache's `get_kv_cache_spec` returns `None` in compact mode, so vLLM allocates nothing for it.
  The indexer reads `index_key_rows(comp) = comp[..., 288:416]`, a strided view of the latent cache, through
  the latent cache's block table and slot mapping.
- The kernels already take the cache's strides, so they read the view unchanged. Offset 288 keeps the 16-byte
  alignment their vector loads need.
- Writers are disjoint: the latent writers touch bytes 0-287 only (`latent_rows`), the key writers bytes
  288-415. The unit test writes both into the same rows in both orders.
- v1, the first version, kept the two caches separate (512-byte latent rows, 128-byte key rows). v1.5 removes
  the key cache's 320 bytes per token per card.

Per token and per card, the latent and key caches drop from 3,200 bytes (BF16) to 1,280 bytes. Everything else
(window rows, compression state rings) is unchanged.

## 3. Read paths, and why attention reads through a BF16 copy

**Index scores** decode the packed key tile in registers right before `tl.dot`. That gives bit-identical
scores (tested on GPU at served shapes).

**Attention does not.** Decoding the latent tile in registers before the attention kernels' `tl.dot` changed
one output element in about 40,000 by one BF16 ulp. The operand bits were identical; the compiled arithmetic
was not, presumably because a register-built operand takes a different path into the matrix instructions than
one loaded from memory.

So the attention kernels are not modified. `unpack_selected` decodes each query's selected rows (512 per
query) into a small BF16 buffer, page `t` for query `t`, with an identity block table and request map. The
**original** Morrowmake kernels then run on it. They load the same bits with the same loads as with a BF16
cache, which makes the result bit-identical by construction rather than by test. Prefill decodes 128 queries
at a time, at most 64 MiB.

The first version used another fix: the decoded tile went through a per-program scratch buffer and was read
back with the BF16 path's masked load. It was exact too, but slower.

## 4. Kernel optimisation (v1 to v2)

The first decoder read every code byte twice (once per nibble) and every scale byte 16 times, one element at a
time. v2 loads each code byte and each scale byte once with contiguous vector loads, splits the nibbles,
interleaves them back with `tl.join` and multiplies each block by its broadcast scale.

Kernel timings on one CMP 170HX, BF16 cache vs compact cache, served shapes (16 local heads, top-512
selection, 128-slot window, keys of a 131,072-position context):

| Kernel | BF16 | Compact v1 | **Compact v2 / v1.5** |
|---|---:|---:|---:|
| Fused attention, decode, 1 query | 37.7 µs | 51.0 µs (+35 %) | **42.0 µs (+11 %)** |
| Fused attention, decode, 8 queries | 43.9 µs | 60.1 µs (+37 %) | **53.7 µs (+22 %)** |
| Sparse attention, prefill, 2,048 queries | 1,761 µs | 4,499 µs (+155 %) | **3,165 µs (+80 %)** |
| Index scores, 1 query, 131,072 keys | 57.3 µs | 172.0 µs (+200 %) | **65.4 µs (+14 %)** |
| Index scores, 2,048 queries, 16,384 keys | 11,868-12,053 µs | 33,188 µs (+175 %) | **14,819-14,895 µs (+23 to +26 %)** |

Every compact result was compared with the BF16 one at these shapes: all identical.

What remains is mostly the prefill attention, where each query decodes its own 512 rows although neighbouring
queries select nearly the same ones. Removing that redundancy needs a de-duplication pass on the GPU; we have
not done it.

## 5. Validation

**GPU unit test** ([`compact-kv/tests/test_gpu.py`](../morrowmake/compact-kv/tests/test_gpu.py), run in the
engine image with the patched files mounted). Every line must print `IDENTICAL`:

- Triton store bytes equal torch pack bytes (latent and key formats), unpacked rows equal the reference qdq;
- latent and key written into the same 512-byte rows, key first and key last;
- Triton GPU load equals the BF16 cache;
- A/A controls of both attention kernels on BF16 inputs (the kernels are deterministic);
- `sm80_sparse_attn`, `sm80_attn_fused` (pre- and post-RoPE, with the `final_out` epilogue), `sm80_index_scores`
  and the torch fallbacks: packed result equals BF16 result.

**Served model.** Morrowmake with layers 18-22 grafted, BF16 cache against compact cache, same frozen windows
([`tests/nll_eval.py`](../tests/nll_eval.py), checked with [`tests/identical.py`](../tests/identical.py)):

| Set | Tokens | Result |
|---|---:|---|
| 192 windows of 1,024 tokens (French, English, code) | 196,416 | every prompt log-probability identical |
| 1,200 windows of 40 tokens | 46,800 | identical |
| 19 windows of 32k tokens (prose and code) | 622,573 | identical |
| 16 windows of 32k tokens of code | 524,272 | identical |
| Self-check: 768 generated tokens | 768 | identical decode log-probabilities |

Data: `data/nll/{,long/,codex/}mm-k3*.json.gz` against `mm-kvc15*.json.gz`, `data/nll/sc-mm-k3.json.gz` against
`sc-mm-kvc15.json.gz`. v1 gave the same identities (`mm-kvc.json.gz`, `codex/mm-kvc.json.gz`, `sc-mm-kvc.json.gz`;
the v2 runs are in our logs only).

## 6. Speed, one stream, no DSpark

Our speed bench ([`tests/speed.py`](../tests/speed.py): unique prompts, warm-up discarded, natural output of at
most 256 tokens), decode in tok/s / prefill in tok/s:

| Prompt | BF16 | Compact v1 | **Compact v1.5** |
|---|---|---|---|
| 3.5k tokens | 69.3 / 3,815 | 64.1 / 3,439 | **66.7 / 3,594** |
| 33k | 66.3 / 4,104 | 61.9 / 3,455 | **64.8 / 3,834** |
| 66k | 64.4 / 3,939 | 60.9 / 3,142 | **63.1 / 3,654** |
| 132k | 62.9 / 3,600 | 58.3 / 2,624 | **60.7 / 3,304** |
| 259k | 59.8 / 3,081 | 54.4 / 1,993 | **56.9 / 2,776** |
| Time to first token at 259k | 84 s | 130 s | **93 s** |

Data: `data/bench/speed/speed-mm-k3.json` (BF16), `speed-mm-kvc.json` (v1), `speed-mm-kvc15.json`.

## 7. Limits

- Morrowmake's sm_80 engine only, release 1.0.0 (engine `a0c57b52`). The patch touches seven files of
  `vllm/models/deepseek_v41/nvidia/`.
- It saves memory on the compressed-attention caches only; weights and activations are unchanged.
- Decode still costs 2-5 % and prefill 6-10 % against BF16 rows.
- The GPU unit test and the served-model identity are our evidence. No independent review yet.
