# Comparison with the Morrowmake recipe

Two public recipes serve DeepSeek-V4.1-Flash-1017 on 4x CMP 170HX:

- [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe), release 1.0.0 at `7f9023cd`. It runs on Morrowmake's vLLM fork [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) at `a0c57b52`.
- [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx), release 1.2.0 at `2ad5741ab1`. This repository builds on it.

This page compares Morrowmake's recipe with our configuration (GNmac plus the changes in this repository). We set out to use Morrowmake's recipe, and switched to GNmac's mainly because Morrowmake's download did not fit on our disk.

**What this comparison is and is not.** We did **not** run Morrowmake's recipe. Everything about it below comes from its source, its documentation and its published results, which we read in a read-only research pass. A second, critical pass then re-checked the claims, and it corrected several, for example that Morrowmake's `bfloat16` KV cache is "not quantized" (it holds the reference's exact rounded values).

Our own numbers come from measurements on our machine ([results.md](results.md)). No number in this page comes from running both stacks on the same machine. Read it as a map of the differences, not as a benchmark.

## Summary

| | Morrowmake 1.0.0 | Ours (GNmac 1.2.0 + this repo) | Which side it favours |
|---|---|---|---|
| Routed experts | LibertAI EXL3 2.9 bpw: layers 18-22 at 2 bits, the rest at 3 bits | Same pack, but layers 18-22 grafted to 3 bits from the 3.5 bpw pack | **Ours: measured** -1.80 % perplexity overall and -4.45 % on French |
| KV cache | `bfloat16` rows holding the reference's exact quantize-dequantize values | `fp8_ds_mla` in the V4 record on sm_80. Coarser than the reference on the window, finer elsewhere | **Morrowmake, likely**; not measured by anyone |
| Indexer | Reference rounding reproduced (FP4 queries and keys, BF16 score rounding, canonical tie-breaking) | FP8 index keys, one fp32 scale per token | Morrowmake, likely; not measured |
| All-reduce | Decode: FP32, rounded once. Prefill MoE: BF16 at each ring hop (a switch makes it exact; not in the measured config) | Decode: BF16 adds in a ring (up to 3 roundings). Prefill: FP32, rounded once | Mixed; small |
| Dense FP8 matmuls | W8A16 (activations not quantized, unlike the reference) | Same | Equivalent |
| Engram | Exact dequantization, matches the reference bit for bit | Same | Equivalent |
| Router gate | Official FP32 bias, FP64 sums | Official FP32 bias, BF16 weight with FP32 accumulation | Equivalent to slightly Morrowmake |
| Speculative decoding | DSpark, up to 7 drafts, adaptive | Off (lossless either way; off for KV room) | Equivalent for quality |
| Verification approach | Per-module bit-exact tests against DeepSeek's own reference code; HumanEval/GSM8K against its own all-switches-off build | Paired NLL A/B between weight configurations, A-B-A across restarts; copy and determinism tests | Different strengths. Neither stack was ever compared with the unquantized model |
| Cold prefill (published / ours) | 3,818 tok/s (2,048 tokens), 3,808 (32,768 tokens) | 2,514 tok/s (24-38k tokens) | **Morrowmake, about 1.5x** (different machines and methods) |
| KV pool at 262k | 1,247,406 tokens (with drafter, 0.95) | 2,717,420 tokens (no drafter, 0.98) | Ours, through configuration choices, not the engine |
| Disk | 671.5 GiB of downloads, 680 GiB required by its preflight check | about 428 GiB | Ours |
| P2P | Requires a driver build that passes its own content check. Its docs name a TRAP31 patch | Works with our driver, which has no TRAP31 patch | Untested for Morrowmake on our driver |

## 1. Quality

### 1.1 Where Morrowmake is likely closer to the original: KV and indexer rounding

**DeepSeek's reference** (`inference/model.py` in the official repository at `1bae0e10`) rounds three caches before storing them:

- the sliding-window KV goes through FP8 in blocks of 32 (UE8M0 scales);
- the compressed KV goes through FP4 in groups of 16 (E4M3 scales);
- the indexer keys and queries go through FP4 in blocks of 32.

**Morrowmake** stores its cache as `bfloat16`, but it applies exactly these quantize-dequantize steps before writing. The cache rows therefore hold the reference's rounded values, and the rounding emulation is tested bit for bit against the reference code. The reference indexer is reproduced too.

**GNmac** (and so ours) uses `--kv-cache-dtype fp8_ds_mla`, hard-coded in `serve.sh`. On sm_80 the engine writes the DeepSeek-V4 record, not the V4.1 one:

- 448 FP8 dims with one scale per 64 dims. That is **coarser** than the reference's per-32 scales on the window.
- The RoPE tail stays in BF16, the compressed KV is in FP8 instead of FP4, and the index keys are in FP8 instead of FP4. All three are **finer** than the reference.

GNmac's docs call this "the model's native format". The research pass found that this is not the case on sm_80: the V4.1 format (`nvfp4_ds_mla`) needs SM100.

**Inference:** on cache rounding, Morrowmake is closer to the reference. How much this changes the output has not been measured by Morrowmake, by GNmac or by us.

The cost of Morrowmake's choice is KV capacity. Its published pool is 1,247,406 tokens, against GNmac's published 1,395,731 in a comparable setup (both with drafter, both at 0.95).

### 1.2 Where we are measurably better: layers 18-22

Both stacks serve the LibertAI EXL3 2.9 bpw routed experts. Morrowmake pins revision `c178a012`, which the research pass found holds the same weights as `0be63c33`, the revision we pin. In that pack, the exllamav3 allocator put layers 18-22 at 2 bits.

We replaced those five layers with the 3-bit experts of LibertAI's 3.5 bpw pack and measured the effect with paired NLL ([results.md, section 8.2](results.md#82-layers-18-22-at-2-bits-k29-a-vs-3-bits-k3-b)):

| Set | Perplexity change, 1,024-token windows | 95 % CI |
|---|---:|---|
| French | -4.45 % | [-4.97, -3.93] |
| English | -0.70 % | [-0.95, -0.48] |
| Code | -0.18 % | [-0.33, -0.04] |
| All | **-1.80 %** | [-2.00, -1.60] |

The noise floor of this comparison is zero: the same weights after a restart gave bit-identical logprobs.

Morrowmake ships the published 2.9 pack with layers 18-22 at 2 bits. Its build checks the SHA-256 of the hybrid's `config.json`, index and `quantization_config.json` (`docker/prepare.sh`), so the same graft there would require editing those pinned hashes.

**Inference:** the graft would help Morrowmake by a similar amount, since the experts are the same tensors, but that is untested. The graft costs about 1.98 GiB per card. On Morrowmake, whose KV takes more memory per token, that would shrink an already smaller pool.

### 1.3 Where the two are equivalent

- **Dense FP8 matmuls.** Both dequantize the official FP8 weights exactly to BF16 and run W8A16 with FP32 accumulation. Neither quantizes the activations to FP8 as the reference does, so both are, if anything, more precise than the reference there.
- **Engram.** Both compute the FP8 x E8M0 product in FP32 and round once to BF16, which matches the reference bit for bit. Where the tables live (RAM, tmpfs or NVMe) does not change the values. On our side this holds **only once the tables are actually loaded**: see the Engram configuration mistake in [results.md, section 2](results.md#2-the-invalid-first-run-no-engram-tables).
- **Speculative decoding.** Both use greedy DSpark drafts verified by standard rejection sampling, so the output tokens match the target's. We turn DSpark off for KV capacity, not for quality.
- **Router gate bias.** Both use the official FP32 bias. LibertAI's export stores the gate in F16; Morrowmake overrides it, and GNmac takes every non-expert tensor from the official checkpoint.

### 1.4 Smaller differences

- **All-reduce rounding.** The reference sums in FP32.
  - Morrowmake's decode all-reduce accumulates in FP32 and rounds once. Its prefill MoE all-reduce goes through NCCL's BF16 ring, rounding at each hop, unless `VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE=1` with `VLLM_ROUND_ONCE_ALLREDUCE_GROUPED=1` is set. That setting was not part of Morrowmake's measured configuration.
  - GNmac's decode all-reduce adds in BF16 around a ring (up to 3 roundings per element), with no switch. Its 1.2.0 prefill sums in FP32 and rounds once.
- **Gate matmul.** Morrowmake sums in FP64 and rounds once; GNmac uses a BF16 weight with FP32 accumulation.

### 1.5 How each side verified fidelity

| | Morrowmake | GNmac | This repository |
|---|---|---|---|
| Reference used | DeepSeek's unmodified reference source, run on the official weights, module by module (RoPE, compressor, gate, Engram, attention, DSpark...) | Its own previous build | Our other weight configurations |
| End-to-end check | HumanEval 157/164 and GSM8K 1,280/1,319, against 157/164 and 1,279/1,319 for its own build with every speed switch off | Paired NLL on 411 short passages; HumanEval 154/164 and GSM8K 1,275/1,319 | Paired NLL on 3 x 64 long and 3 x 400 short frozen windows, A-B-A across restarts; copy test; determinism; decode/prefill self-check |
| Compared with the unquantized model | No | No | No |

Neither recipe, nor this repository, has measured KL or perplexity against the original model.

The routed-expert quantization is the largest deviation on all three, and it is the same pack.

For scale: the llama.cpp DeepSeek-V4.1 pull request (#30817) reports that DeepSeek's own native FP8/FP4 kernels differ from its FP32 reference by a KL of 0.015-0.026. That is the PR author's claim; we have not verified it. If it holds, small differences in KV rounding sit well below the cost of the 2- and 3-bit experts.

### 1.6 Quality verdict

- **KV and indexer rounding fidelity likely favour Morrowmake.** Inferred from code, not measured.
- **Layers 18-22 favour us**, and that difference is measured and large on French.
- **Everything else is equivalent or close to it.**

We cannot say which complete stack is closer to the original model. That would take the experiment in [section 5](#5-what-a-fair-same-machine-ab-would-need).

## 2. Performance

All Morrowmake figures are its published release numbers: measured on 2026-10-24 on its own 4x CMP 170HX at 180 W, with mainline cmpunlocker plus its minimal P2P (TRAP31) patch, Engram in RAM and DSpark on. Ours are from our machine at 200 W.

### 2.1 Cold prefill

| | Prompt length | Rate | Method | Source |
|---|---|---:|---|---|
| Morrowmake 1.0.0 | 2,048 tokens | 3,818 tok/s | prompt tokens / time to first token, median of 2 requests | Morrowmake `docs/results.md` |
| Morrowmake 1.0.0 | 32,768 tokens | 3,808 tok/s | same | same |
| Morrowmake, Engram from SSD | 2,048 tokens | 3,607 tok/s | same, one boot | same |
| Morrowmake, P2P off | 2,048 tokens | 3,804 tok/s | same, one boot | same |
| GNmac 1.2.0, published | ~24k / 34k / 38k | 2,416 tok/s | median of 9 (3 texts x 3 runs) | upstream `docs/results.md` |
| **Ours** (`k3`, no drafter) | ~24k / 34k / 38k | **2,514 tok/s** | same script as GNmac | `data/bench/v2-prefill.log` |

Both methods divide prompt tokens by time to first token, and Morrowmake's 32k figure is close in length to our 24-38k texts.

**Inference:** the gap of roughly 1.5x (3,808 vs 2,514) is real and not an artefact of prompt length. It is still a cross-machine comparison: different drivers, power caps, P2P paths and sample sizes. We have not measured it on one machine.

For long-context agentic work, prefill speed matters: re-reading a 100k-token context takes about 26 s at 3,800 tok/s and about 40 s at 2,500 tok/s (derived).

### 2.2 Decode

| tok/s, structured / code / prose | Morrowmake 1.0.0 (DSpark 7, adaptive) | GNmac 1.2.0 published (DSpark 5) | Ours, DSpark 5 | Ours, production (no drafter) |
|---|---|---|---|---|
| 1 user | 319 / 287 / 162 | 355.7 / 258.4 / 183.6 | 369.8 / 267.3 / 200.1 | 105.4 / 105.0 / 104.8 |
| 8 users, aggregate | 765 / 729 / 432 | 867.6 / 544.0 / 440.5 | 852.9 / 450.3 / 437.8 | 559.7 / 482.4 / 449.2 |

Sources: Morrowmake `docs/results.md`; upstream GNmac `docs/results.md`; `data/bench/dspark-decode.log`; `data/bench/v2-decode.log`.

**The 1-user columns are not measured the same way.**

- Morrowmake divides a run's completion tokens by its *whole* duration, time to first token included.
- GNmac's `bench/mia.py`, which we use, measures the streaming rate *after* the first token.

For the code prompt, whose answer is only about 95 tokens, that difference alone is worth roughly 20 % (inference: about 0.1 s of time to first token against about 0.36 s of decode). So Morrowmake's 1-user figures are understated next to GNmac's and ours.

At 8 users both protocols measure aggregate throughput including prefill. There Morrowmake's code figure (729) stands out against GNmac's 544 and our 450. **Inference:** the short code answers make that point prefill-heavy, which fits the prefill gap above.

Our production configuration runs without a drafter. Its single-user speed is about a third of either recipe with DSpark. At 8 users it is level on prose and lower on structured output.

### 2.3 KV capacity

| | KV pool at 262,144 context | Full-length sequences |
|---|---:|---:|
| Morrowmake 1.0.0 (bf16 KV, DSpark, 0.95) | 1,247,406 | 4.76 (derived) |
| GNmac 1.2.0 (fp8_ds_mla, DSpark, 0.95) | 1,395,731 | 5.3 |
| Ours (fp8_ds_mla, graft, no drafter, 0.98) | 2,717,420 | 10.37 |

Our larger pool comes from configuration choices that cost something else:

- no drafter: +2.78 GiB per card of KV, paid for with single-user speed;
- utilisation 0.98: a thin margin when using `prompt_logprobs`.

The graft goes the other way: it uses 1.98 GiB per card more than the published pack. Morrowmake's `bfloat16` KV takes more memory per token than `fp8_ds_mla`. Morrowmake can turn its drafter off (`EXTRA_ARGS="--speculative-config None"`), but we do not know what pool that gives.

## 3. Disk, RAM and installation

| | Morrowmake 1.0.0 | Ours |
|---|---|---|
| Downloads | Full official checkpoint, 48 shards, 475.3 GiB at `1bae0e10`, plus LibertAI 2.9 bpw at `c178a012`, 196.3 GiB: **671.5 GiB**. Preflight requires 680 GiB free | LibertAI 2.9 (196.3 GiB) + 5 shards of LibertAI 3.5 (about 24 GiB) + official non-expert tensors by HTTP range requests (about 17.5 GiB) + Engram shards 47-48 (about 189 GiB): **about 428 GiB** |
| Why | Its builder takes the non-expert tensors from local official shards. Every one of the 48 shards holds some of them | GNmac's `fetch_official_nonexpert.py` fetches only the needed byte ranges |
| Engram placement | `auto`: RAM when at least 212 GiB is available, otherwise SSD with a row cache. 202-204 GiB of host RAM in RAM mode | `ssd` mode reading from a 200 GB tmpfs. About 203 GiB not available in our container after start |
| Engine image | Pulled, pinned by digest, 9.6 GB compressed. Building from source needs prebuilt native extensions that are not in the repository | Built locally from GNmac's Dockerfile: about 20 minutes here, 17.8 GB |
| Pinning | Weights pinned by revision, hybrid pinned by SHA-256 | GNmac downloads from `main`; our patched scripts pin every revision. We verified the downloads against published SHA-256 |
| API | `127.0.0.1` by default, `API_KEY` setting | `127.0.0.1` by default, key through `EXTRA_ARGS` |

Disk was the deciding factor for us. 575 GiB were free, below Morrowmake's 680 GiB requirement. Morrowmake's build could be "thinned" shard by shard, but the risk of a silent re-download of several hundred GiB made that unattractive.

## 4. P2P and the TRAP31 patch

Morrowmake's documentation states that working P2P on CMP 170HX needs a P2P-capable driver build. Its release uses:

- mainline [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) with its 74-SM override and the maintainer's P2P changes;
- Morrowmake's minimal TRAP31 Booter PLM patch, gated on `ForceP2P=0x11`, packaged as [Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker).

Morrowmake says advertising peer access alone is insufficient.

At every new driver or boot identity, its start runs a content check: peer copies, kernel reads and writes, and a CUDA IPC write from another process, across all ordered GPU pairs. If any byte differs, it starts with P2P **off**. Its own fallback measurement shows what that costs: 1-user prose 133 tok/s instead of 162, and 8 users 344 instead of 432.

Our driver is a cmpunlocker fork with a P2P patch, **without** TRAP31. On it:

- `nvidia-smi topo -p2p` reports OK;
- NCCL passes on all four cards;
- GNmac's stack runs with `NCCL_P2P_LEVEL=SYS` and FlashInfer's PCIe IPC all-reduce.

Morrowmake's own documentation warns that a topology report is not a data-integrity test. We do not know whether our driver passes Morrowmake's content check. Do not use `P2P=force` to skip that check: Morrowmake documents that it may hang or corrupt data.

## 5. What a fair same-machine A/B would need

1. **Disk for both, or a careful sequence.**
   - Separately, the two need about 1.1 TB.
   - The LibertAI 2.9 pack (same weights at both pinned revisions) and the Engram shards could probably be shared (inference). That leaves about 715 GiB: Morrowmake's 671.5 GiB plus our 3.5 bpw shards and the non-expert extract.
2. **One driver for both** that passes Morrowmake's P2P content check. Otherwise Morrowmake runs with P2P off and loses about 20 % at one user, which would not be a fair comparison.
3. **The same power cap and clocks**: 180 W as both recipes publish, or 200 W for both.
4. **The same benchmark scripts against both servers**: GNmac's `bench/mia.py` and `bench/prefill.py`, plus Morrowmake's protocol (time to first token included), each with `--model` set to the served name (`dsv41-flash` vs `deepseek-v4.1-flash`).
   - At least 3 repetitions per point; report the median and the range.
   - Check that no point hits the prefix cache.
5. **Matched speculative settings**: both without a drafter, then both with DSpark. Report the KV pool for each run.
6. **A factorial quality design**, to separate the KV/numerics effect from the expert-width effect:

   | | Layers 18-22 at 2 bits | Layers 18-22 at 3 bits |
   |---|---|---|
   | Morrowmake engine (bf16 KV, reference rounding) | as shipped | graft (requires editing its pinned hybrid hashes) |
   | GNmac engine (fp8_ds_mla) | `k29` | `k3` (ours) |

   - All four would be scored with `tests/nll_eval.py` on the same frozen token windows, in A-B-A order across restarts.
   - First check that Morrowmake's server accepts `prompt_logprobs` and gives bit-identical results across its own restarts.
   - Morrowmake's `VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE=1` variant could be added as a fifth arm.
7. **An external reference.** Relative NLL only ranks the four arms against each other. Saying which is closest to the original model needs reference logits on the same windows, from DeepSeek's reference implementation on hardware that can run the full checkpoint. We do not have that hardware.
8. **The functional gates on both**:
   - the copy test, with and without prefix caching;
   - the determinism test;
   - the decode/prefill self-check;
   - HumanEval and GSM8K with one harness;
   - a multi-turn tool-calling session, since agentic use is what we run.

Until that is done:

- For speed, Morrowmake's published prefill is clearly ahead.
- For quality, the honest summary is the one in [section 1.6](#16-quality-verdict).
