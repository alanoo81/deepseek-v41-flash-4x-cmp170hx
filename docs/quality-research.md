# Fidelity research: getting as close as possible to the original DeepSeek-V4.1-Flash on 4x CMP 170HX

> **Second round (2026-10-31).** The graft chosen here also works in Morrowmake's engine (-1.81 %), which we now serve; 4-bit experts were tested again on layers 36-39 (-0.11 % on long contexts). See [results.md](results.md).

This document covers the quality side of the project, in the order it happened. It explains which quantized setup we chose and why, how we checked that the deployment really runs the model we think it runs, and what we measured. It also covers what we got wrong. Speed numbers appear only where they bear on a fidelity decision; see [`results-gnmac.md`](results-gnmac.md) for the full set. The research process itself (parallel read-only agent workflows with a critic pass) is described in [`agentic-methodology.md`](agentic-methodology.md).

Conventions used throughout:

- **Measured** means we ran it on our hardware and the raw output is in `data/`. Each result gives the file it comes from.
- **Upstream claim** means a number published by someone else that we did not reproduce. The source is named.
- **Inference** marks our own reasoning that no measurement backs.
- GiB = 2^30 bytes. "Per card" means per GPU (tensor-parallel rank) unless stated.
- Where a number comes from our run notes and no raw log is included in `data/`, the text says so.

## Contents

1. [Goal and constraints](#1-goal-and-constraints)
2. [The quantization landscape](#2-the-quantization-landscape)
3. [Why the 3.5 bpw pack does not fit: per-card VRAM arithmetic](#3-why-the-35-bpw-pack-does-not-fit-per-card-vram-arithmetic)
4. [Why we did not use Mia EAQ v1](#4-why-we-did-not-use-mia-eaq-v1)
5. [The graft: layers 18-22 from 2 bits to 3 bits](#5-the-graft-layers-18-22-from-2-bits-to-3-bits)
6. [The copy test, and the deployment that ran without Engram](#6-the-copy-test-and-the-deployment-that-ran-without-engram)
7. [The fix: Engram tables in a tmpfs, read in `ssd` mode](#7-the-fix-engram-tables-in-a-tmpfs-read-in-ssd-mode)
8. [Measuring fidelity: the NLL A-B-A protocol](#8-measuring-fidelity-the-nll-a-b-a-protocol)
9. [Results](#9-results)
10. [KV cache format: an open question](#10-kv-cache-format-an-open-question)
11. [DSpark: lossless, but too expensive in VRAM for us](#11-dspark-lossless-but-too-expensive-in-vram-for-us)
12. [Other deviations from the reference that we know about](#12-other-deviations-from-the-reference-that-we-know-about)
13. [Mistakes and dead ends](#13-mistakes-and-dead-ends)
14. [Open questions](#14-open-questions)
15. [Reproducing the numbers](#15-reproducing-the-numbers)
16. [Credits](#16-credits)

---

## 1. Goal and constraints

The goal was **output as close as possible to the original model**, [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10`. Speed came second. Speed matters only as long as the server stays usable for interactive and multi-agent work.

Hardware:

| Item | Value |
|---|---|
| Board, CPU, RAM | ASRock Rack ROMED8-2T, AMD EPYC 7F52, 512 GB DDR4-3200 on 8 channels |
| GPUs | 4x NVIDIA CMP 170HX (GA100, sm_80), 64 GB each, PCIe Gen2 x16 |
| GPU memory | `nvidia-smi` shows 65,536 MiB total and 625 MiB reserved per card, so about 63.39 GiB is visible to CUDA |
| Driver | NVIDIA open driver 610.43.02 with a cmpunlocker fork and a P2P patch (P2P on, NCCL healthy on all 4 cards) |
| Power and clocks | 200 W cap per card, 170tune +200 MHz at 1,500 MHz |
| Runtime | Docker inside an LXC container, GPUs passed through with CDI |

The starting point is the recipe we build on, [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) @ `2ad5741ab1` (release 1.2.0). It serves a **hybrid checkpoint**, which follows Curtis White's recipe and uses his `quantization_config.json`:

- **Routed experts:** EXL3 from [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), 196.3 GiB as a pack. Layers 18-22 are 2 bits per weight and all other layers are 3.
- **Everything else** comes from the official release, fetched with HTTP Range requests: attention, shared experts, router, head and embedding, the DSpark drafter and vision, 17.5 GiB in all. These tensors stay in their native FP8, BF16 or F32.
- **Engram tables** come from official shards 47-48, 189.1 GiB. They live in host memory or on NVMe, never in VRAM.
- **Total:** 402.9 GiB of weights.

We fixed three requirements before choosing anything:

- **KV pool:** at least about 1.2M tokens at a 262,144-token context, so that several agents can hold long contexts at once.
- **Vision:** kept. We need it for end-to-end tests.
- **Engram:** kept, because it is part of the model.

## 2. The quantization landscape

### 2.1 What the original model looks like

We read the official `config.json` and the safetensors headers of all 48 official shards (header bytes only, fetched by Range request):

| Part | Native format | GiB |
|---|---|---:|
| Routed experts, layers 0-39 (40 x 384 x 3 matrices, 543.6B params) | MXFP4: FP4 e2m1 with one UE8M0 scale per 32 values (4.25 bpw) | 268.9 |
| DSpark drafter routed experts | MXFP4 | 6.7 |
| Attention, KV compressor, indexer | FP8 e4m3, 32x32 UE8M0 block scales | 4.87 |
| Shared experts | FP8 | 1.42 |
| Embedding and LM head | BF16 | 2.47 |
| Vision tower | BF16 | 0.77 |
| Router, norms, mHC, aligner | BF16 / F32 | 0.36 |
| MTP dense | FP8 / BF16 | 0.57 |
| Engram tables (layers 1 and 14) | FP8 with E8M0 scales per 32 | 188.8 |
| **Total** | 48 shards | **475.3** |

Leaving out Engram and the drafter, the native model is about 279 GiB. The 4 cards have about 4 x 60-62 GiB to spend, including KV and runtime. So the routed experts are the only part that has to be compressed, and they are where almost all of the fidelity loss comes from.

### 2.2 Routed-expert size per pack

Each size below is computed from the pack's own shard headers. The extra GiB per card is measured against the 2.9 bpw baseline that GNmac ships.

| Routed experts | GiB | Per card vs 2.9 | Fits with a useful KV pool? |
|---|---:|---:|---|
| LibertAI / Mia 2.9 (K3, K2 on 18-22) | 182.6 | 0 | yes |
| 3 bits on every layer (our graft) | 190.5 | about +2.0 | yes |
| LibertAI 3.5 / easthambuilds 3.5 | 222.1 | +9.9 | no |
| easthambuilds 3.75 | 237.9 | +13.8 | no |
| bertholomus 4.0 | 253.8 | +17.8 | no |
| Native MXFP4 | 268.9 | +21.6 | no |

Per layer (all 384 experts, total over the 4 cards): K4 = 6.344 GiB, K3 = 4.762 GiB, K2 = 3.180 GiB, native MXFP4 = 6.724 GiB. One extra bit per weight over all 40 layers costs about 63.3 GiB.

What we found in the landscape survey, and the published fidelity numbers that went into the decision (all upstream claims):

- **LibertAI 2.9 bpw** (whole pack against the unquantized model): perplexity +1.8 % on code and +0.4 % on prose.
- **LibertAI 3.5 bpw:** 4-bit experts on layers 0-9 and 30-39, 3-bit on 10-29, all non-expert tensors in EXL3 6 bpw. Perplexity +0.8 % on code and +0.1 % on prose.
- **Mia-AiLab EAQ v1 2.9 bpw:** the same per-layer widths as LibertAI 2.9. KL 0.0137 and top-1 agreement 96.6 %, against 0.0155 and 96.2 % for LibertAI on Mia's panel. See section 4.
- **Per-expert mixed-width packs** (sudoingx 2.8, HeNryous 2.75, philbert440's franken-v100 and others) measure about 25-35 % lower KL than the uniform 2.9 packs on two independent panels. They cannot run on the GNmac stack as is: its tensor-core MoE kernels accept only 2-bit or 3-bit layers, with a single width per layer.
- **unsloth GGUF UD-Q8_K_XL** keeps the official MXFP4 experts. KL against DeepSeek's FP32 reference is 0.0048, with 98.5 % top-1. On paper this is the most faithful option. But it needs an unmerged llama.cpp pull request (ggml-org/llama.cpp#30817), and that PR has open Engram corruption bugs: a race in lazy mode, and broken n-gram history on prefix-cache reuse. We set it aside for now.
- **Official MXFP4 experts with some layers on the CPU** (zebgop-ops/dsv41reap-pp): this runs the original expert weights. It is reported at 14-22 tok/s decode, PP4, 131k context. It has not been verified at TP4 and it is slow.

An important caveat on all KL figures (upstream claim, not verified by us): the body of llama.cpp PR #30817 reports that DeepSeek's own native FP8/FP4 kernels differ from DeepSeek's F32 reference by a KL of about 0.015-0.026. So "the original model" is ambiguous: it can mean the F32 maths or DeepSeek's production kernels. Differences of about 0.002 between packs are small next to that floor.

## 3. Why the 3.5 bpw pack does not fit: per-card VRAM arithmetic

We wanted the LibertAI 3.5 bpw experts, the most faithful EXL3 pack with published numbers. The budget says no. All figures are per card at TP=4, `GPU_MEMORY_UTILIZATION=0.95`.

**Budget.** 0.95 x 63.39 GiB = **60.22 GiB per card** for weights, runtime, activations, CUDA graphs and KV.

**Routed experts per card.** At TP4 each rank holds a slice of all 384 experts, not separate experts. The 2,304-wide intermediate dimension is split in 128-wide blocks as 640/640/512/512, rotated per layer.

| Pack | Routed experts per card |
|---|---:|
| 2.9 bpw | 46.02 GiB (worst rank) |
| 3.5 bpw | 55.86 GiB (every rank) |
| **Difference** | **+9.84 GiB** |

**Calibration on the published configuration.** GNmac's own variant A (2.9 pack, DSpark on) leaves 2.95 GiB of KV per card at 0.95, which holds 1,395,731 tokens. Weights come to 52.15 GiB per card: 46.02 routed + 5.98 official rest + 0.15 FP8 head copy. That leaves 60.22 - 52.15 - 2.95 = 5.12 GiB for runtime, the prefill activation peak and CUDA graphs.

**Variants with the 3.5 experts.** KV left per card, with the overhead held at variant A's value:

| Variant | KV per card |
|---|---:|
| 3.5 experts + official FP8 rest | **-6.89 GiB** |
| ... without the DSpark drafter (about -2.18 GiB) | -4.71 GiB |
| ... without drafter and without vision (-0.90 GiB) | -3.81 GiB |
| ... plus utilization 0.98, `--enforce-eager`, 1,024-token chunks | about -0.01 GiB |

Even with every knob turned, no KV is left. The standalone 3.5 pack is not an alternative either. Its non-expert tensors are EXL3, which the stack's loader does not accept, since it takes only routed experts from an EXL3 pack. Its 6 bpw drafter experts are also larger than the official FP4 ones.

**KV cost per token.** On sm_80 the stack uses `fp8_ds_mla`. Only layers 2, 8 and 14 (compression ratio 2) and layer 20 (ratio 1) own a cache; the other layers reuse it.

| Cache | Bytes per entry | Bytes per token |
|---|---:|---:|
| Compressed latent (448 FP8 NoPE + 128 B BF16 RoPE + 8 B scales) | 584 | 3 x 584/2 + 584 = 1,460 |
| FP8 indexer keys | 132 | 3 x 66 + 132 = 330 |
| **Total** | | **1,790** |

Fitting GNmac's five published pool sizes gives **about 1,802 B per token plus about 117 MiB per sequence slot**. The slot cost comes from the sliding-window caches of 43 layers. At `max_model_len` 262,144, one full sequence therefore takes about 0.554 GiB per card. That works out to an effective **~473,000 tokens of pool per free GiB per card**. The cache is replicated on every rank, because the latent is not split by head.

Our own pools confirm the rate:

| Free KV per card | Pool | Tokens per GiB | Source |
|---:|---:|---:|---|
| 3.84 GiB | 1,812,663 | 472k | [`data/logs/01-first-start-engram-NOT-loaded.txt`](../data/logs/01-first-start-engram-NOT-loaded.txt) |
| 5.76 GiB | 2,717,420 | 472k | [`data/logs/08-ceiling-0.98-smoke-stress-nll.txt`](../data/logs/08-ceiling-0.98-smoke-stress-nll.txt) |

**What fits.** Keep the 2.9 pack and replace only its 2-bit layers 18-22 with the 3-bit layers of the 3.5 pack. Going from 2 to 3 bits grows each expert matrix from 2,949,120 B to 4,423,680 B, which is +1.582 GiB per layer and +7.91 GiB for the five layers. That is about +1.9-2.0 GiB per card, depending on the TP rotation. The tensor-core kernels still apply, because every layer stays at 3 bits.

Predicted pools for this graft at 0.95 (inference from the budget model):

| Configuration | Predicted KV pool |
|---|---:|
| Graft with the drafter | about 480,699 tokens (1.016 GiB per card) |
| Graft without the drafter | about 1.5M tokens |
| Graft without drafter or vision | about 1.9M tokens |

We chose the graft without the drafter. The measured pool at 0.95 was **1,812,663 tokens**, 3.84 GiB per card ([`data/logs/01-first-start-engram-NOT-loaded.txt`](../data/logs/01-first-start-engram-NOT-loaded.txt)). So the budget model was conservative by about 0.6 GiB per card. Its own stated uncertainty was about ±0.4 GiB per card, and it carried a known accounting gap of up to 0.8 GiB against GNmac's overhead breakdown.

## 4. Why we did not use Mia EAQ v1

Our first plan was the Morrowmake recipe with Mia-AiLab's EAQ v1 2.9 bpw pack in place of LibertAI's, because Mia's card reports a lower KL. The research changed our minds.

**At the format level, Mia is a drop-in replacement.** Both packs' `quantization_config.json` list the same 46,080 routed-expert entries (`layers.N.ffn.experts.E.w1/w2/w3`). Every entry matches between the two packs: tensor names, the mul1 codebook, `mul1_multiplier` 2212286765, `out_scales=always` and the per-layer widths. The only mismatch is `ffn.gate.bias`, F32 in Mia and F16 in LibertAI. The hybrid loader does not take that tensor from either pack, because the router bias comes from the official FP32 weights.

**But the upstream author tested it in this exact hybrid and rejected it.** GNmac's `docs/tuning-log.md`, line 51, in the "tried and dropped" table: *"Mia-AiLab EAQ-v1 2.9-bpw experts instead of LibertAI | same speed, +1.55 % NLL in our hybrid (2.7 σ)"*.

Why the two results disagree (inference):

- Mia's KL compares two complete checkpoints. That includes their EXL3 attention, indexer and head, and LibertAI's router bias in F16. Mia's own card says F16 rounding of that bias broke router matching in an earlier build.
- The hybrid replaces all of those with official tensors. Most of Mia's measured advantage may therefore sit in parts the hybrid throws away.
- EXL3 calibration also tunes each pack's experts to that pack's own quantized attention outputs, while the hybrid feeds them FP8 attention outputs.

**What we did not do:** we have not measured Mia in our own harness. GNmac's figure is NLL on text, not KL to the original, and the tuning log does not say which passages it used. We defaulted to LibertAI because GNmac's test is the only measurement made in the setup we actually run, and it favours LibertAI. This is listed as an open question in section 14.

## 5. The graft: layers 18-22 from 2 bits to 3 bits

### 5.1 What is grafted

The hybrid directory is a set of relative symlinks plus a rewritten `model.safetensors.index.json`. Both LibertAI packs keep exactly one expert layer per shard: layer L lives in `model-{L+2:05d}-of-000NN.safetensors`. So the graft does three things:

- **Download:** shards `model-00020` to `model-00024-of-00044` of the 3.5 bpw pack @ `04a492fc` (about 24 GiB), instead of the whole 240 GiB pack.
- **Index:** point `layers.18-22.ffn.experts.*` in the hybrid index at those shards, and drop the links to the matching 2-bit shards. The original 2-bit shards stay on disk for A/B tests.
- **Loader filter:** extend `DSV41_EXPERTS_ONLY_FILES_RE` from `of-000(43|39)` to `of-000(43|44|39)`. The non-expert tensors in the 3.5 shards (EXL3 6 bpw attention and others) are then skipped, and the official FP8 tensors are used instead. See [`recipe/serve.sh.patch`](../recipe/serve.sh.patch).

All three downloads are pinned: LibertAI 2.9 @ `0be63c33`, LibertAI 3.5 @ `04a492fc`, official @ `1bae0e10` ([`recipe/tools/download-pinned.sh`](../recipe/tools/download-pinned.sh), [`recipe/tools/fetch_official_nonexpert.py.patch`](../recipe/tools/fetch_official_nonexpert.py.patch)). The upstream recipe fetched the official non-expert tensors from the unpinned `main` branch.

### 5.2 Why the splice is compatible

Before downloading anything, we read the headers and configs of both packs:

- Both were made with exllamav3 1.6.2, the mul1 codebook, the same `mul1_multiplier` and the same calibration size (250 x 2,048).
- Tensor names are identical.
- In the 3.5 pack's `exl3_k_map.json`, layers 10-29 are K3.
- The bit width is read from each layer's trellis shape at load time, not from the config. A K3 layer from the 3.5 pack is therefore indistinguishable to the loader from a K3 layer of the 2.9 pack.

One residual risk remains (inference). The grafted layers were calibrated inside a different pack, with 4-bit neighbours in layers 0-9. Section 9.3 measures whether that matters in practice.

### 5.3 How it was verified

The script is [`recipe/tools/make_hybrid_graft.py`](../recipe/tools/make_hybrid_graft.py). It first builds the plain hybrid with the upstream `make_hybrid.py`, then rewrites the index. For every grafted layer it asserts that:

1. the set of expert tensor names is identical in the two packs;
2. all of the layer's expert tensors sit in a single shard;
3. that shard holds no expert tensors from any other layer (the loader reads every tensor of a linked file);
4. the width **read from the safetensors header** equals the expected K. For each `.trellis` tensor, K = last dimension / 16. The script trusts this, not `exl3_k_map.json`;
5. every file the index references exists after the links are rewritten.

Verification steps, in order:

- **Mock-up test.** We ran the script on a mock models directory built from the real index files of both packs before the weights were downloaded.
- **Checksums.** After the download, we checked the SHA-256 of all 55 LFS files against the Hugging Face tree.
- **Assembled index.** It had 188,241 tensors, 23,040 of them grafted: 5 layers x 384 experts x 3 matrices x 4 stored tensors.
- **Startup log.** It showed the tensor-core MoE hook taking all 40 layers on every rank at `bits_x2=6` (K3), with none skipped or failed. That is 160 lines in all.

The checksum, index and startup checks are recorded in our run notes; those logs are not included in `data/`.

The script also has a `k4x4` profile, which additionally grafts four 4-bit layers (section 9.2).

## 6. The copy test, and the deployment that ran without Engram

### 6.1 Why a copy test

GNmac's issue #1, a LibertAI discussion and Morrowmake's issue #1 all report misspelled names and corrupted paths in agentic use. The llama.cpp PR reports a related failure class: corrupted Engram n-gram history when a cached prompt prefix is reused. So the first thing we tested after startup was exact copying. [`tests/copytest-dsv41.py`](../tests/copytest-dsv41.py) works like this:

- **Task:** copy a list of 40 random identifiers exactly (`word_word_5digits_8hex`), at temperature 0 with reasoning off. The instruction is in French, and so are the script's printed results in `data/logs/`.
- **Pass 1:** 20 unique prompts, so nothing is in the prefix cache.
- **Pass 2:** the same prompts again, served from the prefix cache.
- **Pass 3:** a second conversation turn on a cached prefix (turn 1, its exact answer, then a new list).
- **Scoring:** exact copy yes or no, plus the number of wrong lines.
- **Prefix cache visibility:** `--enable-prompt-tokens-details` makes vLLM report `cached_tokens`, so cache use can be seen.

### 6.2 First result: 15/20, with non-determinism

On the first deployment, these results come from our run notes; the raw output was not kept:

- **Pass 1:** 15/20 exact. The same 5 tasks failed at concurrency 8 and when replayed one at a time.
- **Failure types:** 2 failures were massive. The `_` separators between a word and a number vanished, and every later line copied the mistake. The other 3 failures had 1-2 wrong lines.
- **Turn 2** was 20/20.
- **Temperature 0 was not deterministic.** One identifier came back as `..._soil_f56ba531` on one run and `..._0f56ba531` on another, and one task passed and then failed when replayed alone.

The speed benchmarks of that deployment looked normal and close to upstream's: about 105 tok/s for one user, a cold prefill median of 2,421 tok/s, and no NaNs. They are kept as [`data/bench/INVALID-no-engram-decode.log`](../data/bench/INVALID-no-engram-decode.log) and [`data/bench/INVALID-no-engram-prefill.log`](../data/bench/INVALID-no-engram-prefill.log).

### 6.3 Diagnosis

A read-only diagnosis read the image's source and logs and sent no traffic to the server. It found three things.

**The tokenizer explains the shape of the error.** An underscore before a letter merges into that letter's token (`_n`, `_p`, `_e`). An underscore before a digit is a standalone `_` token. The failing identifiers lost exactly the standalone `_` tokens.

**The one-at-a-time replay did not prove what we thought.** The server log showed the solo replays reusing prompt KV that had been computed during the concurrency-8 run. So "same failures alone and in a batch" did not separate the hypotheses.

**The main finding: the Engram tables had never been loaded.** This was our configuration mistake:

- **What we did.** We had left `DSV41_ENGRAM_STORAGE` unset. We assumed that meant "the stock vLLM mode, tables in RAM" and wanted to avoid reading them from NVMe. The upstream author never tested this mode; their documented path is `ssd`.
- **What happens in that mode.** In "host" mode, the stack's Engram layer reserves pinned host memory with `torch.empty(..., pin_memory=True)`: 2 tables x 23.6 GiB per rank, 188.8 GiB over 4 ranks. It then relies on the normal vLLM weight loader to fill that memory.
- **Why the loader never fills it.** The hybrid index contains only the small Engram tensors (`q_weight`, `k_weight`, `wkv.weight`, `wkv.scale` for layers 1 and 14). It has no `engram.embed.weight` or `engram.embed.scale`. The tables exist only in the separate Engram directory, which only `ssd` mode reads. Upstream's README says so: the tables are streamed from NVMe on demand.
- **Why nothing failed.** vLLM checks for missing weights only when `quantization is None`. The server runs with `exl3` quantization, so the absent tables were never reported. The startup log said "Engram table offloaded to pinned host memory ... 23.60 GiB per rank". That line reports the reservation, not a fill. Loading 2 x ~101 GB of tables would also have taken far longer than the 20 seconds the shard loading took.
- **What Engram did as a result (inference).** Fresh pinned pages are zero-filled. A zero scale byte decodes to a scale of 0.0, so every looked-up row is zero whatever its FP8 bytes hold. The gate kernel computes `hidden + gate * value`, so Engram layers 1 and 14 became identity functions. We did not dump the memory to confirm the zeros. The fix in section 7 is consistent with this explanation.

How much Engram matters (upstream claim): LibertAI's card reports that turning Engram off takes code perplexity from 1.91 to 4.16.

We did check whether we could simply add the tables to the hybrid index, so that host mode would load them. The answer was no, for two reasons:

- **Memory.** The vLLM safetensors iterator calls `get_tensor` on each tensor in full, so every one of the 4 worker processes would materialize about 95 GiB of table. That is an out-of-memory error at the container's limit.
- **It would not save memory anyway.** Host mode already pins the full 188.8 GiB.

### 6.4 What else the diagnosis ranked, and how that turned out

Before the Engram finding, the read-only analysis ranked several other suspects for the copy errors and for the non-determinism at temperature 0:

- prefix-cache hits sending reruns down a different kernel shape;
- batch-shape-dependent kernels;
- the unverified 1-5-row range of a decode MoE variant;
- FP8 versus FP4 indexer keys flipping near-tie top-k choices.

One reviewer also predicted that the errors would turn out to be near-ties that the original model shares, as in an earlier LibertAI discussion. **None of these turned out to matter here.** Once the tables were loaded, the errors and the non-determinism both disappeared (section 7). We keep this list because it is a good catalogue of real numerical effects on this stack. The process lessons are discussed in [`agentic-methodology.md`](agentic-methodology.md), section 7.

## 7. The fix: Engram tables in a tmpfs, read in `ssd` mode

The fix uses the path upstream tested, `ENGRAM_STORAGE=ssd`. In that mode the `engram_ssd` row store reads 264-byte rows (256 FP8 values plus 8 scale bytes) through the kernel page cache. We did not point it at the NVMe copy, because on our storage that would have been slow:

- The models sit on a ZFS dataset with 128 KiB records and no compression.
- The ARC is capped at 6.26 GiB.
- `RWF_NOWAIT` is not supported (errno 95).
- **Inference:** a page-cache miss reads a whole 128 KiB record to return a 264-byte row. The model reads 48 rows per token (8 heads x n-grams 2-4 x 2 Engram layers), so that comes to about 6 MiB per decoded token.

So we put the tables in RAM, but through the tested `ssd` code path:

- [`recipe/engram-ram.sh`](../recipe/engram-ram.sh) mounts a 200 GiB tmpfs inside the container and copies the two Engram shards plus their `config.json` and index into it. The copy took 94 s from NVMe. The tmpfs does not survive a container restart, so `start.sh` calls the script before every start ([`recipe/start.sh.patch`](../recipe/start.sh.patch)).
- `serve.sh` mounts that directory read-only as `/engram` and passes `DSV41_ENGRAM_STORAGE=ssd` ([`recipe/serve.sh.patch`](../recipe/serve.sh.patch)).
- [`recipe/env.example`](../recipe/env.example) sets `ENGRAM_STORAGE=ssd`, `ENGRAM_HOST_DIR` to the tmpfs, and `--enable-prompt-tokens-details`.
- The tmpfs counts as shared memory in the container: 189 GiB "shared" in [`data/logs/02-restart-engram-tmpfs-ssd.txt`](../data/logs/02-restart-engram-tmpfs-ssd.txt). We raised the container's memory limit from 400 to 460 GiB.
- The four small Engram tensors are present both in the official non-expert shards the hybrid uses and in the full Engram shards. We checked that they are identical byte for byte.

**Verification after the fix:**

| Check | Result | Source |
|---|---|---|
| Startup | `Built engram token map (129280 -> 99092 ids) for layers (1, 14)` on all 4 ranks; no more "offloaded to pinned" lines; KV pool 1,817,233 at 0.95 | [`data/logs/02-restart-engram-tmpfs-ssd.txt`](../data/logs/02-restart-engram-tmpfs-ssd.txt) (pool size from run notes) |
| Copy test, 20 lists (seed 20261027) | 20/20 on all 3 passes; pass 2 had 92 % of prompt tokens from cache, pass 3 had 31 % | run notes; the same 20/20 result repeats in [`data/logs/04-stress-0.97-and-copytest.txt`](../data/logs/04-stress-0.97-and-copytest.txt) |
| Copy test, 100 new lists (seed 777) | **100/100 on all 3 passes (300/300)**, 0 wrong lines, 92 % cached in pass 2 | [`data/logs/03-copytest-100-and-determinism.txt`](../data/logs/03-copytest-100-and-determinism.txt) |
| Determinism (5 tasks x 5 solo repeats, seed 4242) | every task: 5/5 exact, **1 distinct output out of 5** | [`data/logs/03-copytest-100-and-determinism.txt`](../data/logs/03-copytest-100-and-determinism.txt) |
| Speed with Engram | 1 user 105.4/105.0/104.8 tok/s (structured/code/prose), 8 users 559.7/482.4/449.2, cold prefill median 2,514 tok/s | [`data/bench/v2-decode.log`](../data/bench/v2-decode.log), [`data/bench/v2-prefill.log`](../data/bench/v2-prefill.log) |

What this shows:

- **The copy errors and the non-determinism both came from the empty tables.** In this test, the prefix-cache / Engram-history failure that the llama.cpp PR reports did not appear on this stack. Pass 2 and pass 3 reuse cached prefixes and stay exact.
- **Reading Engram from RAM through `ssd` mode costs nothing measurable.** Single-user and prefill speed match the no-Engram run within a few percent.
- **The code rows are not comparable.** The 8-user code figure is higher with Engram (482.4 against 375.9 tok/s), but the model wrote answers of different length in the two runs (median 95 completion tokens at 1 user with the tables, 121 without), so that difference is mostly a different amount of work per request, not a speed change. See [`results.md`, section 4.2](results-gnmac.md#42-without-engram-tables-invalid-kept-for-the-record).

**Our lesson:** the upstream benchmarks (decode speed, prefill, no NaNs) cannot detect this failure. A model with two layers silently disabled runs exactly as fast. An exact-copy test caught it, and a plausible-looking speed number did not. The `.env` and `serve.sh` comments now say: never leave `ENGRAM_STORAGE` empty on this stack.

## 8. Measuring fidelity: the NLL A-B-A protocol

Copy tests saturate quickly: after the fix, every configuration we tried scored 20/20. To compare weight configurations we needed a continuous measure. We had no copy of the original model's logits, so we measured **paired held-out negative log-likelihood (NLL)** between configurations of our own stack. The tool is [`tests/nll_eval.py`](../tests/nll_eval.py) (Python standard library only).

### 8.1 Scoring

- **Request:** teacher forcing through the running server. Each window goes to `/v1/completions` as **token ids** (`[BOS=0] + window`), with `max_tokens=1` and `prompt_logprobs=0`. The script checks that the server echoes the same `prompt_token_ids`.
- **No chat template.** `/v1/completions` never applies one, and the model's tokenizer config adds no BOS on its own, which is why the script sends BOS explicitly.
- **No prefix cache.** In this vLLM, a request with `prompt_logprobs` set skips prefix-cache reads. Each request also carries a unique `cache_salt` as a second guard.
- **What is scored:** with `skip=1`, the first token after BOS is not scored, so a 1,024-token window scores 1,023 tokens.
- **Failure guard:** a logprob of -inf or below -9998 (how vLLM encodes -inf) stops the run.

### 8.2 Corpora (frozen once, as token ids)

| Set | Source |
|---|---|
| fr | 8 Project Gutenberg novels in French: Madame Bovary, Germinal, Du cote de chez Swann, Vingt mille lieues sous les mers, Les trois mousquetaires, Le rouge et le noir, Le comte de Monte-Cristo I, Les miserables I |
| en | 8 Project Gutenberg books in English: Middlemarch, North and South, Tom Jones, The King in Yellow, Leviathan, Wuthering Heights, The Picture of Dorian Gray, War and Peace |
| code | The GNmac recipe source at `2ad5741ab1`, MIT, through an explicit file-pattern whitelist that excludes the files we replaced or added (our two-line `start.sh` change is probably included, since `start.sh` is whitelisted and was already patched). It was committed after the model's release, so it cannot be in the training data |

Windows are cut at evenly spaced positions between 5 % and 95 % of each text, starting on a paragraph or line boundary, using the server's own `/tokenize`. They are then saved as token ids with SHA-256 hashes. Every configuration sees exactly the same tokens, and `compare` refuses runs whose corpus hashes or parameters differ.

There are two window sets, because the window length decides which MoE kernel runs:

| Set | Size | Kernel path |
|---|---|---|
| **Long** | 64 windows x 1,024 tokens per corpus (192 windows) | Prefill MoE kernels. 1,024 tokens fit one 2,048-token prefill chunk and keep the logits spike near 1.4 GiB per card. A 2,048-token window would have needed about 2.7 GiB, which vLLM's startup memory profile does not account for |
| **Short** | 400 windows x 40 tokens per corpus (1,200 windows) | BOS + 40 tokens is at most 42 rows, which sends the routed experts through the decode-shaped kernels, including the cooperative kernel that runs 4-bit layers |

**Limitation (inference).** Gutenberg text is almost certainly memorized, which is why English perplexity is as low as 1.32 on long windows. That compresses differences between configurations. The code set is held out but partly derived from vLLM and exllamav3.

### 8.3 Design: A-B-A across restarts

The plan we first wrote had a classic mistake: an A/A pair inside one server process as the "noise floor". GNmac documents that same-shape runs in one process are bit-exact, so an in-process A/A pair gives zero and proves nothing about the noise that a restart or a kernel change adds. The review pass caught this, and we ran **A-B-A across restarts** instead:

1. K3 (graft, all layers 3-bit), server freshly started: `k3`, `k3-short`, `sc-k3`.
2. Restart with the candidate weights, same environment otherwise: `k4`, `k4-short`, `sc-k4`.
3. Restart on the K3 weights again: `k3bis`, `k3bis-short`. **This pair is the null comparison.**
4. Later, restart on the original 2.9 pack: `k29`, `k29-short`.
5. Later still, K3 at utilization 0.98 instead of 0.97: `k3-098`.

Nothing else ran on the server during a run, because batch composition changes the numerics. We also set `VLLM_EXL3_FAT_THRESHOLD=2048` in every configuration. It disables a lazily allocated "fat expert" scratch path that 4-bit layers would hit outside the memory budget, and it has no effect on K3 layers.

### 8.4 Statistics

The unit is the window: `d_i = NLL_B - NLL_A` is the per-window mean in nats per token. `compare` reports, per corpus and pooled:

- the change in perplexity in percent;
- a percentile bootstrap 95 % CI (20,000 resamples, fixed seed, stratified by corpus for the pooled row);
- the t-based CI;
- an exact sign test and a Wilcoxon signed-rank test;
- the count of windows where B is better or worse;
- the minimum detectable effect at 80 % power (MDE);
- the change in top-1 accuracy;
- token-level |delta logprob|.

We call an effect resolved when its bootstrap CI excludes 0 and it is well above the cross-restart null.

We had to fix one bug by running the tool: the sign test summed probabilities with `math.fsum` over binomial terms, which overflowed for n = 400. It now uses exact integers.

### 8.5 The decode path: `selfcheck`

`prompt_logprobs` exercises only the prefill path. The `selfcheck` subcommand covers decode:

- **Decode against prefill:** it generates 64 tokens greedily with logprobs after 1,024-token prompts (decode path), rescores the same tokens by teacher forcing (prefill path), and reports how far the two disagree.
- **Causality canary:** a random 1,000-token block repeated once. The first copy must stay near-random, around 13 nats per token, and the second copy must be almost free. GNmac uses the same test, where the first copy stays at 13.0 nats.

### 8.6 Finding: runs are bit-exact across restarts on the prefill path

| Comparison | Long windows | Short windows | Data |
|---|---|---|---|
| `k3` vs `k3bis` (restart, same weights) | every one of 192 windows identical, every token logprob identical (mean abs delta 0) | every one of 1,200 windows identical | [`data/nll/k3.json`](../data/nll/k3.json), [`data/nll/k3bis.json`](../data/nll/k3bis.json), `*-short.json` |
| `k3` vs `k3-098` (restart, utilization 0.98, larger KV pool) | identical | not run | [`data/nll/k3-098.json`](../data/nll/k3-098.json) |

So on this stack the cross-restart noise in teacher-forced NLL is exactly zero, across three server processes and two memory settings. GNmac's troubleshooting page says greedy text is not bit-reproducible across processes in TP4 decode. That statement is about decode, and our cross-restart evidence covers only the prefill path. Within one process, repeated greedy decodes were byte-identical (section 7). With a zero null, every non-zero difference between configurations comes from the change itself, partly from the weights and partly from any kernel change that comes with them.

There is one exception. **The K4 configuration was not bit-exact within one process on long windows.** Each run rescores 6 windows a second time as a determinism probe:

- **K3, 2.9 and short K4 runs:** the rescore is identical within 5e-8, which is float32 serialization.
- **Long K4 run:** the rescore differed on 4 of 6 windows, by up to 0.0011 nats per token on the window mean and up to 1.7 nats on one token ([`data/nll/k4.json`](../data/nll/k4.json), field `repeat`).

K4 layers run prefill through the stock exllamav3 MoE kernel, not the tensor-core kernel. **Inference:** that kernel's reduction order is not fixed. We did not investigate further, because we did not keep K4.

## 9. Results

Reproduce each table with `python3 tests/nll_eval.py compare <A.json> <B.json>`. dPPL is the perplexity change of B against A; negative means B predicts the held-out text better. "B better / worse" counts windows.

### 9.1 Reference perplexities of the served configuration (K3 graft)

| Corpus | Long windows (1,024) | Short windows (40) |
|---|---:|---:|
| fr | 2.1242 | 12.5134 |
| en | 1.3191 | 6.9919 |
| code | 2.9846 | 23.9634 |
| all | 2.0298 | 12.7989 |

Data: [`data/nll/k3.json`](../data/nll/k3.json), [`data/nll/k3-short.json`](../data/nll/k3-short.json).

These numbers say nothing about distance to the original model by themselves. They are the baseline for the paired comparisons below.

### 9.2 4-bit experts on layers 0, 1, 3 and 6: no measurable gain

**Choice of layers.** We used [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) @ `f8f20b37`, `data/alpha_sensitivity_1017.json`. sudoingX injects relative Gaussian noise into one layer's routed-expert output and measures the KL of the final distribution, giving alpha_l = KL / sigma^2. On the 1017 checkpoint the values fall almost steadily with depth: L0 5.17, L1 5.12, L6 4.42, L2 3.99, L5 3.99, L3 3.95, ..., L30-39 from 0.37 down to 0.02.

Our selection went through two rounds:

- **First pick:** {0, 1, 6, 5}. Five per-expert allocation plans from other packs, and the hyper-connection write gains on Pollard's card, also put the early layers on top.
- **Review:** layer 5 was swapped for layer 3. Their alphas differ by 1 %, well within noise, and layer 3 balances the TP rotation, which saves 0.088 GiB on the worst card.
- **Source:** the four 4-bit layers came from shards `00002`, `00003`, `00005` and `00008-of-00044` of the 3.5 pack @ `04a492fc`, each SHA-256-checked.
- **Rejected alternative:** the 3.5 pack itself puts 4 bits on layers 0-9 and 30-39. Its card gives no reason. Layers 30-39 are the least sensitive by every measure we found (2.7 % of total alpha), so we did not copy that choice.

**Kernel path.** The tensor-core MoE hook rejects 4-bit layers (`bits_x2 8`), so they fall back to the upstream exllamav3 kernels:

- the cooperative kernel for decode, up to 42 rows;
- the stock `exl3_moe` kernel for larger batches and prefill.

The cooperative kernel is compiled for K4. But GNmac's patches touch its 4-bit register path, and its bit-exact checks covered K2 and K3 only. So we treated it as unverified until it was tested:

| Check | K3 | K4 | Data |
|---|---:|---:|---|
| Mean abs (decode - prefill) logprob, 768 tokens | 0.0258 | 0.0211 | [`sc-k3.json`](../data/nll/sc-k3.json), [`sc-k4.json`](../data/nll/sc-k4.json) |
| p99 / max abs difference | 0.393 / 1.130 | 0.279 / 0.797 | same |
| Greedy token not rank 1 when rescored by prefill | 2.2 % | 2.6 % | same |
| Canary, first copy / second copy (nats per token) | 13.17 / 0.062 | 13.20 / 0.060 | same |

The cooperative K4 kernel behaves like the K3 path, and the copy test was 20/20 (run notes).

**Memory.** These figures come from run notes; the K4 startup log is not in `data/`:

| | Model load per card | KV pool at 0.97 |
|---|---:|---:|
| K3 graft | 53.24 GiB | 2,417,392 tokens |
| K4 (layers 0, 1, 3, 6) | 54.82 GiB | 1,670,368 tokens |

That is +1.58 GiB per card, and both figures match the review's prediction exactly.

**Result, K3 (A) vs K4 (B)**, long windows ([`data/nll/k3.json`](../data/nll/k3.json) vs [`data/nll/k4.json`](../data/nll/k4.json)):

| Corpus | n | PPL K3 | PPL K4 | dPPL % | Bootstrap 95 % CI | K4 better / worse | MDE % |
|---|---:|---:|---:|---:|---|---:|---:|
| fr | 64 | 2.1242 | 2.1250 | +0.040 | [-0.145, +0.230] | 31 / 33 | 0.274 |
| en | 64 | 1.3191 | 1.3212 | +0.160 | [-0.025, +0.374] | 35 / 29 | 0.293 |
| code | 64 | 2.9846 | 2.9858 | +0.040 | [-0.065, +0.145] | 31 / 33 | 0.152 |
| **all** | 192 | 2.0298 | 2.0314 | **+0.080** | **[-0.014, +0.183]** | 97 / 95 | 0.141 |

Short windows ([`data/nll/k3-short.json`](../data/nll/k3-short.json) vs [`data/nll/k4-short.json`](../data/nll/k4-short.json)):

| Corpus | n | dPPL % | Bootstrap 95 % CI | MDE % |
|---|---:|---:|---|---:|
| fr | 400 | +0.376 | [-0.097, +0.858] | 0.685 |
| en | 400 | -0.376 | [-0.914, +0.143] | 0.769 |
| code | 400 | -0.060 | [-0.471, +0.356] | 0.596 |
| **all** | 1,200 | **-0.021** | **[-0.297, +0.254]** | 0.396 |

The comparison against the second K3 run (`k3bis`) gives the same tables, because `k3` and `k3bis` are identical.

**Reading the result:**

- **Measured:** no effect. Every CI contains zero, and the pooled long-window CI rules out a gain larger than 0.014 %.
- **The prior prediction was wrong.** Before the run, the review predicted a gain of 0.15-0.35 % on code from these four layers, an inference from the alpha data and LibertAI's 2.9-to-3.5 gap. The measured code CI, [-0.065, +0.145], excludes that whole range.
- **Possible reasons** (all inference, none tested):
  - the error of K3 on these layers may already be small next to other error sources;
  - alpha is measured with Gaussian noise, not EXL3's error structure;
  - the K4 layers run through different kernels with their own rounding;
  - the memorized corpora compress the differences.

**Decision:** we went back to all-K3. The 4-bit layers cost 747k tokens of KV pool for nothing we could measure.

### 9.3 Original 2.9 pack vs the graft: a real, large gain, mostly in French

This compares the GNmac hybrid as published (2-bit layers 18-22) against our graft. The `k29` run came after the A-B-A above. The K3 baseline was bit-identical across three restarts before and after it, so the comparison has a zero null.

Long windows, A = 2.9 original, B = graft ([`data/nll/k29.json`](../data/nll/k29.json) vs [`data/nll/k3.json`](../data/nll/k3.json); printed in [`data/logs/05-ab-2.9orig-vs-graft.txt`](../data/logs/05-ab-2.9orig-vs-graft.txt)):

| Corpus | n | PPL 2.9 | PPL graft | dPPL % | Bootstrap 95 % CI | Graft better / worse | Sign p | Wilcoxon p | d top-1 (pts) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **fr** | 64 | 2.2231 | 2.1242 | **-4.451** | [-4.974, -3.934] | **64 / 0** | 1.1e-19 | 3.6e-12 | +1.303 |
| en | 64 | 1.3284 | 1.3191 | -0.701 | [-0.946, -0.477] | 50 / 14 | 7.1e-06 | 2.8e-07 | +0.205 |
| code | 64 | 2.9901 | 2.9846 | -0.184 | [-0.331, -0.039] | 36 / 28 | 0.38 | 0.058 | +0.047 |
| **all** | 192 | 2.0669 | 2.0298 | **-1.797** | [-1.996, -1.600] | 150 / 42 | 1.9e-15 | 2.7e-21 | +0.518 |

Short windows ([`data/nll/k29-short.json`](../data/nll/k29-short.json) vs [`data/nll/k3-short.json`](../data/nll/k3-short.json)):

| Corpus | n | dPPL % | Bootstrap 95 % CI |
|---|---:|---:|---|
| fr | 400 | -0.884 | [-1.607, -0.164] |
| en | 400 | -0.880 | [-1.479, -0.284] |
| code | 400 | +0.049 | [-0.426, +0.520] |
| **all** | 1,200 | **-0.573** | [-0.919, -0.221] |

**Measured:**

- Raising layers 18-22 from 2 to 3 bits lowers held-out perplexity by 1.8 % overall on long windows.
- The gain is 4.45 % on French, where **every one of the 64 windows** improves, and top-1 accuracy rises by 1.3 points.
- English gains 0.7 %. On code the effect is small, -0.18 %, with a sign test that is not significant.
- The copy test was 20/20 for both configurations ([`data/logs/05-ab-2.9orig-vs-graft.txt`](../data/logs/05-ab-2.9orig-vs-graft.txt)). That test is saturated and cannot separate them.

**The price is VRAM** ([`data/logs/05-ab-2.9orig-vs-graft.txt`](../data/logs/05-ab-2.9orig-vs-graft.txt), K3 figures from run notes):

| Configuration at 0.97 | Model load per card | KV pool |
|---|---:|---:|
| Original 2.9 pack | 51.3 GiB | 3,347,338 tokens |
| Graft | 53.24 GiB | 2,417,392 tokens |

The graft's pool still clears our 1.2M floor by a wide margin.

**Inference, untested:**

- LibertAI publishes the whole 2.9 pack at +0.4 % perplexity on prose against the original. If that figure comes from English text, it understates what the 2-bit layers cost in other languages.
- We do not know why French is hit so much harder. One hypothesis: layers 18-22 carry more of the load for languages that are rarer in the calibration data. We have no evidence for it.
- In sudoingX's error model, the 18-22 graft was predicted to be worth about twice the four-layer K4 upgrade (0.154 vs 0.068). The measured gap is far wider: -1.8 % against about 0.

**Decision:** the graft is the served configuration: all 40 layers at K3, official FP8 for everything else.

### 9.4 Utilization ceiling

| Utilization | KV pool | Stress peak per card (8 x 100k tokens) | Source |
|---|---:|---:|---|
| 0.97 | 2,417,392 (5.13 GiB per card) | 62,876-62,926 MiB | [`data/logs/04-stress-0.97-and-copytest.txt`](../data/logs/04-stress-0.97-and-copytest.txt) |
| 0.98 | 2,717,420 (5.76 GiB per card) | 63,556-63,580 MiB | [`data/logs/08-ceiling-0.98-smoke-stress-nll.txt`](../data/logs/08-ceiling-0.98-smoke-stress-nll.txt) |

About 64,911 MiB per card is usable. At 0.98, NLL scoring with `prompt_logprobs` on 1,024-token windows peaked at 64,534-64,676 MiB per card, leaving about 0.23 GiB. It did not run out of memory, and its results were bit-identical to those at 0.97 (`k3-098`). If you run the NLL harness on a server at 0.98, use shorter windows or drop to 0.97 first.

## 10. KV cache format: an open question

Neither recipe stores the KV cache the way DeepSeek's reference code does on sm_80, and they deviate in opposite ways.

What the reference (`inference/model.py` @ `1bae0e10`) does:

| Cache | Reference rounding |
|---|---|
| Sliding-window KV | FP8, one UE8M0 scale per 32 values over all 512 dims, RoPE included |
| Compressed KV | FP4 in groups of 16 with E4M3 scales |
| Indexer queries and keys | FP4 in blocks of 32 with E8M0 scales |

**GNmac (what we run).**

- **Format:** `--kv-cache-dtype fp8_ds_mla`, which the README calls "the model's native format". On sm_80 that is not the case: upstream vLLM writes the V4.1 record only on SM100. Other architectures get the **V4 record**: 448 NoPE dims in FP8 with one UE8M0 scale per 64 values, and the RoPE part kept in BF16. Indexer keys are FP8.
- **Precision against the reference:** coarser on the sliding window (one scale per 64 instead of per 32), finer on everything else (BF16 RoPE, FP8 instead of FP4 for compressed KV and index keys).
- **No switch exists.** On this image the sm_80 attention class forces `fp8_ds_mla`: `auto` and `fp8` are rewritten to it, and `bfloat16` fails at startup with a `ValueError`. The reference format (`nvfp4_ds_mla`) requires SM100.
- **Inference:** per-64 instead of per-32 power-of-two scales mostly hurt elements more than about 2^15 below their block's maximum, which should be negligible. This is not measured. GNmac only ever compared builds with each other, never with the original model.

**Morrowmake** ([Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe), compared in [`comparison-morrowmake.md`](comparison-morrowmake.md)):

- Uses `--kv-cache-dtype bfloat16`, but the cache rows hold **exactly the reference's quantize-dequantize values**: window FP8/32/UE8M0, compressed FP4/16/E4M3, index keys FP4/32/E8M0.
- Reproduces the reference indexer, and tests its rounding emulation bit for bit against DeepSeek's unmodified reference code on the official weights.
- On paper this is closer to the reference than GNmac. It costs pool size: 1,247,406 tokens against GNmac's 1,395,731 at 262k context in their published configurations.
- We could not run it. It downloads the full official checkpoint plus the expert pack, about 680 GiB of preflight requirement, and we had 575 GiB available.

**Status:** open. We do not know how much GNmac's mixed-direction KV format changes the output. Measuring it needs either the Morrowmake stack on the same weights or reference logits.

## 11. DSpark: lossless, but too expensive in VRAM for us

DSpark is the model's own speculative drafter. MTP speculation is not available for V4.1 in this vLLM (refused in `speculative.py`), so DSpark is the only drafter.

**Why it is lossless (from reading the code, not measured):**

- Drafts are argmax tokens, verified with standard rejection sampling, which treats a greedy draft as a one-hot proposal.
- So greedy output is the same token sequence, and sampled output keeps the target model's distribution.
- GNmac's confidence gate force-rejects any draft it skips, and it acts only on all-greedy batches.

So dropping DSpark was a VRAM decision, not a quality one.

**Measured cost** with `SPEC_TOKENS=5`, graft, utilization 0.97 ([`data/logs/06-dspark-kv-bench-copytest.txt`](../data/logs/06-dspark-kv-bench-copytest.txt), [`data/logs/07-dspark-stress-vram.txt`](../data/logs/07-dspark-stress-vram.txt), [`data/bench/dspark-decode.log`](../data/bench/dspark-decode.log)):

| | Without DSpark | With DSpark |
|---|---:|---:|
| Model load per card | 53.24 GiB (run notes) | 55.54 GiB |
| KV memory per card | 5.13 GiB | 2.35 GiB (**-2.78 GiB**) |
| KV pool | 2,417,392 | **1,113,781** (below our 1.2M floor) |
| 1 user, structured / code / prose (tok/s) | 105.4 / 105.0 / 104.8 | 369.8 / 267.3 / 200.1 |
| 8 users, structured / code / prose (tok/s) | 559.7 / 482.4 / 449.2 | 852.9 / 450.3 / 437.8 |
| Draft acceptance, median (structured / code / prose) | n/a | 0.994 / 0.70-0.71 / 0.43-0.53 |
| Stress peak per card, 8 x 100k tokens | 62,876-62,926 MiB | 63,652-63,808 MiB |
| Copy test | 20/20 | 20/20 |

The budget research had estimated the drafter at about 2.18 GiB per card; the measured cost is 2.78 GiB. DSpark gives a large single-user speedup but almost nothing at 8 users: structured +52 %, code -7 %, prose -2.5 %. It also halves the KV pool, so we serve without it. At 0.97 the speed figures without DSpark are those of the 0.95 runs in the table: the setting changes the pool, not the kernels.

## 12. Other deviations from the reference that we know about

These come from a source-level audit of both stacks against the official reference code. None of them has been measured on our side.

| Item | Our stack (GNmac) | Reference | Direction |
|---|---|---|---|
| Routed experts | EXL3 K3 (LibertAI) | MXFP4 | lossy, by far the largest deviation |
| FP8 dense matmuls | Weights dequantized exactly to BF16; W8A16 with FP32 accumulation | Activations quantized to FP8 in blocks of 32 | more precise than the reference (inference) |
| Engram | FP8 rows times E8M0, computed in FP32 and rounded once to BF16 | same | exact (once the tables are actually loaded) |
| Decode all-reduce | FlashInfer PCIe IPC ring with BF16 adds per hop | FP32 | slightly lossy; no switch |
| Prefill all-reduce | FP32 sum in rank order, rounded once | FP32 | equivalent |
| Router | BF16 inputs with FP32 accumulation; official FP32 bias | FP32 | stock-vLLM level |
| KV cache | `fp8_ds_mla` V4 record | see section 10 | mixed |
| Sampling | All our measurements are at temperature 0 | The official card recommends temperature 1.0 and top_p 1.0 | Mia's card says top_p 0.95; the official card says 1.0 |

## 13. Mistakes and dead ends

We list these because others running this stack are likely to hit the same ones.

1. **Engram "host" mode (our configuration error).** This is the most serious one. For the first hours of the deployment, the model ran with two layers silently disabled. Every number from that period is invalid: the 15/20 copy test and the first speed benchmarks, kept as `data/bench/INVALID-*`. Upstream never claimed that mode worked. We assumed it. See sections 6 and 7.
2. **A plausible benchmark is not a correctness check.** The invalid speed numbers matched upstream's and looked fine. That is exactly the pattern we had learned to distrust from earlier benchmarking work: a pleasing number gets less scrutiny than a disappointing one.
3. **The first copy-test replay was confounded.** The "solo" replays reused prefix-cache KV from the batched run. We now report `cached_tokens` on every copy test.
4. **An in-process A/A pair as noise floor.** We planned it, and the review caught it before we ran it. A-B-A across restarts replaced it (section 8.3).
5. **2,048-token NLL windows.** The first design would have risked running a card out of memory, because the logits spike is not in vLLM's memory profile. We used 1,024 tokens instead.
6. **A sign-test overflow** in our own NLL tool for n = 400 (`math.fsum` on binomial terms). We found it only by running the tool, and fixed it with exact integers.
7. **The K4 prediction.** The expected 0.15-0.35 % code gain did not appear (section 9.2). The fourth layer also changed between the first pick and the review (5, then 3).
8. **Budget estimates.** The pool for the graft without drafter was predicted at about 1.5M tokens and measured at 1.81M. The drafter was estimated at about 2.18 GiB per card and measured at 2.78. Both errors were within or near the stated uncertainty, but they are not small next to a 1.2M floor.
9. **Mia as the first plan.** We picked Mia from its card's KL. The format check showed it was a drop-in replacement, and the upstream author's in-hybrid test showed it was worse (section 4).
10. **llama.cpp UD-Q8_K_XL.** It is still the most faithful on paper. We set it aside for its known Engram bugs, not because of anything we measured.

## 14. Open questions

1. **Distance to the original model.** Everything above is relative between configurations of one stack. We never measured KL against the original model or DeepSeek's reference. One feasible route, proposed in review but not done:
   - generate top-k reference logits from the most faithful available build (llama.cpp `--kl-divergence-base` with UD-Q8_K_XL);
   - delete it to free the disk;
   - score our stack against those logits with top-20 `prompt_logprobs`.
2. **KV format** (section 10): what GNmac's V4 record on sm_80 costs, or gains, compared with the reference rounding that Morrowmake reproduces.
3. **Mia EAQ v1 inside the hybrid, on our harness and corpora.** It would confirm or overturn the upstream +1.55 % NLL measurement. That figure was taken on an unstated passage set, and we did not reproduce it.
4. **Why French gains 4.45 % from the graft** while code gains 0.18 %. Would other languages behave like French?
5. **Per-expert mixed-width packs** (sudoingx, HeNryous, philbert440) measure 25-35 % lower KL than uniform packs on independent panels. They need mixed-width or 4-5-bit kernels that the fast path does not have.
6. **Native MXFP4 experts for a few layers inside the hybrid.** The stack already runs the official FP4 drafter experts on sm_80. A main layer would cost 6.72 GiB against 4.76 GiB at K3, with exact weights. This is untested engineering.
7. **The non-determinism of the K4 prefill path** within one process (section 8.6). Its cause is unconfirmed.
8. **Agentic behaviour.** The upstream issue reports of misspelled tool and path names predate our Engram finding. We do not know whether other deployments ran with empty tables. Our copy tests pass, but a copy test is not an agentic harness.
9. **Sampling at temperature 1.0 / top_p 1.0** (the official recommendation) is untested. All our quality numbers are at temperature 0, or teacher-forced.

## 15. Reproducing the numbers

All NLL results can be recomputed from the shipped run files, with no GPU:

```
python3 tests/nll_eval.py compare data/nll/k3.json  data/nll/k3bis.json   # null: identical
python3 tests/nll_eval.py compare data/nll/k3.json  data/nll/k4.json      # 4-bit layers 0,1,3,6
python3 tests/nll_eval.py compare data/nll/k29.json data/nll/k3.json      # 2.9 original vs graft
python3 tests/nll_eval.py compare data/nll/k3-short.json data/nll/k4-short.json
python3 tests/nll_eval.py compare data/nll/k29-short.json data/nll/k3-short.json
python3 tests/nll_eval.py compare data/nll/k3.json  data/nll/k3-098.json  # 0.97 vs 0.98: identical
```

The run files keep per-token logprobs and the SHA-256 of every window's token ids. Rebuilding the corpora against a live server (`nll_eval.py build`) downloads the Gutenberg texts and reads the GNmac source at `2ad5741ab1`. Because the windows are cut with the server's own tokenizer, the hashes will match only with the same tokenizer.

On a live server, the order we used was:

1. `build` (long and short sets);
2. `run` K3 (long and short), then `selfcheck`;
3. restart with the candidate weights; same three commands;
4. restart on K3; same commands again;
5. `compare`.

Run nothing else on the server during a run, and keep every environment variable identical between configurations except the weights.

## 16. Credits

- [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) @ `2ad5741ab1`: the recipe, image, kernels and documentation this work builds on, including the tuning-log entry on Mia, the causality canary and the paired-NLL practice. The code corpus is its source. Recipe MIT, except `patches/vllm-exl3/`, which is AGPL-3.0-only.
- [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) @ `0be63c33` and [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) @ `04a492fc`: the routed experts, including the grafted layers.
- [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10`: the model, its reference inference code and the Engram tables.
- Curtis White ([CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX)): the hybrid checkpoint recipe and its `quantization_config.json`.
- [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe): the alternative recipe we compared against, and the source of the clearest account of reference KV rounding on sm_80.
- [Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw): the alternative expert pack we evaluated on paper. [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks): the decode benchmark protocol used for every speed figure here.
- [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41): per-layer alpha sensitivity data for the 1017 checkpoint, which drove the K4 layer choice.
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format and kernels, including the cooperative MoE kernel that ran the 4-bit layers.
- [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3): the EXL3 plugin for vLLM (AGPL-3.0-only).
- [zebgop-ops/dsv41reap-pp](https://github.com/zebgop-ops/dsv41reap-pp): the sm_80 V4.1 model code and the Engram SSD row store that our fix relies on.
