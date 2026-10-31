# DeepSeek-V4.1-Flash-1017 on 4x CMP 170HX, fidelity first

[Version française](README.fr.md)

We serve [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on four
NVIDIA CMP 170HX cards (sm_80, 64 GB each, PCIe Gen2 x16), and we choose fidelity to the original model over raw
speed. Every choice below was measured on this machine against the alternative, with paired perplexity on frozen
text and with the raw data published.

What we serve now (2026-10-31):

- **[Morrowmake](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe)'s engine, release 1.0.0**: it
  reproduces the reference model's rounding of the attention caches, and that shows from 32k tokens on.
- **The routed experts of layers 18-22 grafted from 2 to 3 bits**: the largest quality gain we found.
- **A lossless compact KV cache of our own**: the same bits as Morrowmake's BF16 cache, in half the memory.
- **DSpark speculative decoding on**: it generates the same text, 2 to 3.7 times faster for one user.

The repository holds patches, tools, tests, raw measurement data and documentation. It holds no weights; its only
engine code is the compact KV cache, as a patch against Morrowmake's engine.

## At a glance

| | Measured |
|---|---|
| Quality against our previous production (GNmac engine with the same graft) | **-1.64 %** perplexity on 32k-token code windows (16 of 16 better), **-0.58 %** on 32k prose; +0.14 % on 1,024-token windows |
| Quality gain of the 3-bit graft (layers 18-22) | **-1.81 %** perplexity overall, **-4.45 %** on French, -2.31 % on 32k prose |
| Compact KV cache | **bit-identical** prompt log-probabilities on 1.39 M tokens; KV pool **x1.99** (1.60 M → 3.19 M tokens without DSpark, 1.36 M with it); decode -2 to -5 % |
| Speed, one user (MiaAI protocol, structured / code / prose) | **267 / 212 / 153 tok/s** with DSpark, 72 without |
| Speed, 8 users, aggregate | **570 / 291 / 349 tok/s** with DSpark |
| Long prompts | time to first token **93 s at 259k tokens** |
| 1M-token context (separate profile, no DSpark) | exact recall of 8 hidden codes: **8 / 8 at 997k tokens** (first token after 12 min); KV pool 4.2 M tokens ([results](docs/results.md#5-contexts-beyond-262144-tokens)) |

## Quality comparisons

[`tests/nll_eval.py`](tests/nll_eval.py) scores the same frozen windows on each configuration (French and English
public-domain books, Python code) and compares them window by window. These scoring runs reproduce bit for bit
across server restarts, so a difference below is not restart noise; the bootstrap 95 % CI says how sure it is.
Negative means B predicts the text better. "32k prose" has 8 French, 8 English and 3 code windows.

| A → B | 1,024-token windows | 40-token windows | 32k prose | 32k code |
|---|---|---|---|---|
| **Our previous production → what we serve** | +0.14 % [+0.03, +0.26] | -0.21 % [-0.52, +0.11] | **-0.58 % [-0.74, -0.43]** | **-1.64 % [-1.92, -1.37]** |
| Engine only, same weights (GNmac → Morrowmake) | +0.15 % [+0.02, +0.29] | -0.29 % [-0.59, +0.02] | **-0.57 % [-0.73, -0.42]** | not run |
| Graft of layers 18-22, 2 → 3 bits (Morrowmake engine) | **-1.81 % [-2.01, -1.61]** | -0.49 % [-0.84, -0.13] | **-2.31 % [-2.75, -1.79]** | -0.06 % [-0.12, -0.00] |
| Compact KV cache (BF16 rows → packed rows) | identical | identical | identical | identical |
| Layers 36-39, 3 → 4 bits | -0.04 % [-0.08, +0.01] | -0.08 % [-0.22, +0.07] | -0.11 % [-0.15, -0.07] | -0.11 % [-0.14, -0.09] |
| Layers 0, 1, 3, 6, 3 → 4 bits (GNmac engine) | +0.08 % [-0.01, +0.18] | -0.02 % [-0.30, +0.25] | not run | not run |
| Release shortcuts off (Morrowmake) | +0.04 % [-0.05, +0.13] | not run | not run | +0.00 % [-0.04, +0.05] |

What the table says:

- **The weights matter most, and the graft is where it pays.** Layers 18-22 are the only 2-bit layers of the
  published pack. Moving them to 3 bits gains 1.8 % overall and 4.5 % on French, in both engines. Moving further
  layers from 3 to 4 bits gains about 0.1 % at best: the experts are at a plateau.
- **The engine matters on long contexts.** At equal weights, Morrowmake's engine is 0.15 % worse than GNmac's on
  1,024-token windows and 0.6 % better on 32k (2 % better on the three code windows of that set). *Inference:* it reproduces the reference's FP4
  rounding of the compressed cache and of the indexer keys, where GNmac stores FP8 records; the indexer picks which
  past positions each token reads, and that choice gets harder with more positions.
- **We picked the long contexts.** Against our previous production, the current configuration is 1.6 % better on
  long code and 0.6 % better on long prose, and 0.14 % worse on short windows. Agentic and coding work happens in
  long contexts.
- **The compact KV cache costs no quality at all**: not a small change, no change, on every token we scored.

Details, per-language rows and the data files: [docs/results.md](docs/results.md#2-quality).

## Our optimizations

1. **3-bit graft of layers 18-22.** The routed experts of these five layers come from LibertAI's 3.5 bpw pack
   instead of the 2.9 bpw one (same tensor names, 3 bits instead of 2). Cost: 1.98 GiB per card. Tool:
   [`morrowmake/tools/graft_layers.py`](morrowmake/tools/graft_layers.py), which checks the tensor sets and the bit
   width (refusing anything but the expected one) before it rewrites the index.
2. **Lossless compact KV cache** ([docs/compact-kv-cache.md](docs/compact-kv-cache.md)). The reference already
   rounds the compressed latent and the indexer keys to 4-bit floats with one scale per block, so we store the
   4-bit codes and the scale bytes and decode them back to the same bits. The latent and the indexer key of a
   position share one 512-byte row (1,280 bytes per token per card instead of 3,200). Three versions:
   - v1: exact, but -5 to -9 % decode and -10 to -35 % prefill;
   - v2: vectorised decoding, and attention through a BF16 copy of each query's selected rows, so Morrowmake's own
     kernels run unchanged and identity holds by construction (decode -2 to -5 %, prefill -6 to -10 %);
   - v1.5: the indexer keys move into the latent rows (another +16 % of pool).
3. **The engine, chosen on long contexts.** Section above; the first comparison, at 1,024 tokens, would have
   picked the other engine ([docs/comparison-morrowmake.md](docs/comparison-morrowmake.md)).
4. **DSpark kept on.** Same 768 generated tokens in the self-check, decode log-probabilities within 2e-5; single-user
   speed x2.1 to x3.7. The no-DSpark profile stays available for a 3.19 M-token pool.
5. **Contexts up to 1M tokens, as a separate profile.** No DSpark, `--max-model-len 1048576`: 8 of 8 hidden codes
   recalled exactly at 131k, 522k and 997k tokens. A 1M maximum makes every request 21-25 % slower, so it is not
   the default.
6. **Measured and rejected:** 4-bit experts (two tests, at most 0.1 %, -12 to -17 % speed); the release's
   numerical shortcuts turned off (no measurable change, 2.3x slower); memory setting 0.98 (same bits, runs out of
   memory) ([docs/results.md, section 6](docs/results.md#6-settings-we-tried-and-rejected)).

## Speed

MiaAI-Lab protocol (temperature 0, thinking off, 400-token cap), tok/s, structured / code / prose:

| | 1 user | 8 users, aggregate |
|---|---|---|
| **What we serve** (Morrowmake + graft + compact cache + DSpark) | **267 / 212 / 153** | **570 / 291 / 349** |
| Same, no DSpark | 72 / 73 / 72 | 397 / 245 / 341 |
| Our previous production (GNmac + graft, no DSpark) | 105 / 105 / 105 | 560 / 482 / 449 |
| GNmac + graft + DSpark | 366 / 268 / 202 | 884 / 563 / 476 |

**GNmac's engine is faster**, by about 1.45x for one stream without DSpark and 1.3-1.4x with it. Compare the
structured and prose columns: the code prompt's answer is 48 tokens long on Morrowmake and 95 on GNmac, so the code
column does not compare across engines. We traded that speed for the long-context quality above. Morrowmake reads long prompts 1.4-1.7x faster: the first token of a
259k-token prompt comes after 93 s instead of 159 s. Single-stream decode at long contexts and the KV pools:
[docs/results.md](docs/results.md#4-speed).

## Configuration in production

| Item | Value |
|---|---|
| Hardware | ASRock Rack ROMED8-2T, AMD EPYC 7F52, 512 GB DDR4-3200 (8 channels), 4x CMP 170HX 64 GB on PCIe Gen2 x16, 200 W, 170tune +200 MHz @ 1500 MHz |
| Driver | NVIDIA open driver 610.43.02 with a cmpunlocker fork and a P2P patch; Morrowmake's P2P check passes without its TRAP31 patch |
| Engine | Morrowmake 1.0.0 (`7f9023cd`), image `ghcr.io/morrowmake/vllm-cmp170hx@sha256:f67d9be4…` (engine `a0c57b52`), unmodified except the seven compact-cache files mounted over it |
| Weights | Official FP8 non-expert tensors and Engram tables (`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`); routed experts from LibertAI's 2.9 bpw pack (Morrowmake's pinned revision), layers 18-22 from LibertAI's 3.5 bpw pack @ `04a492fc` |
| KV cache | compact v1.5 (`VLLM_SM80_COMPACT_KV=1`), prefix caching on |
| Serving | TP 4, 262,144-token context, 8 sequences, DSpark (7 draft tokens), memory setting 0.95, Engram tables in RAM |
| KV pool | 1,360,109 tokens (5.2 full-length requests); 3,189,000 without DSpark |
| API | tool calls and reasoning parsers `deepseek_v41`, API key required, `/health` open |

## Repository layout

```
morrowmake/                       the current configuration, on top of Morrowmake 1.0.0
  compact-kv/                     lossless compact KV cache (engine files Apache-2.0)
    compact-kv.patch              changes to six engine files
    vllm/.../ops/sm80_kv_pack.py  packing, unpacking, Triton loaders and stores
    build-overlay.sh              builds kvc-overlay/ from the pinned image + the patch
    tests/test_gpu.py             bit-identity unit test on GPU; tests/bench_gpu.py: kernel timings
  tools/graft_layers.py           grafts layers from another EXL3 pack into the export checkpoint
  start.sh.patch                  mounts kvc-overlay/ over the image's files
  prepare.sh.patch                hybrid index hash of the 18-22 graft
  env.example, start-dspark.sh, start-nospec.sh
recipe/                           the previous configuration, on top of GNmac-lab 1.2.0
tests/
  nll_eval.py                     paired NLL / perplexity (build, run, compare, selfcheck)
  identical.py                    bit-level identity of two runs
  speed.py                        decode and prefill per stream from 3.5k to 259k tokens
  longctx.py                      exact recall of 8 codes hidden in a long document (up to 1M tokens)
  copytest-*.py, stress-dsv41.py  exact-copy, determinism and stress tests
data/
  nll/                            every scored run (long/, codex/ = 32k prose and code; .gz = gzip JSON)
  bench/                          speed runs (speed/), MiaAI outputs (mia-json*/), older logs
  logs/                           experiment logs of the first round
docs/
  results.md                      every measurement of the current configuration
  compact-kv-cache.md             design and validation of the compact cache
  setup.md                        step-by-step rebuild
  comparison-morrowmake.md        the two engines, measured side by side
  quality-research.md, agentic-methodology.md, results-gnmac.md, setup-gnmac.md
```

## Quick start

Full steps with checks: [docs/setup.md](docs/setup.md).

1. Install Morrowmake 1.0.0 (`7f9023cd`) and run it once as released.
2. Download the five donor shards of LibertAI's 3.5 bpw pack, run `morrowmake/tools/graft_layers.py`, apply
   `prepare.sh.patch`, rebuild the hybrid.
3. Run `morrowmake/compact-kv/build-overlay.sh`, apply `start.sh.patch`, set `VLLM_SM80_COMPACT_KV=1`, run the GPU
   unit test.
4. `./start-dspark.sh`. The log must report a KV pool of 1,360,109 tokens.

## The previous configuration

Until 2026-10-31 we served the [GNmac-lab recipe](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx)
1.2.0 with the same graft, no DSpark and a 0.98 memory setting. It is still documented in [`recipe/`](recipe/),
[docs/setup-gnmac.md](docs/setup-gnmac.md) and [docs/results-gnmac.md](docs/results-gnmac.md). Its findings still
hold, in particular:

- **The Engram tables can silently fail to load.** In GNmac's "host" mode the tables were reserved but never
  filled; the model kept working, with corrupted identifiers and non-deterministic output. Decode speed does not
  show it; an exact-copy test does ([docs/agentic-methodology.md, section 7](docs/agentic-methodology.md#7-failure-analysis-the-engram-tables-were-empty)).
- **An A-B-A design across restarts gives a zero noise floor** on these stacks, which makes small differences
  measurable.

## How the work was done

An AI coding agent (Claude Code) ran the research, the experiments and the measurements; a human operator made
every decision. Critic passes checked the research reports, and every claim that drove a decision was measured.
[docs/agentic-methodology.md](docs/agentic-methodology.md) describes the method, what the critics caught and the
measurement pitfalls we guard against. [docs/quality-research.md](docs/quality-research.md) covers the first
round of fidelity research (quantization options, VRAM budgets, the Engram diagnosis).

## Credits

- [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe) 1.0.0 and
  [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx): the engine and the recipe we serve.
- [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) 1.2.0:
  the recipe of our first configuration.
- [LibertAIDAI](https://huggingface.co/LibertAIDAI): the EXL3 packs,
  [2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) and
  [3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) @ `04a492fc`.
- [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10`:
  the model.
- Curtis White ([CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX)):
  the hybrid checkpoint layout.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks): the decode benchmark protocol (not
  included here).
- [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41): per-layer sensitivity
  data.
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) and [vLLM](https://github.com/vllm-project/vllm).

## Licensing

- **This repository** is MIT ([LICENSE](LICENSE)), **except the engine files of `morrowmake/compact-kv/`**
  (`compact-kv.patch` and `sm80_kv_pack.py`), which modify Morrowmake's Apache-2.0 vLLM fork and are Apache-2.0
  ([their LICENSE](morrowmake/compact-kv/LICENSE)).
  [NOTICE.md](NOTICE.md) lists every file derived from another project.
- **Weights** are not included: DeepSeek's weights and LibertAI's packs are MIT, under their owners' terms.
- The previous configuration's engine image (GNmac) contains AGPL-3.0 code; see [NOTICE.md](NOTICE.md) if you build
  it.

## Disclaimer

A homelab project: one machine, one set of measurements. We are not affiliated with DeepSeek, NVIDIA, Morrowmake,
GNmac-lab, LibertAI or any other project named here. CMP 170HX cards need a modified driver to work as described;
check the terms that apply to you. Every number is what we measured on our hardware, with its data file. The
perplexity changes are relative between our configurations: we did not measure the distance to the unquantized
model.
