# DeepSeek-V4.1-Flash-1017 on 4x CMP 170HX, fidelity first

[Version française](README.fr.md)

This repository documents how we serve [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on four NVIDIA CMP 170HX cards (sm_80, 64 GB each, PCIe Gen2 x16). The configuration favours fidelity to the original model over raw speed. It builds on the [GNmac-lab recipe](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) at commit `2ad5741ab1` and makes four changes:

- Routed experts of layers 18-22 go from 2 bits to 3 bits.
- No DSpark drafter.
- The Engram tables are actually loaded. Our first deployment ran without them, and nothing reported it.
- `gpu_memory_utilization` is 0.98.

Measured results:

- A copy test of 40 random identifiers per list, run on 100 lists in 3 passes, came back **300/300 exact**. Repeated requests gave byte-identical output.
- Grafting 3-bit layers 18-22 lowers perplexity by **1.80 % overall** and by **4.45 % on French text** against the published 2.9 bpw pack. Both bootstrap 95 % CIs exclude zero.
- The KV pool holds **2,717,420 tokens** at a 262,144-token context.
- Decode runs at about **105 tok/s for one user** and **449-560 tok/s aggregate for 8 users**.
- Cold prefill runs at **2,514 tok/s** (median, 24-38k-token prompts).

The repository contains only patches, scripts, test tools and raw measurement data. It contains no weights and no engine code.

## Configuration in production

| Item | Value |
|---|---|
| Hardware | ASRock Rack ROMED8-2T, AMD EPYC 7F52, 512 GB DDR4-3200 (8 channels), 4x CMP 170HX 64 GB on PCIe Gen2 x16 |
| GPU software | NVIDIA open driver 610.43.02 with a cmpunlocker fork and a P2P patch (P2P on). Cards capped at 200 W, 170tune +200 MHz @ 1500 MHz |
| Runtime | Upstream recipe 1.2.0 (`2ad5741ab1`). Docker image built locally from its `docker/` directory, run in an LXC container. GPUs passed with CDI (`--device nvidia.com/gpu=all`) |
| Layout | Tensor parallel 4 (`LAYOUT=tp4`), `MAX_MODEL_LEN=262144`, `MAX_NUM_SEQS=8` |
| Routed experts | [LibertAIDAI EXL3 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) @ `0be63c33`, 3 bits per weight. Layers 18-22 are replaced by the 3-bit experts of [LibertAIDAI EXL3 3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) @ `04a492fc` (shards `00020`-`00024-of-00044`). All 40 layers are 3-bit |
| Everything else | Official FP8 non-expert tensors and Engram tables from `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`. Hybrid layout and `quantization_config.json` from Curtis White |
| Engram tables | `ENGRAM_STORAGE=ssd`, read from a 200 GiB tmpfs in RAM (`recipe/engram-ram.sh`) that holds 2 x ~101.5 GB shards (about 189 GiB). **Never leave `ENGRAM_STORAGE` empty** (see Key findings) |
| KV cache | `fp8_ds_mla` (upstream default), prefix caching on, `--enable-prompt-tokens-details` so `cached_tokens` is visible |
| Memory | `GPU_MEMORY_UTILIZATION=0.98`, giving **2,717,420 tokens** of KV (5.76 GiB per card, 10.37 full 262k sequences) |
| Speculative decoding | Off (`SPEC_TOKENS=0`) |
| Vision, tool calls | On. Smoke test passes (chat, `get_weather` tool call, image) |
| API | Bound to `127.0.0.1`, no key by default. Add `--api-key` through `EXTRA_ARGS` before you expose it |

The full environment file is [`recipe/env.example`](recipe/env.example). The production log at 0.98 is [`data/logs/08-ceiling-0.98-smoke-stress-nll.txt`](data/logs/08-ceiling-0.98-smoke-stress-nll.txt).

**Margin at 0.98.** Normal serving peaked at 63,580 MiB per card during an 8 x 100k-token stress run, out of 64,911 MiB usable, so about 1.3 GiB was left. NLL scoring with `prompt_logprobs` on 1,024-token windows peaked at 64,676 MiB, which left about 0.23 GiB. It did not run out of memory, but if your workload uses `prompt_logprobs` on long windows, use 0.97 (2,417,392 tokens) or windows of 512 tokens or fewer.

## Key findings

- **The Engram "host" mode failed silently. This was our configuration mistake.** We left `DSV41_ENGRAM_STORAGE` unset so that the tables would stay in RAM. In that mode the stack reserves 2 x 23.6 GiB of pinned memory per rank but never fills it:
  - The hybrid index contains only the small Engram tensors (`q`, `k`, `wkv`), not `engram.embed.*`.
  - vLLM skips its missing-weights check when a quantization method is set.
  - The upstream author never tested this mode. Their documented path is `ssd`.
  - Engram layers 1 and 14 most likely acted as identity functions. We infer that the reserved memory held zeros but did not dump it to check.
  - Symptoms: 15/20 exact copies, identifiers with dropped `_` separators, and output that changed between runs at temperature 0. That run's copy-test output was not kept; the startup log is [`data/logs/01-first-start-engram-NOT-loaded.txt`](data/logs/01-first-start-engram-NOT-loaded.txt), and the decode/prefill benchmarks from that run are kept as `data/bench/INVALID-no-engram-*.log`.
  - The fix was `ssd` mode reading from a tmpfs. After it, the copy test scored 20/20 and then 100/100 on all three passes, and 5 tasks x 5 solo repeats gave byte-identical output ([`data/logs/03-copytest-100-and-determinism.txt`](data/logs/03-copytest-100-and-determinism.txt)).
  - Upstream benchmarks do not catch this failure: decode speed is essentially unchanged without Engram.
- **Grafting 3-bit experts into layers 18-22 is a real, significant fidelity gain.** We scored both configurations with paired NLL on frozen corpora: 64 windows of 1,024 tokens per set. Perplexity changes, with bootstrap 95 % CI:

  | Set | Perplexity change | 95 % CI | Windows better |
  |---|---|---|---|
  | French | **-4.45 %** | [-4.97, -3.93] | 64 of 64 |
  | English | -0.70 % | | |
  | Code | -0.18 % | | |
  | All | **-1.80 %** | [-2.00, -1.60] | |

  - Short 40-token windows: -0.57 % overall.
  - The cost is VRAM. The original 2.9 pack loads 51.3 GiB per card and leaves 3,347,338 KV tokens at 0.97. The graft leaves 2,417,392 at 0.97.
  - Data: [`data/logs/05-ab-2.9orig-vs-graft.txt`](data/logs/05-ab-2.9orig-vs-graft.txt), `data/nll/k29.json` vs `data/nll/k3.json`.
  - *Inference:* LibertAI's published figures put the full 2.9 pack at only +0.4 % perplexity on prose versus the original. If those figures come from English text, they understate what the 2-bit layers cost in other languages.
- **Adding 4-bit layers gave no measurable gain.** We moved layers 0, 1, 3 and 6 to 4 bits (from the 3.5 bpw pack). We picked these layers using sudoingX's per-layer sensitivity data.
  - Long windows: +0.08 % perplexity, CI [-0.01, +0.18]. Short windows: -0.02 %, CI [-0.30, +0.25].
  - The 4-bit cooperative kernel itself worked: the self-check and the causality canary passed, and the copy test scored 20/20.
  - The change costs about 1.58 GiB per card (model load 53.24 to 54.82 GiB per card, from our run notes), and the KV pool falls from 2,417,392 to 1,670,368 tokens at 0.97, so we went back to all 3-bit.
  - Data: `data/nll/k4*.json`, `data/nll/sc-k4.json`.
- **DSpark is a trade-off we declined.** The drafter takes 2.78 GiB per card, and the KV pool falls to 1,113,781 tokens, below our floor of 1.2M for multi-agent use.
  - Single-user speed rises x3.5 on structured output, x2.5 on code and x1.9 on prose.
  - At 8 users the gain mostly disappears: structured +52 %, code -7 %, prose -2.5 %.
  - The acceptance ratio was 99.4 % on the structured (counting) prompt, but only about 70 % on code and 43-53 % on prose ([`data/bench/dspark-decode.log`](data/bench/dspark-decode.log)).
  - MTP is not available for V4.1 in this vLLM; only DSpark is.
  - DSpark is lossless in principle (standard rejection sampling, by code reading). We dropped it for the KV pool, not for quality.
  - Data: [`data/logs/06-dspark-kv-bench-copytest.txt`](data/logs/06-dspark-kv-bench-copytest.txt), [`data/logs/07-dspark-stress-vram.txt`](data/logs/07-dspark-stress-vram.txt).
- **Results reproduce bit-exactly across restarts in the prefill path** (all-3-bit configuration). We ran an A-B-A design: K3, then K4, then K3 again, each after a server restart. The two K3 runs produced bit-identical prompt logprobs on all 192 long windows. So did K3 at 0.97 against K3 at 0.98.
  - The upstream docs say greedy text is not bit-reproducible across restarts in TP4 decode. Our cross-restart evidence covers the prefill (teacher-forcing) path only.
  - Within one process, repeated decodes at temperature 0 were byte-identical.
  - The 4-bit trial was the exception: its long-window prefill rescoring was not bit-exact even within one process (4 of 6 repeated windows differed, `repeat` field of `data/nll/k4.json`).
  - Data: `data/nll/k3.json`, `k3bis.json`, `k3-098.json`. Run `tests/nll_eval.py compare` to reproduce the comparison.
- **The full 3.5 bpw pack does not fit, so the graft is the affordable part of it.** Its experts need about 9.8 GiB more per card than the 2.9 pack, and no variant leaves room for a usable KV pool. The upstream fast MoE kernels take only 2-bit or 3-bit layers, with a single width per layer.
- **We kept LibertAI over Mia-AiLab EAQ-v1 2.9 bpw.** The two packs are interchangeable at the format level. The upstream author measured Mia's experts in this same hybrid at +1.55 % NLL. Mia's better KL comes from its dense layers, and the hybrid replaces those with official FP8.

## How the work was done

- [`docs/quality-research.md`](docs/quality-research.md): the fidelity audit and the quantization options. It covers why 3 bits on layers 18-22, why not 4 bits, the 3.5 bpw VRAM budget, the Engram diagnosis, and the NLL method (A-B-A across restarts, paired statistics, frozen token windows).
- [`docs/agentic-methodology.md`](docs/agentic-methodology.md): how the research was run. An AI coding agent (Claude Code) ran parallel read-only research workflows. A critic pass checked each report, and every deployment decision was left to the human operator. The doc also covers the measurement rules we follow: unique prompts and checked `cached_tokens` against prefix-cache contamination, warm-up passes, no acceptance ratios from degenerate text, and a re-measured baseline for any comparison. It explains why we test for identifier corruption, which matters for agentic tool use.
- [`docs/results.md`](docs/results.md): all measurements with their data files.
- [`docs/comparison-morrowmake.md`](docs/comparison-morrowmake.md): comparison with the [Morrowmake recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe). On paper it is more faithful in some respects (a `bfloat16` KV cache, tested bit-exact against the reference code). We did not choose it because its download (about 671.5 GiB, 680 GiB required by its preflight check) did not fit our disk.

### Speed (no DSpark, Engram in RAM, 200 W)

Measured with the MiaAI-Lab decode protocol: temperature 0, thinking off, 400-token cap. The rates are in tok/s for the structured, code and prose prompts. These runs used `gpu_memory_utilization` 0.95; the setting changes the KV pool size, not the kernels.

| | Structured | Code | Prose |
|---|---:|---:|---:|
| 1 user, streaming | 105.4 | 105.0 | 104.8 |
| 2 users, aggregate | 174.3 | 148.3 | 167.6 |
| 4 users, aggregate | 325.2 | 282.0 | 289.2 |
| 8 users, aggregate | 559.7 | 482.4 | 449.2 |

Cold prefill was 2,514 tok/s (median of 9 runs; three public-domain texts of about 24k, 34k and 38k tokens; no cache hits).

- Data: [`data/bench/v2-decode.log`](data/bench/v2-decode.log), [`data/bench/v2-prefill.log`](data/bench/v2-prefill.log), `data/bench/mia-json/engramtmpfs-*.json`.
- For reference, upstream publishes these figures with DSpark at 180 W: 356/258/184 for 1 user, 868/544/441 for 8 users, and 2,416 tok/s prefill.
- Stability: we ran 2 waves of 8 concurrent requests, each with a unique 100k-token prompt. All 16 completed with no errors ([`data/logs/04-stress-0.97-and-copytest.txt`](data/logs/04-stress-0.97-and-copytest.txt)).

## Repository layout

```
recipe/
  serve.sh.patch, start.sh.patch       patches against GNmac-lab @ 2ad5741ab1 (CDI GPUs, optional drafter,
                                       Engram dir mount, graft-aware expert-file regex)
  tools/fetch_official_nonexpert.py.patch   pinned revision for the official tensors
  tools/download-pinned.sh           downloads with pinned revisions (2.9 pack, 3.5 shards 20-24, official)
  tools/make_hybrid_graft.py           builds the hybrid and grafts layers (profiles k3, k4x4); checks K in headers
  engram-ram.sh                        copies the Engram tables into a tmpfs before start
  env.example                          production environment
tests/
  copytest-dsv41.py, copytest-solo.py, copytest-determ.py   exact-copy and determinism tests
  stress-dsv41.py                      N concurrent long-context requests, peak VRAM per card
  nll_eval.py                          paired NLL / perplexity evaluation (build, run, compare, selfcheck)
data/
  nll/       per-window NLL runs (k29 = original 2.9 pack, k3 = graft, k4 = 4-bit test, k3bis = restart, sc-* = selfcheck)
  bench/     decode and prefill logs, MiaAI-protocol JSON; INVALID-* = the run without Engram tables
  logs/      experiment logs, numbered in order
docs/        setup, research, methodology, results, comparison
```

## Quick start

Full instructions are in [`docs/setup.md`](docs/setup.md). In short:

1. Clone the upstream recipe and check out `2ad5741ab1`.
2. Apply `recipe/*.patch` with `patch -p1`.
3. Copy `recipe/tools/*`, `recipe/engram-ram.sh` and `recipe/env.example` (as `.env`) into it.
4. Run `tools/download-pinned.sh`. It downloads about 428 GiB.
5. Run `tools/make_hybrid_graft.py <MODELS_DIR> DeepSeek-V4.1-Flash-1017-hybrid k3`.
6. Run `./start.sh`. The first start compiles kernels and takes about 10 minutes.
7. Run the smoke test, then `tests/copytest-dsv41.py`.

**Paths to check.** `engram-ram.sh` takes `MODELS_DIR` and `ENGRAM_HOST_DIR` from `.env`, but the patched `start.sh` calls it only when `ENGRAM_HOST_DIR` is under `/opt/engram-ram/`. The test scripts expect the server on `127.0.0.1`, and `tests/stress-dsv41.py` reads the Gutenberg texts cached by upstream `bench/prefill.py` (run it from the recipe root, or pass `--corpus-dir`).

**Memory.** Plan for about 190 GiB of RAM for the tables plus the engine's own host memory. We raised our container's limit to 460 GiB.

## Credits

- [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) @ `2ad5741ab1`: the recipe, image, kernels and benchmarks this work builds on.
- [LibertAIDAI](https://huggingface.co/LibertAIDAI): the EXL3 packs, [2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) @ `0be63c33` and [3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) @ `04a492fc`.
- [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10`: the model.
- Curtis White: the hybrid checkpoint recipe and `quantization_config.json`, from [CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX).
- [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe): the alternative recipe we compared against.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks): the decode benchmark protocol (the upstream `bench/mia.py` downloads and runs it; it is not included here).
- [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41): per-layer sensitivity data for the 1017 checkpoint.
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format and kernels.
- [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3): the EXL3 plugin for vLLM.

## Licensing

- **This repository** (patches, scripts, tests, data, docs) is MIT; see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md), which lists the files derived from upstream.
- **The upstream recipe** is MIT, except `patches/vllm-exl3/`, which is AGPL-3.0-only because it modifies vcruz305/vllm-exl3. We do not redistribute those files. The engine image you build from the upstream recipe contains AGPL-3.0 code. If you serve it to other people over a network, AGPL-3.0 section 13 applies; see the upstream `NOTICE.md`.
- **Weights** are not included. The DeepSeek weights and the LibertAI packs are MIT, under their owners' terms. Curtis White's `quantization_config.json` is fetched from his repository and is not copied here; the upstream notice says it states no licence.
- The MiaAI-Lab benchmark script (AGPL-3.0) is not included. Our `data/bench/` files are its outputs.

## Disclaimer

This is a homelab project with a single machine and a single set of measurements. We are not affiliated with DeepSeek, NVIDIA, GNmac-lab, LibertAI, Morrowmake or any other project named here. CMP 170HX cards need a modified driver to work as described, so check the terms that apply to you. The numbers are what we measured on our hardware, with the data files cited. They are not guarantees. Perplexity changes are relative between our configurations. We did not measure the distance to the unquantized model ourselves.
