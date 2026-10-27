# Notices

This repository's own scripts, patches, tests, measurement data and documents are under the MIT licence: see [LICENSE](LICENSE). Copyright 2026 alanoo81.

This repository contains **no model weights and no engine code**. It has no AGPL-licensed files.

## Files derived from upstream

These files are derived from [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) at commit `2ad5741ab1` (release 1.2.0). The upstream files they modify are MIT-licensed, Copyright (c) 2026 GNmac-lab. Upstream's MIT notice applies to the upstream portions; our changes are MIT as well.

| File here | Derived from (upstream, MIT) | Nature of the change |
|---|---|---|
| `recipe/serve.sh.patch` | `serve.sh` | Unified diff: GPUs through CDI, optional DSpark drafter, configurable Engram storage and directory, file regex for the grafted shards, `VLLM_EXL3_FAT_THRESHOLD` passthrough |
| `recipe/start.sh.patch` | `start.sh` | Unified diff: run `engram-ram.sh` before `serve.sh` |
| `recipe/tools/fetch_official_nonexpert.py.patch` | `tools/fetch_official_nonexpert.py` | Unified diff: pinned revision (`OFF_REV`) |
| `recipe/tools/download-pinned.sh` | `tools/download.sh` | Modified copy (attribution in its header): pinned revisions, extra 3.5 bpw shards, no hybrid build step |

The three patches contain context lines copied from the upstream files.

The other files are ours. `recipe/tools/make_hybrid_graft.py` calls upstream `tools/make_hybrid.py` but does not include it (its header says so). The `data/` files are our own measurement outputs.

## Upstream recipe and its AGPL part

[GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) is MIT-licensed, **except** `patches/vllm-exl3/`. Those files modify [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3), which is AGPL-3.0-only, so they are AGPL-3.0-only too.

We do not copy or modify any of those files. However, the engine image you build with upstream `docker/build.sh`, which this repository's instructions use unchanged, **contains AGPL-3.0-only code** (vllm-exl3 with upstream's changes).

If you let other people use a server built from that image over a network, AGPL-3.0 section 13 requires you to offer them the corresponding source. Per the upstream `NOTICE.md`, that means the upstream repository plus the upstream commits it pins. Read the upstream `NOTICE.md` for the full list of components in the image and their licences. It includes vLLM, FlashInfer, Deepinfer-SM80, zebgop-ops/dsv41reap-pp, kaka86mm/dsv41-flash-pp4-170hx, CurtisAccelerate's TC-MoE kernels and parts of Morrowmake/vllm-cmp170hx, mostly Apache-2.0.

## Models and data used, not redistributed

| Item | Revision | Licence | How it is used |
|---|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `1bae0e10` | MIT | Non-expert tensors (fetched by byte range), Engram tables. Downloaded by the user |
| [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | `0be63c33` | MIT | Routed experts, layers 0-39 (except 18-22 in our configuration). The quantization is LibertAI's |
| [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) | `04a492fc` | MIT | Routed experts of layers 18-22 at 3 bits (shards `00020`-`00024-of-00044`); layers 0, 1, 3, 6 at 4 bits in the experimental `k4x4` profile |
| `quantization_config.json` by Curtis White, from [CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX) | `b8b9e99` | No licence stated (per the upstream notice) | The hybrid checkpoint layout is Curtis White's. The file is fetched from his repository at a pinned commit, not copied here |
| Project Gutenberg texts | n/a | Public domain in the US | Prefill benchmark, stress test and NLL corpora. Downloaded at run time. Our NLL run files store per-token scores of windows cut from them and a SHA-256 of each window's token IDs, not the text |
| GNmac-lab recipe source | `2ad5741ab1` | MIT, Copyright (c) 2026 GNmac-lab | The NLL code corpus, read from your local copy at run time. Our NLL run files store only per-token scores and window hashes, not its text |

## Tools and data credited

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) (AGPL-3.0): the decode benchmark protocol and its `tests/bench_decode.py`. Upstream `bench/mia.py` downloads it at a pinned commit and runs it unmodified. It is **not** included here. The files in `data/bench/` are its outputs on our server.
- [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41): per-layer sensitivity measurements for the 1017 checkpoint, used to choose which layers to try at 4 bits. Not included here.
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) (MIT): the EXL3 format and kernels, inside the engine image.
- [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) (AGPL-3.0-only): the EXL3 quantization plugin for vLLM, inside the engine image.
- [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe) (MIT): the alternative recipe compared in `docs/comparison-morrowmake.md`. Nothing is copied from it. Its published figures are quoted with attribution.

## Trademarks and affiliation

DeepSeek, NVIDIA, CMP and other names are trademarks of their owners. This project is not affiliated with DeepSeek, NVIDIA, GNmac-lab, LibertAI, Morrowmake, MiaAI-Lab or any other project named here.
