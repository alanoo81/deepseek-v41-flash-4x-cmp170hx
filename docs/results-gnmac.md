# Results: the previous configuration (GNmac engine)

> **Previous configuration.** This page documents the GNmac-lab 1.2.0 configuration we served until 2026-10-31. What we serve now (Morrowmake engine, the same graft, a lossless compact KV cache) is in [setup.md](setup.md) and [results.md](results.md).

This page lists every number we measured, with the file it comes from. Paths are relative to the repository root.

- **Measured** means it is in a data file, or in our engineering notes when the raw output was not kept (marked "notes").
- **Derived** means simple arithmetic on measured numbers.
- **Inference** is our interpretation, labelled as such.

All measurements were taken in October 2026 on one machine:

- 4x CMP 170HX, 64 GB each, PCIe Gen2 x16, P2P on
- 200 W per card, 170tune +200 MHz at 1500 MHz
- EPYC 7F52, 512 GB DDR4-3200 in 8 channels

The software is the GNmac-lab recipe at `2ad5741ab1` (image built locally) with this repository's patches. It ran TP4, at a context of 262,144 tokens, with 8 sequences, `fp8_ds_mla` KV and prefix caching on. The served model name is `deepseek-v4.1-flash`.

## Contents

1. [Configurations](#1-configurations)
2. [The invalid first run (no Engram tables)](#2-the-invalid-first-run-no-engram-tables)
3. [KV pool and per-card memory](#3-kv-pool-and-per-card-memory)
4. [Decode throughput](#4-decode-throughput)
5. [Cold prefill](#5-cold-prefill)
6. [Stress runs and peak VRAM](#6-stress-runs-and-peak-vram)
7. [Copy tests and determinism](#7-copy-tests-and-determinism)
8. [NLL / perplexity](#8-nll--perplexity)
9. [How the measurements guard against known biases](#9-how-the-measurements-guard-against-known-biases)
10. [What we did not measure](#10-what-we-did-not-measure)

## 1. Configurations

| Label | Routed experts | Drafter | Engram | `gpu_memory_utilization` | Notes |
|---|---|---|---|---|---|
| `no-engram` | `k3` graft | none | **host mode, tables never loaded** | 0.95 | **INVALID** for quality and as a reference. Files named `INVALID-*`, `01-*`, and `base-18-22q3-nospec-engramRAM-*` (that label is wrong: the tables were empty) |
| `k3` | 2.9 pack, layers 18-22 grafted at 3 bits from the 3.5 pack (all 40 layers K3) | none | `ssd` mode from a tmpfs in RAM | 0.95 / 0.97 / 0.98 | **Production at 0.98.** Benchmarks at 0.95 (`v2-*`, `engramtmpfs-*`) |
| `k29` | 2.9 pack as published (layers 18-22 at K2) | none | tmpfs | 0.97 | A/B baseline for the graft |
| `k4x4` | `k3` plus layers 0, 1, 3, 6 at K4 (from the 3.5 pack) | none | tmpfs | 0.97 | 4-bit experiment |
| `dspark` | `k3` | DSpark, 5 draft tokens | tmpfs | 0.97 | Speed/KV trade-off test |

The utilisation changes only the KV pool size. NLL results at 0.97 and 0.98 are bit-identical ([section 8](#8-nll--perplexity)).

## 2. The invalid first run (no Engram tables)

Our first deployment left `ENGRAM_STORAGE` empty. We expected the engine's "host" mode to keep the Engram tables in RAM. On this stack it does not load them:

- It reserves 2 x 23.6 GiB of pinned memory per rank (188.8 GiB over 4 ranks) and never fills it.
- The hybrid index has no `engram.embed.*` entries.
- vLLM skips its missing-weights check when a quantization method is set.

We found this by reading the code after the copy test failed. A read-only diagnosis pass located it.

The likely effect is that Engram layers 1 and 14 act as identity functions. That is an inference: we did not dump the pinned memory to check that it held zeros.

The mistake was ours. Upstream documents and tests the `ssd` mode only.

| Check | `no-engram` | `k3` (fixed) | Source |
|---|---|---|---|
| Server starts, smoke test | yes | yes | `data/logs/01-first-start-engram-NOT-loaded.txt`; `data/logs/08-ceiling-0.98-smoke-stress-nll.txt` |
| Log line `Built engram token map (129280 -> 99092 ids) for layers (1, 14)` | absent | present | `data/logs/02-restart-engram-tmpfs-ssd.txt` |
| KV pool at 0.95 | 1,812,663 | 1,817,233 | log 01; notes |
| Copy test, 20 lists, pass 1 | **15/20** | 20/20 | notes (output of the failing run not kept) |
| Same request repeated at temperature 0 | outputs differed | byte-identical | notes; `data/logs/03-copytest-100-and-determinism.txt` |
| Decode, 1 user, structured | 105.5 tok/s | 105.4 tok/s | `data/bench/INVALID-no-engram-decode.log`; `data/bench/v2-decode.log` |
| Cold prefill median | 2,421 tok/s | 2,514 tok/s | `data/bench/INVALID-no-engram-prefill.log`; `data/bench/v2-prefill.log` |

What the copy failures looked like (notes):

- Two of the five failures were massive: the `_` separators between word and number disappeared, as in `fgpyknilkc0151136896829`.
- The other three had one or two wrong lines.
- The same five tasks failed at concurrency 8 and at concurrency 1.
- In solo replays, one task passed once and failed once, and another produced `..._soil_f56ba531` in one run and `..._0f56ba531` in another.

**Lesson:** the server, the smoke test and the speed benchmarks all looked healthy without the tables. Only a fidelity test caught the problem. Speed benchmarks cannot validate a deployment.

## 3. KV pool and per-card memory

The pool size is from the `GPU KV cache size` log line. The concurrency column is the number of full 262,144-token sequences that fit.

| Config | Util | KV pool, tokens | Full 262k sequences | KV GiB per card | Startup memory per card | Source |
|---|---|---:|---:|---:|---|---|
| `no-engram` (INVALID) | 0.95 | 1,812,663 | 6.91 | 3.84 | 54.04-54.13 GiB weights + non-torch, 2.26 GiB peak activations | `data/logs/01-first-start-engram-NOT-loaded.txt` |
| `k3` | 0.95 | 1,817,233 | 6.93 (derived) | | | notes |
| `k3` | 0.97 | 2,417,392 | 9.22 | 5.13 | 54.04 GiB weights, 2.29 GiB peak activations, 0.9 GiB CUDA graphs | notes |
| **`k3` (production)** | **0.98** | **2,717,420** | **10.37** | **5.76** | | `data/logs/08-ceiling-0.98-smoke-stress-nll.txt` |
| `k29` | 0.97 | 3,347,338 | 12.77 | | model loading 51.3 GiB | `data/logs/05-ab-2.9orig-vs-graft.txt` |
| `k4x4` | 0.97 | 1,670,368 | 6.37 (derived) | | model loading 54.82 GiB | notes |
| `dspark` | 0.97 | 1,113,781 | 4.25 | 2.35 | model loading 55.54 GiB | `data/logs/06-dspark-kv-bench-copytest.txt` |

At startup each card had 62.82-62.86 GiB free out of 63.39 GiB visible. The 0.95 budget is 60.22 GiB per card (log 01).

The "weights + non-torch" and "model loading" figures come from different vLLM log lines and are not directly comparable. The KV pool is the clean comparison.

**Derived: what each change costs in VRAM.** We measured about 472,000 KV tokens per GiB per card (1,812,663 / 3.84 and 2,717,420 / 5.76 both give 472k). Converting the pool differences at 0.97:

| Change vs `k3` at 0.97 | KV tokens lost (or gained) | GiB per card (derived) | Predicted before the run |
|---|---:|---:|---|
| `k29` (layers 18-22 back to 2 bits) | +929,946 | -1.97 | -1.98 (read-only VRAM budget, from safetensors headers) |
| `k4x4` (4 layers K3 to K4) | -747,024 | +1.58 | +1.58 (4 x 0.395) |
| `dspark` (drafter loaded) | -1,303,611 | +2.76 | about 2.18 (budget research, from upstream's accounting). The KV-memory log lines give 2.78 (5.13 - 2.35) |

The `k29` and `k4x4` predictions were computed from the safetensors headers before those runs, and matched to within 0.02 GiB per card. The drafter estimate was about 0.6 GiB too low.

**Published references, not measured by us and not like-for-like:**

| | KV pool at 262k | Source |
|---|---:|---|
| GNmac 1.2.0 (2.9 pack, DSpark on, 0.95) | 1,395,731 | upstream `docs/results.md` |
| Morrowmake 1.0.0 (2.9 pack, `bfloat16` KV, DSpark on, 0.95) | 1,247,406 | Morrowmake `docs/results.md` |

## 4. Decode throughput

**Protocol.** MiaAI-Lab's `bench_decode.py`, run unmodified by upstream `bench/mia.py`:

- three prompts: structured (count 1-200), code (a `clamp_range` function), prose (how a hash map works);
- temperature 0, thinking off, 400-token cap;
- **1 user**: median of 5 streaming decode rates, measured *after* the first token;
- **2/4/8 users**: median of 3 runs of aggregate throughput (completion tokens over the batch's wall time, prefill included). All requests of a batch are sent at once.

There was one warm-up request before each point, not counted. Units: tok/s.

### 4.1 Production weights, no drafter (`k3`, 0.95)

| tok/s | Structured | Code | Prose |
|---|---:|---:|---:|
| 1 user | 105.4 (105.0-105.4) | 105.0 (104.4-105.7) | 104.8 (104.6-105.3) |
| 2 users, aggregate | 174.3 | 148.3 | 167.6 |
| 4 users, aggregate | 325.2 | 282.0 | 289.2 |
| 8 users, aggregate | 559.7 (554.2-561.8) | 482.4 (467.3-486.1) | 449.2 (448.2-450.4) |

Sources: `data/bench/v2-decode.log`, `data/bench/mia-json/engramtmpfs-18-22q3-nospec-*.json`. The ranges are the minimum and maximum of the runs behind each median. No NaN in any run.

- The code prompt's answer is about 95 tokens long and ends naturally (`finish_reason: stop`), so its multi-user rates include a large share of prefill time.
- The structured and prose answers run to the 400-token cap.

### 4.2 Without Engram tables (INVALID, kept for the record)

| tok/s | Structured | Code | Prose |
|---|---:|---:|---:|
| 1 user | 105.5 | 106.1 | 105.4 |
| 2 users | 176.4 | 134.0 | 166.1 |
| 4 users | 331.1 | 211.1 | 282.5 |
| 8 users | 560.4 | 375.9 | 445.7 |

Sources: `data/bench/INVALID-no-engram-decode.log`, `data/bench/mia-json/base-18-22q3-nospec-engramRAM-*.json`. The `engramRAM` in those file names is wrong: the tables were empty.

- **Measured:** structured and prose, which run to a fixed 400 tokens, are within about 1 % of the fixed configuration. Reading Engram rows from a tmpfs costs nothing measurable in decode.
- **The code rows are not comparable.** Without Engram the model wrote a different, longer answer: a median of 121 completion tokens at 1 user, against 95 with the tables. The 376 vs 482 tok/s difference at 8 users is mostly a different amount of work per request, not a speed change.

### 4.3 DSpark on (`dspark`, 0.97)

| tok/s | Structured | Code | Prose |
|---|---:|---:|---:|
| 1 user | 369.8 (x3.51) | 267.3 (x2.55) | 200.1 (x1.91) |
| 8 users, aggregate | 852.9 (+52 %) | 450.3 (-7 %) | 437.8 (-2.5 %) |
| Acceptance ratio, 1 user / 8 users | 0.994 / 0.994 | 0.705 / 0.710 | 0.433 / 0.528 |

Ratios are against [4.1](#41-production-weights-no-drafter-k3-095). Sources: `data/bench/dspark-decode.log`, `data/bench/mia-json/dspark-greffe-*.json`, `data/logs/06-dspark-kv-bench-copytest.txt`.

- The 8-user structured runs spread widely: 544.6, 852.9 and 954.5.
- The structured prompt (counting) is almost perfectly predictable, which is why its acceptance is 99.4 %. That figure does not represent typical text.
- MTP is not available for V4.1 in this vLLM build: it refuses the method, and only DSpark steps exist under `mtp.*` (notes).

**Inference:** DSpark is lossless by construction (greedy drafts verified by standard rejection sampling, from reading the code). We turned it off because it shrinks the KV pool below our 1.2M-token floor, and because at 8 concurrent users it gains nothing on code and prose.

### 4.4 Upstream published (GNmac 1.2.0)

These are upstream's figures, not ours: DSpark on with 5 drafts and variable verify width, 180 W, Engram on NVMe, the 2.9 pack as published, utilisation 0.95.

| tok/s | Structured | Code | Prose |
|---|---:|---:|---:|
| 1 user | 355.7 | 258.4 | 183.6 |
| 2 users | 433.1 | 243.8 | 238.8 |
| 4 users | 626.4 | 387.8 | 348.7 |
| 8 users | 867.6 | 544.0 | 440.5 |

Source: upstream `docs/results.md` at `2ad5741ab1`. Our `dspark` run differs in power (200 W), weights (the graft) and Engram placement, so set side by side the two are only a sanity check:

- 1 user: 369.8 / 267.3 / 200.1 (ours) vs 355.7 / 258.4 / 183.6 (upstream);
- 8 users: 852.9 / 450.3 / 437.8 vs 867.6 / 544.0 / 440.5.

Morrowmake's published decode figures use a different definition of the 1-user rate. They are discussed in [comparison-morrowmake.md](comparison-morrowmake.md).

## 5. Cold prefill

**Protocol.** Upstream `bench/prefill.py`:

- three public-domain Gutenberg texts cut to about 24k, 34k and 38k tokens with the server's tokenizer;
- 3 runs each, every run starting at a random line so that the prefix cache cannot hit;
- one request at a time, `max_tokens` 1;
- rate = prompt tokens / time to first token, median of the 9 runs.

| Text (tokens) | `k3`, run 1 / 2 / 3 (tok/s) | `no-engram` (INVALID), run 1 / 2 / 3 |
|---|---|---|
| pride-and-prejudice (~24,027) | 2,550 / 2,568 / 2,569 | 2,470 / 2,483 / 2,460 |
| moby-dick (~34,026) | 2,515 / 2,511 / 2,514 | 2,410 / 2,421 / 2,424 |
| war-and-peace (~38,026) | 2,481 / 2,505 / 2,489 | 2,376 / 2,396 / 2,411 |
| **Median of 9** | **2,514** | 2,421 |

Sources: `data/bench/v2-prefill.log`, `data/bench/INVALID-no-engram-prefill.log`. Every run reports `cached=0`.

For reference:

- Upstream GNmac 1.2.0 publishes 2,416 tok/s with the same script (180 W, DSpark on, Engram on NVMe).
- The stress runs below read about 800,000 prompt tokens per wave in about 390 s, roughly 2,050 tok/s aggregate with 8 concurrent 100k-token prompts and generation included (derived).

**Inference:** with the tables in the tmpfs, the first run on each text is not slower than the later ones. Upstream reports about 10 % slower first runs with Engram on NVMe while the page cache fills.

## 6. Stress runs and peak VRAM

`tests/stress-dsv41.py`:

- 8 concurrent requests per wave;
- each prompt is a unique 100,000-token Gutenberg excerpt (random start plus nonce, so no prefix-cache hits), with 512 output tokens at temperature 0;
- the script samples `nvidia-smi` for peak memory per card.

Usable memory is 64,911 MiB per card.

| Config | Waves | Prompt tokens per wave | Generated | Wall time | Result | Peak MiB per card (GPU 0 / 1 / 2 / 3) | Margin to 64,911 | Source |
|---|---|---:|---:|---:|---|---|---:|---|
| `k3` 0.97 | 2 | 800,134 / 800,136 | 1,841 / 1,781 | 392 / 389 s | 16/16 OK, server alive | 62,900 / 62,896 / 62,926 / 62,876 | about 1.9 GiB | `data/logs/04-stress-0.97-and-copytest.txt` |
| `dspark` 0.97 | 2 | 800,134 / 800,136 | 1,836 / 1,631 | 391 / 390 s | 16/16 OK | 63,652 / 63,808 / 63,798 / 63,768 | about 1.1 GiB | `data/logs/07-dspark-stress-vram.txt` |
| `k3` 0.98 | 1 | 800,134 | 1,937 | 388 s | 8/8 OK | 63,580 / 63,574 / 63,566 / 63,556 | about 1.3 GiB | `data/logs/08-ceiling-0.98-smoke-stress-nll.txt` |
| `k3` 0.98, NLL with `prompt_logprobs` (1,024-token windows) | | | | | no OOM | 64,540 / 64,534 / 64,546 / 64,676 | **about 0.23 GiB** | `data/logs/08-ceiling-0.98-smoke-stress-nll.txt` |
| `k3` 0.98, `prompt_logprobs` on a 2,001-token prompt (self-check canary, 2026-10-30) | | | | | **CUDA OOM on GPU 0, engine dead** (about 1 GiB requested) | | none | [comparison-morrowmake.md, 0.7](comparison-morrowmake.md#a07-side-finding-prompt_logprobs-can-crash-our-production-configuration) |

- Under DSpark, the peak during the decode benchmarks was the same as during the stress run (63.8 GB). The peak is reached quickly and depends little on the load (notes).
- `prompt_logprobs` allocates a logits buffer that the startup memory profile does not count. That is why the NLL run comes so close to the limit at 0.98, and why a 2,001-token request killed the engine. Use 0.97 or windows of 1,024 tokens or fewer for scoring. The server has no restart policy: after such a crash it stays down.

## 7. Copy tests and determinism

`tests/copytest-dsv41.py`: copy a list of 40 random identifiers (`word_word_5digits_8hex`) exactly, at temperature 0 with reasoning off. Three passes:

1. Cold.
2. Same prompts again, with prefix-cache hits.
3. A second conversation turn on a cached prefix, copying a new list.

A list counts as exact only if every line matches.

| Config | Lists | Pass 1 | Pass 2 (cached prompt tokens) | Pass 3 (cached prompt tokens) | Wall time | Source |
|---|---:|---|---|---|---|---|
| `no-engram` 0.95 | 20 | **15/20** | n/a (`cached_tokens` not exposed yet) | 20/20 | | notes |
| `k3` 0.95 | 20 | 20/20 | 20/20 (92 %) | 20/20 (31 %) | | notes |
| `k3` 0.95 | 100 (seed 777) | 100/100, 0 wrong lines | 100/100 (64,000 / 69,940 = 92 %) | 100/100 (64,000 / 205,580 = 31 %) | 185 / 161 / 207 s | `data/logs/03-copytest-100-and-determinism.txt` |
| `k3` 0.97, after the stress run | 20 | 20/20 | 20/20 (12,800 / 13,921 = 92 %) | 20/20 (12,800 / 40,985 = 31 %) | 38 / 34 / 44 s | `data/logs/04-stress-0.97-and-copytest.txt` |
| End of the `k29` vs `k3` A/B session | 20 | 20/20 | 20/20 (92 %) | 20/20 (31 %) | 39 / 34 / 44 s | `data/logs/05-ab-2.9orig-vs-graft.txt` |
| `k4x4` 0.97 | 20 | 20/20 | | | | notes |
| `dspark` 0.97 | 20 | 20/20 | 20/20 (92 %) | 20/20 (31 %) | 33 / 25 / 37 s | `data/logs/06-dspark-kv-bench-copytest.txt` |

**Determinism** (`tests/copytest-determ.py`, `k3`): 5 tasks x 5 repetitions sent one at a time. Every task came back exact 5/5, with 1 distinct output out of 5, that is byte-identical (`data/logs/03-copytest-100-and-determinism.txt`).

What the copy test can and cannot tell:

- It caught the missing Engram tables (15/20).
- It shows no prefix-cache or Engram-history corruption on this stack. That failure mode is reported for an unmerged llama.cpp branch and in an open upstream issue.
- With the tables loaded it is **saturated**: `k29` and `k3` both score 20/20, although NLL separates them clearly. Use it as a gate, not as a quality ranking.

## 8. NLL / perplexity

**Method.** `tests/nll_eval.py`, paired, on frozen token windows:

- **Corpora:**
  - French: 8 Gutenberg books;
  - English: 8 Gutenberg books;
  - code: the upstream recipe's source.
- **Window sets:**
  - long: 64 windows of 1,024 tokens per corpus, which run the prefill MoE kernels;
  - short: 400 windows of 40 tokens per corpus, which run the decode-shaped MoE kernels.
- Scoring is by teacher forcing through `/v1/completions` with `prompt_logprobs`. Requests that ask for prompt logprobs skip the prefix cache, and each request carries a cache salt.
- `d` = NLL_B - NLL_A per window, in nats per token; negative means B predicts the text better.
- The CI is a bootstrap 95 % interval on the perplexity change.
- **Design:** A-B-A across server restarts, so that the cross-restart noise is measured, not assumed.

Reproduce any table: `python3 tests/nll_eval.py compare data/nll/<A>.json data/nll/<B>.json`.

### 8.1 Null pairs (noise floor)

| Pair | Windows | Result |
|---|---|---|
| `k3` vs `k3bis` (same weights, server restarted after the `k4x4` run), long | 192 | **bit-identical**: dPPL 0.000 %, every token logprob identical |
| `k3-short` vs `k3bis-short` | 1,200 | bit-identical |
| `k3` (0.97) vs `k3-098` (0.98, another restart) | 192 | bit-identical |
| Same request twice within one run (`repeat` field) | 6 per run | `k3`, `k3bis`, `k29`: 0 difference. `k4` long: max \|d mean NLL\| 0.00109 |

Files: `data/nll/k3.json`, `k3bis.json`, `k3-098.json`, `k3-short.json`, `k3bis-short.json`.

So on this stack the prefill scoring path is bit-reproducible across restarts with the same weights. The noise floor of the comparisons below is zero, and any difference is the weights.

The `k4x4` configuration was the only one not bit-identical within a run, a small drift on repeats. **Inference:** this comes from the 4-bit cooperative kernel path.

### 8.2 Layers 18-22 at 2 bits (`k29`, A) vs 3 bits (`k3`, B)

Long windows (1,024 tokens):

| Set | n | PPL A | PPL B | dPPL % | 95 % CI | B better / worse | Sign p | Wilcoxon p |
|---|---:|---:|---:|---:|---|---|---:|---:|
| French | 64 | 2.2231 | 2.1242 | **-4.451** | [-4.974, -3.934] | 64 / 0 | 1.1e-19 | 3.6e-12 |
| English | 64 | 1.3284 | 1.3191 | -0.701 | [-0.946, -0.477] | 50 / 14 | 7.1e-06 | 2.8e-07 |
| Code | 64 | 2.9901 | 2.9846 | -0.184 | [-0.331, -0.039] | 36 / 28 | 0.38 | 0.058 |
| **All** | 192 | 2.0669 | 2.0298 | **-1.797** | [-1.996, -1.600] | 150 / 42 | 1.9e-15 | 2.7e-21 |

Short windows (40 tokens):

| Set | n | PPL A | PPL B | dPPL % | 95 % CI | B better / worse | Wilcoxon p |
|---|---:|---:|---:|---:|---|---|---:|
| French | 400 | 12.6250 | 12.5134 | -0.884 | [-1.607, -0.164] | 213 / 187 | 0.014 |
| English | 400 | 7.0539 | 6.9919 | -0.880 | [-1.479, -0.284] | 218 / 182 | 0.0099 |
| Code | 400 | 23.9517 | 23.9634 | +0.049 | [-0.426, +0.520] | 184 / 216 | 0.44 |
| **All** | 1,200 | 12.8726 | 12.7989 | **-0.573** | [-0.919, -0.221] | 615 / 585 | 0.0070 |

Files: `data/nll/k29.json`, `data/nll/k29-short.json` vs `data/nll/k3.json`, `data/nll/k3-short.json`; printed output in `data/logs/05-ab-2.9orig-vs-graft.txt`.

- **Measured:** the graft is a large and significant improvement. Every one of the 64 French windows got better, and mean top-1 accuracy on French rose by 1.3 percentage points.
- **Inference:** LibertAI reports the whole 2.9 pack at +0.4 % perplexity on prose versus the original. That figure may be dominated by English; French loses far more to the 2-bit layers.
- Expected size before the run: sudoingX's per-layer sensitivity model (noise-injection alpha on the 1017 checkpoint) scored this graft about 2.3 times as valuable as moving layers 0, 1, 6 and 5 to 4 bits, a set close to the one tested below. That is a prediction from the research pass, not a measurement.

### 8.3 Layers 0, 1, 3, 6 at 4 bits (`k4x4`)

A = `k3`, B = `k4`. The `k3bis` vs `k4` comparison gives identical figures because `k3` and `k3bis` are bit-identical.

| Windows | Set | dPPL % | 95 % CI | B better / worse |
|---|---|---:|---|---|
| Long | French | +0.040 | [-0.145, +0.230] | 31 / 33 |
| Long | English | +0.160 | [-0.025, +0.374] | 35 / 29 |
| Long | Code | +0.040 | [-0.065, +0.145] | 31 / 33 |
| Long | **All** | **+0.080** | [-0.014, +0.183] | 97 / 95 |
| Short | French | +0.376 | [-0.097, +0.858] | 189 / 211 |
| Short | English | -0.376 | [-0.914, +0.143] | 204 / 196 |
| Short | Code | -0.060 | [-0.471, +0.356] | 197 / 203 |
| Short | **All** | **-0.021** | [-0.297, +0.254] | 590 / 610 |

Minimum detectable effect at 80 % power on the combined set (from `compare`): 0.14 % on long windows and 0.40 % on short windows.

Files: `data/nll/k4.json`, `data/nll/k4-short.json`.

- **Measured:** no detectable change in either direction. The token-level logprobs did change (mean |d| 0.03-0.09 nats on long windows), so the 4-bit weights were in use.
- **Decision:** back to all-K3. The 4-bit layers cost about 1.58 GiB of KV per card for no measurable gain.
- Layer choice: layers 0 and 1 rank first in every sensitivity source we found (sudoingX's alpha, five published per-expert allocation plans). Layer 6 ranks third by alpha. Layer 3 was preferred over 5 (a tie in alpha) because it balances the TP rotation and saves VRAM on the worst card.

### 8.4 Decode-path self-check and causality canary

`nll_eval.py selfcheck`:

- greedy generation of 64 tokens after 4 prompts of 1,024 tokens, then teacher-forced rescoring of the same tokens;
- plus a random 1,000-token block repeated once. The first copy should stay near 13 nats per token; the second should be near 0.

| | `k3` | `k4x4` |
|---|---:|---:|
| Tokens compared | 768 | 768 |
| Mean \|decode logprob - prefill logprob\| | 0.0258 | 0.0211 |
| p99 / max | 0.393 / 1.130 | 0.279 / 0.797 |
| Prefill rank of the decoded token not 1 | 2.2 % | 2.6 % |
| Canary, first copy / second copy (nats per token) | 13.17 / 0.062 | 13.20 / 0.060 |

Files: `data/nll/sc-k3.json`, `data/nll/sc-k4.json`. Both configurations pass. The decode and prefill paths agree as closely as expected for different kernels and batch shapes, and attention is causal.

### 8.5 Caveats on the NLL numbers

- These are **relative** perplexities between our configurations. We never measured the distance to the unquantized model.
- Gutenberg classics are probably in the training data, and the code set is partly derived from vLLM and exllamav3. Memorized text can understate quantization damage.
- The code set and its windows are frozen by SHA-256 (`corpus_sha256` in each run file). Rebuilt corpora are comparable with ours only if the hashes match.
- An earlier version of the sign test overflowed (`math.fsum` at n=400). It now uses exact integers, and the p-values above come from the fixed `compare`.

## 9. How the measurements guard against known biases

We have been caught before by benchmarks that gave plausible but wrong numbers. These are the checks applied here:

| Bias | Guard in these measurements |
|---|---|
| Prefix-cache hits inflate prefill or hide errors | Prefill runs start at random lines and log `cached=0`. Stress prompts carry a nonce. NLL requests use a cache salt and `prompt_logprobs` (which skips the cache). The copy test reports cached tokens per pass, so cache use is shown, not assumed |
| First-run warm-up | One warm-up request per decode point, excluded. 3 runs per prefill text, 5 or 3 runs per decode point, medians reported with min-max where kept |
| Forcing output length (`ignore_eos`) measures degenerate text | Not used. Every decode run ended naturally (`finish_reason: stop`) or at the 400-token cap (`length`, the prose prompt); none generated past an end-of-sequence token |
| Counting stream chunks instead of tokens (undercounts speculative decoding) | Token counts come from the server's `usage` field, recorded per request in the JSON files |
| Acceptance ratios on degenerate or trivially predictable text | We report acceptance per prompt type. The 99.4 % figure is the counting prompt and is not representative |
| Burst arrivals | The MiaAI protocol sends all requests of a batch at once. Our multi-user figures are burst figures, as are upstream's. Staggered agent traffic may behave differently |
| Comparing different methods | All our tables come from the same scripts. Upstream's and Morrowmake's published figures are marked as such and not mixed into our tables |
| A speed benchmark taken as a health check | See [section 2](#2-the-invalid-first-run-no-engram-tables): speed was normal without the tables |

## 10. What we did not measure

- KL or perplexity against the original unquantized model, or against DeepSeek's reference implementation.
- Other task accuracy than HumanEval and GSM8K (measured on 2026-10-30, [comparison-morrowmake.md, 0.4](comparison-morrowmake.md#a04-humaneval-and-gsm8k): 157-158/164 and 1,277-1,280/1,319 over two runs).
- Engram read from NVMe (`ssd` mode without the tmpfs) on our machine.
- Context lengths above 262,144 tokens, the `pp4` layout, and needle-in-a-haystack retrieval.
- Long agentic sessions with tool calls under load. The smoke test checks a single tool call.
- Thermal behaviour under sustained load.
- Morrowmake's speed and long-context quality on the same machine: measured on 2026-10-30 and 31, see [comparison-morrowmake.md](comparison-morrowmake.md) and [results.md](results.md).
