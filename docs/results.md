# Results: the current configuration (Morrowmake engine, 3-bit graft, compact KV cache)

This page lists what we measured from 2026-10-30 to 2026-10-31, with the file each number comes from. The
measurements of the previous configuration (GNmac engine) are in [results-gnmac.md](results-gnmac.md).

- **Measured** means it is in a data file under `data/`, or in a log quoted here.
- **Inference** is our interpretation, labelled as such.

One machine for everything: 4x CMP 170HX (64 GB, PCIe Gen2 x16, P2P on, 200 W, 170tune +200 MHz at 1500 MHz),
EPYC 7F52, 512 GB DDR4-3200 in 8 channels, NVIDIA open driver 610.43.02 (cmpunlocker fork, P2P patch).
Only one stack runs at a time: each uses all four cards.

## Contents

1. [Configurations](#1-configurations)
2. [Quality](#2-quality)
3. [KV pool](#3-kv-pool)
4. [Speed](#4-speed)
5. [Contexts beyond 262,144 tokens](#5-contexts-beyond-262144-tokens)
6. [Settings we tried and rejected](#6-settings-we-tried-and-rejected)
7. [What we did not measure](#7-what-we-did-not-measure)

## 1. Configurations

| Name in the data | Engine | Routed experts | Compressed-attention caches | Memory setting |
|---|---|---|---|---|
| `ours` | GNmac-lab 1.2.0 (`2ad5741ab1`), our previous production | LibertAI 2.9 pack, layers 18-22 grafted to 3 bits | `fp8_ds_mla` | 0.98 |
| `k29` | GNmac-lab 1.2.0 | LibertAI 2.9 pack as published (layers 18-22 at 2 bits) | `fp8_ds_mla` | 0.97 |
| `morrowmake` | Morrowmake 1.0.0 (`7f9023c`, engine `a0c57b52`) as published | LibertAI 2.9 pack as published | BF16 | 0.95 |
| `mm-k3` | Morrowmake 1.0.0 | layers 18-22 grafted to 3 bits | BF16 | 0.95 |
| **`mm-kvc15`** | **Morrowmake 1.0.0 + compact KV v1.5** | **layers 18-22 grafted to 3 bits** | **compact (lossless)** | **0.95** |
| `mm-k3k4` | as `mm-kvc15` | plus layers 36-39 at 4 bits | compact | 0.95 |
| `mm-kvc-faithful` | as `mm-kvc15` with compact v1, release shortcuts off | as `mm-kvc15` | compact v1 | 0.95 |

`mm-kvc15` is what we serve, with DSpark on (the release default). Compact v1 (`mm-kvc`) and v1.5 give
bit-identical log-probabilities (`data/nll/mm-kvc.json.gz`, `sc-mm-kvc.json.gz`); v1.5 has faster kernels and a
larger pool. The quality runs were made without DSpark:
it changes the speed, not the scored log-probabilities (section [2.4](#24-dspark)).

Everything else is identical across the Morrowmake rows: the official FP8 non-expert tensors, the Engram tables
in RAM, a 262,144-token context, 8 sequences. The `ours` and `k29` rows keep the GNmac setup described in
[results-gnmac.md](results-gnmac.md).

## 2. Quality

### 2.1 Method

[`tests/nll_eval.py`](../tests/nll_eval.py) scores frozen token windows with `prompt_logprobs` (teacher
forcing, the prefill path) and compares runs pair by pair, window by window (bootstrap 95 % CI of the
perplexity change, sign and Wilcoxon tests). Four sets:

| Set | Windows | Content |
|---|---|---|
| Long windows | 64 x 1,024 tokens per corpus | French, English (public-domain books from Project Gutenberg) and code |
| Short windows | 400 x 40 tokens per corpus | same corpora |
| 32k "prose" | 8 French, 8 English and 3 code windows, 32,768 tokens each | each window is one book (or one source tree) read in order |
| 32k code | 16 x 32,768 tokens | Python sources of exllamav3 and vLLM, file after file |

The 32k sets were added on 2026-10-30, after the first comparison with Morrowmake (which stopped at 1,024
tokens). They are where the engines differ.

**Noise floor.** Prompt-log-probability runs reproduce bit for bit across server restarts on both engines:
`ours` scored twice (A1 and A2, two restarts apart) is identical on the long, short and 32k-prose sets (there is
no repeat of the 32k-code set), and the Morrowmake runs below that are expected to be identical are (section
[2.3](#23-the-compact-kv-cache-is-bit-identical)). A difference between two configurations is therefore not
restart noise; whether it matters is what the CI says. The decode path is different: GNmac's self-check does not
reproduce across restarts (`sc-ours-A1` against `sc-ours-A2`: 684 of 768 values differ), Morrowmake's does
(`sc-mm-k3` against `sc-mm-kvc15`, two different starts: identical).

### 2.2 The comparisons

Perplexity change from A to B, bootstrap 95 % CI; negative means B predicts the text better. "B better" counts
windows.

| A → B | Long windows (1,024) | Short (40) | **32k prose** | **32k code** |
|---|---|---|---|---|
| **Previous production → current** (`ours` → `mm-kvc15`) | +0.14 % [+0.03, +0.26] | -0.21 % [-0.52, +0.11] | **-0.58 % [-0.74, -0.43]** (14 of 19) | **-1.64 % [-1.92, -1.37]** (16 of 16) |
| Engine only, same weights (`k29` → `morrowmake`) | +0.15 % [+0.02, +0.29] | -0.29 % [-0.59, +0.02] | -0.57 % [-0.73, -0.42] (14 of 19) | not run |
| Graft 2 → 3 bits, Morrowmake engine (`morrowmake` → `mm-k3`) | **-1.81 % [-2.01, -1.61]** (151 of 192) | -0.49 % [-0.84, -0.13] | **-2.31 % [-2.75, -1.79]** (18 of 19) | -0.06 % [-0.12, -0.00] |
| Graft 2 → 3 bits, GNmac engine (`k29` → `k3`), for reference | -1.80 % [-2.00, -1.60] | -0.57 % [-0.92, -0.22] | not run | not run |
| Compact KV cache (`mm-k3` → `mm-kvc15`) | identical | identical | identical | identical |
| Layers 36-39 at 4 bits (`mm-kvc15` → `mm-k3k4`) | -0.04 % [-0.08, +0.01] | -0.08 % [-0.22, +0.07] | -0.11 % [-0.15, -0.07] (17 of 19) | -0.11 % [-0.14, -0.09] (16 of 16) |
| Release shortcuts off (`mm-kvc15` → `mm-kvc-faithful`) | +0.04 % [-0.05, +0.13] | not run | not run | +0.00 % [-0.04, +0.05] |

Per corpus, for the rows that matter:

| A → B | Long windows: French / English / code | 32k: French / English / code |
|---|---|---|
| `ours` → `mm-kvc15` | +0.32 % / +0.11 % / -0.01 % | -0.54 % / -0.08 % / **-2.01 %** |
| `k29` → `morrowmake` | +0.32 % / +0.10 % / +0.04 % | -0.55 % / -0.05 % / **-2.03 %** |
| `morrowmake` → `mm-k3` | **-4.45 %** / -0.69 % / -0.24 % | **-4.79 %** / -0.60 % / -0.14 % |

**Reading.**

- **The engine gap depends on the context length.** With the same weights, Morrowmake's engine is slightly
  worse than GNmac's on windows of 1,024 tokens (+0.15 %, French +0.32 %) and better from 32k tokens on
  (-0.57 % overall, -2.0 % on the three code windows of the 32k set). *Inference:* Morrowmake reproduces the reference's FP4 rounding of the
  compressed cache and of the indexer keys, where GNmac stores `fp8_ds_mla` records and an FP8 indexer cache;
  the indexer chooses which compressed positions each query reads, and that choice only gets hard when there
  are many positions to choose from.
- **The graft works in both engines, by the same amount.** -1.81 % with Morrowmake, -1.80 % with GNmac on
  long windows, -4.45 % on French in both.
- **The engine and the graft add up.** From our previous production (GNmac + graft) to the current
  configuration (Morrowmake + graft + compact cache): -0.58 % on 32k prose, -1.64 % on 32k code with all 16
  windows better, and +0.14 % on 1,024-token windows. We chose the long contexts: agentic and coding work
  lives there.
- **The compact KV cache changes nothing** (2.3).
- **4-bit experts are at the plateau.** Layers 36-39 at 4 bits gain 0.11 % on long contexts and nothing
  measurable on short windows, against 1.8 % for the 2 → 3-bit graft. Layers 0, 1, 3 and 6 at 4 bits gained
  nothing on the GNmac engine (+0.08 % [-0.01, +0.18], [results-gnmac.md, 8.3](results-gnmac.md#83-layers-0-1-3-6-at-4-bits-k4x4)).

Data, all in `data/nll/`: long and short windows `ours-A1*.json`, `k29*.json`, `k3*.json`,
`morrowmake-B*.json`, `mm-k3*.json.gz`, `mm-kvc15*.json.gz`, `mm-k3k4*.json.gz`,
`mm-kvc-faithful.json.gz`; 32k prose in `long/`; 32k code in `codex/`. Run
`python3 tests/nll_eval.py compare A B` on any pair to reproduce a row.

### 2.3 The compact KV cache is bit-identical

Morrowmake with the graft, BF16 cache (`mm-k3`) against compact cache (`mm-kvc15`), checked with
[`tests/identical.py`](../tests/identical.py):

| Set | Values compared | Different |
|---|---:|---:|
| Long windows | 196,416 | 0 |
| Short windows | 46,800 | 0 |
| 32k prose | 622,573 | 0 |
| 32k code | 524,272 | 0 |
| Self-check, generated tokens | 768 | 0 |

The design and the GPU unit test are in [compact-kv-cache.md](compact-kv-cache.md).

### 2.4 DSpark

DSpark is Morrowmake's speculative decoder (a drafter proposes 7 tokens, the model verifies them). With it on,
the self-check generates the same 768 greedy tokens: 308 of the 768 decode log-probabilities differ, by at
most 1.9e-5 (mean 8e-7), because the verification step scores several tokens at once and sums in another
order. *Inference:* a different token at any position would change its log-probability by far more than
2e-5, so the generated text is the same. It is not bit-identical in the log-probabilities.

Data: `data/nll/sc-mm-kvc15.json.gz` against `sc-mm-kvc-dspark.json.gz` (measured with compact v1, whose
self-check is bit-identical to v1.5's without DSpark).

## 3. KV pool

Tokens of KV cache vLLM reports at start, 262,144-token context, 8 sequences (the engine lines are in
`data/logs/09-2026-10-31-server-starts-and-prefill.txt`; the GNmac rows come from [results-gnmac.md](results-gnmac.md)):

| Configuration | KV memory per card | KV pool | Full-length requests at once |
|---|---:|---:|---:|
| `ours` (GNmac, `fp8_ds_mla`, 0.98) | 5.76 GiB | 2,717,420 | 10.4 |
| `ours` + DSpark (5 tokens, 0.98) | | 1,413,911 | 5.4 |
| `morrowmake` (2.9 pack, BF16, 0.95, no DSpark) | 7.76 GiB | 2,154,551 | 8.2 |
| `mm-k3` (graft, BF16, 0.95) | 5.78 GiB | 1,604,950 | 6.1 |
| compact v1 | 5.78 GiB | 2,738,502 | 10.4 |
| **`mm-kvc15`, no DSpark** | **5.78 GiB** | **3,189,000** | **12.2** |
| **`mm-kvc15` + DSpark (served)** | **2.53 GiB** | **1,360,109** | **5.2** |
| `mm-k3k4` | | 2,317,960 | 8.8 |

- The graft costs 1.98 GiB per card (2.15 M → 1.60 M tokens with BF16 rows); the compact cache more than
  pays for it.
- **0.98 is not usable with Morrowmake** on this machine: it gives 2,133,214 tokens (BF16 rows) and the same
  log-probabilities as 0.95, but it ran out of memory on a 2,001-token `prompt_logprobs` request and on a
  plain 3.5k-token request. Morrowmake stays at its default, 0.95.

## 4. Speed

### 4.1 MiaAI-Lab protocol (short prompts)

Temperature 0, thinking off, 400-token cap; one user = median streaming rate of 5 runs; 2, 4 and 8 users =
completion tokens / wall time of a batch of simultaneous requests. tok/s, structured / code / prose:

| Configuration | 1 user | 2 users | 4 users | 8 users |
|---|---|---|---|---|
| GNmac + graft, no DSpark (previous production) | 105.4 / 105.0 / 104.8 | 174 / 148 / 168 | 325 / 282 / 289 | 560 / 482 / 449 |
| GNmac + graft + DSpark | 366 / 268 / 202 | 454 / 261 / 258 | 659 / 399 / 365 | 884 / 563 / 476 |
| **`mm-kvc15` + DSpark (served)** | **267 / 212 / 153** | 349 / 169 / 217 | 485 / 306 / 295 | **570 / 291 / 349** |
| `mm-kvc15`, no DSpark | 72.1 / 72.9 / 72.0 | 144 / 93 / 139 | 257 / 183 / 251 | 397 / 245 / 341 |
| `mm-k3k4`, no DSpark | 62.4 / 64.5 / 63.6 | 129 / 96 / 124 | 225 / 168 / 218 | 334 / 215 / 283 |
| `mm-kvc-faithful` (compact v1), no DSpark | 30.8 / 31.1 / 30.6 | 66 / 57 / 62 | 111 / 67 / 107 | 150 / 111 / 132 |

- **The code column does not compare across engines.** The code prompt's answer is 48 tokens long on
  Morrowmake and 95 on GNmac (`completion_tokens_median` in the JSON files); shorter answers carry more fixed
  cost per token, so compare the engines on the structured and prose columns. (`mm-kvc-faithful` also answered
  95 tokens.)
- **The current configuration is slower than our previous one.** GNmac's engine decodes about 1.45x faster
  at one stream without DSpark (105 against 72 tok/s), and 1.3-1.4x faster with DSpark (structured 366 against
  267, prose 202 against 153). We traded that speed for the long-context quality of 2.2.
- DSpark multiplies single-stream speed by 2.1 to 3.7 on Morrowmake; at 8 users the gain is 1.0 to 1.4x.
- Data: `data/bench/mia-json-2026-10-31/` (one JSON per configuration, prompt and concurrency; `mm-kvc15-*`,
  `gnmac-dspark-*`, `mm-k3k4-*`, `mm-kvc-faithful-*`), the GNmac no-DSpark row from
  [results-gnmac.md](results-gnmac.md).

### 4.2 Long contexts, one stream

[`tests/speed.py`](../tests/speed.py): a unique random prefix per prompt (prefix cache hit rate checked at
0 %), one warm-up per length, natural output of at most 256 tokens. Decode tok/s / prefill tok/s:

| Prompt | GNmac + graft | GNmac + DSpark | Morrowmake + graft, BF16 | **`mm-kvc15`** | compact v1 + DSpark |
|---|---|---|---|---|---|
| 3.5k | 103.2 / 2,615 | 318 / 2,400 | 69.3 / 3,815 | **66.7 / 3,594** | 246 / 3,437 |
| 33k | 101.0 / 2,573 | 206 / 2,405 | 66.3 / 4,104 | **64.8 / 3,834** | 124 / 3,440 |
| 66k | 101.2 / 2,393 | 225 / 2,211 | 64.4 / 3,939 | **63.1 / 3,654** | 129 / 3,118 |
| 132k | 101.4 / 2,088 | 207 / 1,928 | 62.9 / 3,600 | **60.7 / 3,304** | 155 / 2,613 |
| 259k | 100.7 / 1,628 | 333 / 1,532 | 59.8 / 3,081 | **56.9 / 2,776** | 123 / 1,985 |
| Time to first token at 259k | 159 s | 169 s | 84 s | **93 s** | 131 s |

- **Morrowmake reads long prompts 1.4 to 1.7x faster** than GNmac, so the time to the first token at 259k
  drops from 159 s to 93 s even with the compact cache. GNmac decodes faster at every length.
- With DSpark, decode speed depends on what the drafter guesses: these summaries of repetitive text are
  easy at some lengths and not at others.
- Cold prefill on real text (median of 9 runs, three books of 24-38k tokens): GNmac 2,349, Morrowmake BF16
  4,004, compact v1 3,392, `mm-kvc15` 3,749 tok/s (`data/logs/09-2026-10-31-server-starts-and-prefill.txt`).
- The "aggregate" figures `speed.py` also prints are dominated by the concurrent prefills and are not
  comparable with the MiaAI aggregates; we do not quote them.
- Data: `data/bench/speed/speed-*.json` (`gnmac`, `gnmac-dspark`, `mm-k3` = BF16, `mm-kvc` = compact v1,
  `mm-kvc15`, `mm-kvc-dspark`).

## 5. Contexts beyond 262,144 tokens

DeepSeek-V4.1-Flash declares 1,048,576 positions (YaRN, factor 16 over 65,536). Morrowmake validates 262,144.
We started `mm-kvc15` without DSpark and with `--max-model-len 1048576`:

- KV pool **4,211,001 tokens** (5.67 GiB per card): **4 requests of 1M tokens at once**. vLLM counts the
  sliding-window layers' fixed per-request cost against the maximum length, so the pool is quoted larger
  than at 262,144 for nearly the same memory.

- Exact recall ([`tests/longctx.py`](../tests/longctx.py)): 8 random access codes hidden from 5 % to 95 % depth
  in a document of public-domain books, then a question asking for all of them, temperature 0, thinking off:

  | Prompt | Recall | Time to first token | Decode |
  |---:|---:|---:|---:|
  | 131,238 tokens | **8 / 8** | 40.6 s | 54.5 tok/s |
  | 522,280 tokens | **8 / 8** | 256 s | 49.6 tok/s |
  | 996,972 tokens | **8 / 8** | 709 s (about 1,400 tok/s) | 46.0 tok/s |

- **A 1M maximum slows every request down.** Short prompts (MiaAI protocol) on the same server: one user
  57 / 57 / 57 tok/s against 72 / 73 / 72 with the 262,144 maximum, 8 users 296 / 186 / 264 against 397 / 245 /
  341 (-21 % to -25 %). *Inference:* the engine sizes some per-step work by the maximum length (its full CUDA
  graphs replay their captured score width). We therefore keep 262,144 as the default and run 1M as a separate
  profile.
- This is a recall test, not a measure of quality at 1M tokens: it shows that positions up to 1M are read
  back exactly, not how well the model reasons over them.
- Data: `data/bench/longctx-1m.json`, `data/bench/mia-json-2026-10-31/mm-1m-nospec-*.json`,
  `data/logs/09-2026-10-31-server-starts-and-prefill.txt`.

## 6. Settings we tried and rejected

| Setting | Quality | Cost | Decision |
|---|---|---|---|
| Layers 36-39 at 4 bits | -0.11 % on long contexts, nothing on short windows | KV 3.19 M → 2.32 M, one stream -12 to -13 %, 8 users -12 to -17 % (4-bit experts have no fast MoE kernel in Morrowmake) | rejected |
| Release shortcuts off (BF16 all-reduce of `wo_b`, the MoE v2 kernels, one rounding per MoE all-reduce) | +0.04 % [-0.05, +0.13], code +0.00 % | about 2.3x slower | rejected: the release's shortcuts cost nothing measurable |
| Memory setting 0.98 | identical log-probabilities | out of memory on normal requests | rejected, 0.95 |
| No DSpark | identical text | 3.7x slower single stream on structured output | kept as the second profile, for the 3.19 M-token pool |

## 7. What we did not measure

- Distance to the unquantized model: every perplexity change here is relative between our configurations.
- Downstream tasks (HumanEval, GSM8K) on the current configuration. On the previous comparison they could not
  separate the stacks ([comparison-morrowmake.md](comparison-morrowmake.md)).
- Quality at 1M tokens beyond the recall test of section 5.
- Reproducibility across restarts of the 32k-code set (no repeat run).
- The 32k code set on the GNmac 2.9 pack (`k29`), so the "engine only" row has no 32k-code cell.
- Power and energy per token.
