# Results: the current configuration (Morrowmake engine, 3-bit graft, compact KV cache)

This page lists what we measured from 2026-10-30 to 2026-11-01, with the file each number comes from. The
measurements of GNmac-lab 1.2.0 with our patches (Engram, copy and stress tests) are in [results-gnmac.md](results-gnmac.md).

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
| `ours` | GNmac-lab 1.2.0 (`2ad5741ab1`) with our patches | LibertAI 2.9 pack, layers 18-22 grafted to 3 bits | `fp8_ds_mla` | 0.98 |
| `k29` | GNmac-lab 1.2.0 | LibertAI 2.9 pack as published (layers 18-22 at 2 bits) | `fp8_ds_mla` | 0.97 |
| `m101-release` | Morrowmake 1.0.1 (tag `v1.0.1`, plugin `0b34823c`) as published | LibertAI 2.9 pack as published | BF16 | 0.95 |
| `m101-bf16` | Morrowmake 1.0.1 | layers 18-22 grafted to 3 bits | BF16 | 0.95 |
| **`m101-kvc`** | **Morrowmake 1.0.1 + compact KV v1.5** | **layers 18-22 grafted to 3 bits** | **compact (lossless)** | **0.95** |
| `morrowmake` | Morrowmake 1.0.0 (`7f9023c`, engine `a0c57b52`) as published | LibertAI 2.9 pack as published | BF16 | 0.95 |
| `mm-k3` | Morrowmake 1.0.0 | layers 18-22 grafted to 3 bits | BF16 | 0.95 |
| `mm-kvc15` | Morrowmake 1.0.0 + compact KV v1.5 | layers 18-22 grafted to 3 bits | compact (lossless) | 0.95 |
| `mm-k3k4` | as `mm-kvc15` | plus layers 36-39 at 4 bits | compact | 0.95 |
| `mm-kvc-faithful` | as `mm-kvc15` with compact v1, release shortcuts off | as `mm-kvc15` | compact v1 | 0.95 |

`m101-kvc` is what we serve since 2026-11-01, with DSpark on (the release default); `mm-kvc15` was the same recipe on
1.0.0, and the tests that were not repeated on 1.0.1 say so. Compact v1 (`mm-kvc`) and v1.5 give bit-identical log-probabilities (`data/nll/mm-kvc.json.gz`, `sc-mm-kvc.json.gz`); v1.5 has faster kernels and a
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
| **GNmac → Morrowmake + compact cache, same grafted weights** (`ours` → `m101-kvc`) | +0.17 % [+0.06, +0.30] | +0.23 % [-0.09, +0.55] | **-0.54 % [-0.70, -0.38]** (15 of 19) | **-1.62 % [-1.89, -1.35]** (16 of 16) |
| Engine only, same weights (`k29` → `m101-release`) | +0.14 % [+0.02, +0.25] | -0.12 % [-0.44, +0.20] | -0.57 % [-0.73, -0.42] (14 of 19) | not run |
| Graft 2 → 3 bits, Morrowmake engine (`m101-release` → `m101-kvc`) | **-1.76 % [-1.97, -1.55]** (150 of 192) | -0.23 % [-0.57, +0.12] | **-2.28 % [-2.70, -1.75]** (18 of 19) | -0.06 % [-0.11, -0.02] |
| Graft 2 → 3 bits, GNmac engine (`k29` → `k3`), for reference | -1.80 % [-2.00, -1.60] | -0.57 % [-0.92, -0.22] | not run | not run |
| Compact KV cache (`m101-bf16` → `m101-kvc`) | identical | identical | identical | identical |
| Layers 36-39 at 4 bits, on 1.0.0 (`mm-kvc15` → `mm-k3k4`) | -0.04 % [-0.08, +0.01] | -0.08 % [-0.22, +0.07] | -0.11 % [-0.15, -0.07] (17 of 19) | -0.11 % [-0.14, -0.09] (16 of 16) |
| Release shortcuts off, on 1.0.0 (`mm-kvc15` → `mm-kvc-faithful`) | +0.04 % [-0.05, +0.13] | not run | not run | +0.00 % [-0.04, +0.05] |

German and Italian (64 windows of 1,024 tokens from 8 Gutenberg books each, `data/nll/lang/`), `m101-release` →
`m101-kvc`: German **-3.06 % [-3.78, -2.35]** (54 of 64 windows, every book better), Italian -0.98 % [-1.63, -0.38]
(29 of 64). In Italian the two best-known books gain 5.2 % and 4.5 %; of the six lesser-known novels, five lose
0.17 to 0.63 % and one gains 0.23 %. *Inference:* part of the graft's gain may be recall of text seen in training.

Per corpus, for the rows that matter:

| A → B | Long windows: French / English / code | 32k: French / English / code |
|---|---|---|
| `ours` → `m101-kvc` | +0.38 % / +0.12 % / +0.02 % | -0.47 % / -0.12 % / **-1.84 %** |
| `k29` → `m101-release` | +0.31 % / +0.11 % / -0.01 % | -0.54 % / -0.07 % / **-1.96 %** |
| `m101-release` → `m101-kvc` | **-4.39 %** / -0.68 % / -0.16 % | **-4.72 %** / -0.62 % / -0.04 % |

**Reading.**

- **The engine gap depends on the context length.** With the same weights, Morrowmake's engine is slightly
  worse than GNmac's on windows of 1,024 tokens (+0.14 %, French +0.31 %) and better from 32k tokens on
  (-0.57 % overall, -2.0 % on the three code windows of the 32k set). *Inference:* Morrowmake reproduces the reference's FP4 rounding of the
  compressed cache and of the indexer keys, where GNmac stores `fp8_ds_mla` records and an FP8 indexer cache;
  the indexer chooses which compressed positions each query reads, and that choice only gets hard when there
  are many positions to choose from.
- **The graft works in both engines, by about the same amount.** -1.76 % with Morrowmake, -1.80 % with GNmac on
  long windows, -4.39 % and -4.45 % on French.
- **The engine and the graft add up.** With the same grafted weights, from GNmac's engine to Morrowmake's with
  the compact cache: -0.54 % on 32k prose, -1.62 % on 32k code with all 16 windows better, and +0.17 % on
  1,024-token windows. Against the two recipes as released: -1.76 % (Morrowmake) and -1.63 % (GNmac) on
  1,024-token windows, -2.28 % and -2.83 % on 32k prose (`m101-release` and `k29` against `m101-kvc`). We chose the long contexts: agentic and coding work
  lives there.
- **The compact KV cache changes nothing** (2.3).
- **4-bit experts are at the plateau.** Layers 36-39 at 4 bits gain 0.11 % on long contexts and nothing
  measurable on short windows, against 1.8 % for the 2 → 3-bit graft. Layers 0, 1, 3 and 6 at 4 bits gained
  nothing on the GNmac engine (+0.08 % [-0.01, +0.18], [results-gnmac.md, 8.3](results-gnmac.md#83-layers-0-1-3-6-at-4-bits-k4x4)).

Data, all in `data/nll/`: long and short windows `m101-release*.json.gz`, `m101-bf16*.json.gz`, `m101-kvc*.json.gz`,
`ours-A1*.json`, `k29*.json`, `k3*.json`, `morrowmake-B*.json`, `mm-k3*.json.gz`, `mm-kvc15*.json.gz`, `mm-k3k4*.json.gz`,
`mm-kvc-faithful.json.gz`; 32k prose in `long/`; 32k code in `codex/`. Run
`python3 tests/nll_eval.py compare A B` on any pair to reproduce a row.

### 2.3 The compact KV cache is bit-identical

Morrowmake 1.0.1 with the graft, BF16 cache (`m101-bf16`) against compact cache (`m101-kvc`), checked with
[`tests/identical.py`](../tests/identical.py). The same comparison on 1.0.0 (`mm-k3` against `mm-kvc15`) gave the same
result on the same sets.

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
the self-check generates the same 768 greedy tokens: rescoring them through the prefill path gives 768 identical
values out of 768, which a single different token would break. The decode log-probabilities are not bit-identical,
because the verification step scores several tokens at once and sums in another order: 323 of 768 differ, median
7e-7, 99th percentile 0.004, maximum 0.044 (one token, on one sequence). On 1.0.0 the maximum was 1.9e-5 (308 of 768),
so 1.0.1's drafting path moves these values more; the text does not change.

Data: `data/nll/sc-m101-kvc.json.gz` against `sc-m101-kvc-dspark.json.gz`; on 1.0.0, `sc-mm-kvc15.json.gz` against
`sc-mm-kvc-dspark.json.gz` (compact v1, whose self-check is bit-identical to v1.5's without DSpark).

### 2.5 From Morrowmake 1.0.0 to 1.0.1

1.0.1 changes the base (upstream vLLM `330c5bffee`, torch 2.14), splits each routed expert across the four cards and
rounds the all-reduce sums once. Perplexity change, 1.0.0 → 1.0.1, same weights on both sides:

| | 1,024 | 40 | 32k prose | 32k code | German / Italian |
|---|---|---|---|---|---|
| As released (`morrowmake` → `m101-release`) | -0.02 % [-0.12, +0.10] | +0.17 % [-0.09, +0.44] | +0.00 % [-0.03, +0.04] | +0.03 % [+0.00, +0.06] | +0.19 % [-0.01, +0.40] / +0.01 % [-0.18, +0.20] |
| Our recipe (`mm-kvc15` → `m101-kvc`) | +0.03 % [-0.07, +0.14] | **+0.44 % [+0.18, +0.70]** | +0.04 % [-0.01, +0.09] | +0.02 % [-0.01, +0.06] | +0.16 % [-0.05, +0.37] / -0.20 % [-0.42, +0.01] |

Nothing moves from 1,024 tokens up. **The 40-token windows of our recipe lose 0.44 %**, French +0.72 %, code +0.45 %,
English +0.15 %. We looked for the switch responsible, cutting one at a time on `m101-kvc`, against `mm-kvc15`:

| Switch | 40-token windows | 1,024-token windows |
|---|---|---|
| `VLLM_SM80_ALLREDUCE_ROUND_ONCE=0` (`m101-noro`) | +0.44 %, bit-identical to `m101-kvc` | -0.02 % [-0.11, +0.08] |
| `VLLM_EXL3_MOE_EXPERT_ROTATION=0` (`m101-norot`) | +0.33 % [+0.07, +0.59] | -0.01 % [-0.12, +0.09] |

Neither explains it, and the remainder is not the compact cache either (bit-identical to BF16 on 1.0.1). We did not
test both off together, nor the new base alone, so **the cause is unknown**. The release itself moves by +0.17 %
within noise on this set, so the same cause may apply to it. We adopted 1.0.1 anyway: long contexts are what we chose
this engine for, and they did not move.

Our recipe, 1.0.0 → 1.0.1, speed and memory (tok/s, structured / code / prose; both answer the code prompt in 48
tokens alone, not the same totals in batches):

| | 1.0.0 | 1.0.1 |
|---|---|---|
| DSpark, 1 user | 267 / 212 / 153 | 288 / 251 / 155 |
| DSpark, 2 users | 349 / 169 / 217 | 367 / 167 / 195 |
| DSpark, 4 users | 485 / 306 / 295 | 511 / 249 / 301 |
| DSpark, 8 users | 570 / 291 / 349 | 639 / 307 / 362 |
| No DSpark, 1 user | 72 / 73 / 72 | 79 / 80 / 79 |
| No DSpark, 8 users | 397 / 245 / 341 | 421 / 253 / 345 |
| KV pool, DSpark / no DSpark | 1,360,109 / 3,189,000 | 1,466,159 / 3,090,900 |

Each figure comes from one boot. Morrowmake publishes +14 / +12 / +9 % at 8 users for its release on its machine; we
measure +12 / +5 / +4 % on our recipe.

Data: `data/nll/m101-noro*.json.gz`, `m101-norot*.json.gz`, the files of section 2.2 and
`data/nll/{long,codex,lang}/`.

## 3. KV pool

Tokens of KV cache vLLM reports at start, 262,144-token context, 8 sequences (the engine lines are in
`data/logs/10-2026-11-01-server-starts-and-prefill-1.0.1.txt` and, for 1.0.0, in
`data/logs/09-2026-10-31-server-starts-and-prefill.txt`; the GNmac rows come from [results-gnmac.md](results-gnmac.md)):

| Configuration | KV memory per card | KV pool | Full-length requests at once |
|---|---:|---:|---:|
| **`m101-kvc` + DSpark (served)** | **2.73 GiB** | **1,466,159** | **5.6** |
| **`m101-kvc`, no DSpark** | **5.60 GiB** | **3,090,900** | **11.8** |
| `m101-bf16` (graft, BF16, no DSpark) | 5.60 GiB | 1,555,036 | 5.9 |
| `m101-release`, DSpark | 4.73 GiB | 1,296,145 | 4.9 |
| `m101-release`, no DSpark | 7.58 GiB | 2,105,722 | 8.0 |
| `ours` (GNmac, `fp8_ds_mla`, 0.98) | 5.76 GiB | 2,717,420 | 10.4 |
| `ours` + DSpark (5 tokens, 0.98) | | 1,413,911 | 5.4 |
| On 1.0.0: `morrowmake`, no DSpark | 7.76 GiB | 2,154,551 | 8.2 |
| On 1.0.0: `morrowmake`, DSpark | 4.56 GiB | 1,247,657 | 4.8 |
| On 1.0.0: `mm-k3` (graft, BF16) | 5.78 GiB | 1,604,950 | 6.1 |
| On 1.0.0: compact v1 | 5.78 GiB | 2,738,502 | 10.4 |
| On 1.0.0: `mm-kvc15`, no DSpark / DSpark | 5.78 / 2.53 GiB | 3,189,000 / 1,360,109 | 12.2 / 5.2 |
| On 1.0.0: `mm-k3k4` | | 2,317,960 | 8.8 |

- The graft costs 1.98 GiB per card (2.11 M → 1.56 M tokens with BF16 rows); the compact cache more than
  pays for it.
- **0.98 is not usable with Morrowmake** on this machine (tested on 1.0.0): it gives 2,133,214 tokens (BF16 rows) and
  the same log-probabilities as 0.95, but it ran out of memory on a 2,001-token `prompt_logprobs` request and on a
  plain 3.5k-token request. Morrowmake stays at its default, 0.95.

## 4. Speed

### 4.1 MiaAI-Lab protocol (short prompts)

Temperature 0, thinking off, 400-token cap; one user = median streaming rate of 5 runs; 2, 4 and 8 users =
completion tokens / wall time of a batch of simultaneous requests. tok/s, structured / code / prose:

| Configuration | 1 user | 2 users | 4 users | 8 users |
|---|---|---|---|---|
| GNmac + graft, no DSpark | 105.4 / 105.0 / 104.8 | 174 / 148 / 168 | 325 / 282 / 289 | 560 / 482 / 449 |
| GNmac + graft + DSpark | 366 / 268 / 202 | 454 / 261 / 258 | 659 / 399 / 365 | 884 / 563 / 476 |
| **`m101-kvc` + DSpark (served)** | **288 / 251 / 155** | 367 / 167 / 195 | 511 / 249 / 301 | **639 / 307 / 362** |
| `m101-kvc`, no DSpark | 79.1 / 80.3 / 79.1 | 156 / 100 / 149 | 267 / 190 / 259 | 421 / 253 / 345 |
| `m101-release` + DSpark | 297 / 235 / 158 | 375 / 147 / 212 | 521 / 238 / 300 | 624 / 378 / 377 |
| On 1.0.0: `mm-k3k4`, no DSpark | 62.4 / 64.5 / 63.6 | 129 / 96 / 124 | 225 / 168 / 218 | 334 / 215 / 283 |
| On 1.0.0: `mm-kvc-faithful` (compact v1), no DSpark | 30.8 / 31.1 / 30.6 | 66 / 57 / 62 | 111 / 67 / 107 | 150 / 111 / 132 |

- **The code column does not compare across configurations.** At one user `m101-release` and `m101-kvc` both answer
  the code prompt in 48 tokens, GNmac in 95. In batches the greedy answers differ again (at 8 users, 760 tokens in
  all for `m101-release`, 478 for `m101-kvc`; `completion_tokens_median` in the JSON files). Shorter answers carry more
  fixed cost per token, so compare on the structured and prose columns.
- **Against Morrowmake 1.0.1 as released,** we are 3 % and 2 % slower at one user (structured, prose), and +2 % and
  -4 % at 8 users; the 2- and 4-user prose points are noisier (-8 %, +0 %).
- **Morrowmake's published figures are higher than what we measure.** It publishes 297 / 297 / 170 at one user and
  868 / 795 / 464 at 8 on its own machine (180 W), we measure 297 / 235 / 158 and 624 / 378 / 377 for the same release
  here (200 W). The gap was there at 1.0.0 (published 760 structured at 8 users, measured here 585). We do not know
  how much comes from the machine and how much from the method.
- **Our configuration is slower than GNmac's engine.** GNmac's engine decodes about 1.33x faster at one stream
  without DSpark (105 against 79 tok/s), and about 1.3x with DSpark (structured 366 against 288, prose 202 against
  155). We traded that speed for the long-context quality of 2.2.
- DSpark multiplies single-stream speed by 2.0 to 3.6 on our configuration; at 8 users the gain is 1.05 to 1.5x.
- Data: `data/bench/mia-json-2026-11-01/` (one JSON per configuration, prompt and concurrency; `m101-kvc-dspark-*`,
  `m101-kvc-nospec-*`, `m101-release-dspark-*`, `m101-1m-nospec-*`), `data/bench/mia-json-2026-10-31/` (1.0.0 and
  `gnmac-dspark-*`), the GNmac no-DSpark row from [results-gnmac.md](results-gnmac.md).

### 4.2 Long contexts, one stream

[`tests/speed.py`](../tests/speed.py): a unique random prefix per prompt (prefix cache hit rate checked at
0 %), one warm-up per length, natural output of at most 256 tokens. Decode tok/s / prefill tok/s:

| Prompt | GNmac + graft | GNmac + DSpark | `m101-release` + DSpark | **`m101-kvc`** | **`m101-kvc` + DSpark** |
|---|---|---|---|---|---|
| 3.5k | 103.2 / 2,615 | 318 / 2,400 | 285.7 / 3,683 | **73.8 / 3,587** | **257.5 / 3,527** |
| 33k | 101.0 / 2,573 | 206 / 2,405 | 179.9 / 3,935 | **72.8 / 3,818** | **178.9 / 3,755** |
| 66k | 101.2 / 2,393 | 225 / 2,211 | 277.2 / 3,758 | **72.7 / 3,632** | **144.9 / 3,579** |
| 132k | 101.4 / 2,088 | 207 / 1,928 | 204.5 / 3,431 | **71.9 / 3,284** | **155.1 / 3,243** |
| 259k | 100.7 / 1,628 | 333 / 1,532 | 214.6 / 2,958 | **69.9 / 2,787** | **243.5 / 2,744** |
| Time to first token at 259k | 159 s | 169 s | 88 s | **93 s** | **95 s** |

- **Morrowmake reads long prompts 1.4 to 1.7x faster** than GNmac, so the time to the first token at 259k
  drops from 159 s to 93 s even with the compact cache. GNmac decodes faster at every length.
- The compact cache reads prompts 3-6 % slower than the release (88 s against 93 s at 259k).
- With DSpark, decode speed depends on what the drafter guesses: these summaries of repetitive text are
  easy at some lengths and not at others.
- Cold prefill on real text (median of 9 runs, three books of 24-38k tokens): GNmac 2,349, `m101-release` 3,724,
  `m101-kvc` 3,621 tok/s (`data/logs/10-2026-11-01-server-starts-and-prefill-1.0.1.txt`).
- The "aggregate" figures `speed.py` also prints are dominated by the concurrent prefills and are not
  comparable with the MiaAI aggregates; we do not quote them.
- Data: `data/bench/speed/speed-*.json` (`gnmac`, `gnmac-dspark`, `m101-release-dspark`, `m101-kvc-nospec`,
  `m101-kvc-dspark`; on 1.0.0 `mm-k3` = BF16, `mm-kvc` = compact v1, `mm-kvc15`, `mm-kvc-dspark`,
  `mm-release-dspark`).

## 5. Contexts beyond 262,144 tokens

DeepSeek-V4.1-Flash declares 1,048,576 positions (YaRN, factor 16 over 65,536). Morrowmake validates 262,144.
We started `m101-kvc` without DSpark and with `--max-model-len 1048576`:

- KV pool **4,063,104 tokens** (5.48 GiB per card): **3.9 requests of 1M tokens at once**. vLLM counts the
  sliding-window layers' fixed per-request cost against the maximum length, so the pool is quoted larger
  than at 262,144 for nearly the same memory.

- Exact recall ([`tests/longctx.py`](../tests/longctx.py)): 8 random access codes hidden from 5 % to 95 % depth
  in a document of public-domain books, then a question asking for all of them, temperature 0, thinking off:

  | Prompt | Recall | Time to first token | Decode |
  |---:|---:|---:|---:|
  | 131,238 tokens | **8 / 8** | 41.5 s | 58.7 tok/s |
  | 522,280 tokens | **8 / 8** | 258 s | 55.7 tok/s |
  | 996,972 tokens | **8 / 8** | 703 s (about 1,400 tok/s) | 51.5 tok/s |

- **A 1M maximum slows every request down.** Short prompts (MiaAI protocol) on the same server: one user
  61 / 61 / 61 tok/s against 79 / 80 / 79 with the 262,144 maximum, 8 users 313 / 174 / 279 against 421 / 253 /
  345 (-19 % to -31 %). *Inference:* the engine sizes some per-step work by the maximum length (its full CUDA
  graphs replay their captured score width). We therefore keep 262,144 as the default and run 1M as a separate
  profile.
- This is a recall test, not a measure of quality at 1M tokens: it shows that positions up to 1M are read
  back exactly, not how well the model reasons over them.
- Data: `data/bench/longctx-m101-1m.json`, `data/bench/mia-json-2026-11-01/m101-1m-nospec-*.json`,
  `data/logs/10-2026-11-01-server-starts-and-prefill-1.0.1.txt`; on 1.0.0, `data/bench/longctx-1m.json`.

## 6. Settings we tried and rejected

Measured on 1.0.0, not repeated on 1.0.1, except the no-DSpark row.

| Setting | Quality | Cost | Decision |
|---|---|---|---|
| Layers 36-39 at 4 bits | -0.11 % on long contexts, nothing on short windows | KV 3.19 M → 2.32 M (on 1.0.0), one stream -12 to -13 %, 8 users -12 to -17 % (4-bit experts have no fast MoE kernel in Morrowmake) | rejected |
| Release shortcuts off (BF16 all-reduce of `wo_b`, the MoE v2 kernels, one rounding per MoE all-reduce) | +0.04 % [-0.05, +0.13], code +0.00 % | about 2.3x slower | rejected: the release's shortcuts cost nothing measurable |
| Memory setting 0.98 | identical log-probabilities | out of memory on normal requests | rejected, 0.95 |
| No DSpark | identical text | 3.6x slower single stream on structured output | kept as the second profile, for the 3.09 M-token pool |

## 7. What we did not measure

- Distance to the unquantized model: every perplexity change here is relative between our configurations.
- Downstream tasks (HumanEval, GSM8K) on the current configuration. Morrowmake reports 1.0.1 level with 1.0.0 on
  its release; on our previous comparison they could not separate the stacks ([comparison-morrowmake.md](comparison-morrowmake.md#a04-humaneval-and-gsm8k)).
- Quality at 1M tokens beyond the recall test of section 5.
- The cause of the 40-token loss of section 2.5.
- Speed of `m101-bf16`: on 1.0.1 the cost of the compact cache alone is known only against the release, whose weights
  differ.
- Reproducibility across restarts of the 32k-code set (no repeat run).
- The 32k code set on the GNmac 2.9 pack (`k29`), so the "engine only" row has no 32k-code cell.
- Power and energy per token.
