# Setup: reproducing the deployment

This guide rebuilds the configuration described in the [README](../README.md): DeepSeek-V4.1-Flash-1017 on 4x CMP 170HX, tensor parallel 4, routed experts at 3 bits in every layer (layers 18-22 grafted from the 3.5 bpw pack), Engram tables read from a tmpfs in RAM, no DSpark drafter, `gpu_memory_utilization` 0.98.

The recipe is not ours. We start from [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) at commit `2ad5741ab1` (release 1.2.0) and apply three patches, two extra scripts and an environment file. Read the upstream README first: everything it says about drivers, P2P and the image still applies.

The numbers quoted below come from our measurements. The data files are listed in [results.md](results.md).

## 1. Prerequisites

### Hardware we used

| Item | Value |
|---|---|
| Board, CPU | ASRock Rack ROMED8-2T, AMD EPYC 7F52 |
| RAM | 512 GB DDR4-3200, 8 channels |
| GPUs | 4x CMP 170HX, 64 GB each (`nvidia-smi` shows 65,536 MiB, of which 64,911 MiB are usable), 74 SMs |
| Links | PCIe Gen2 x16 on all four cards |
| Driver | NVIDIA open kernel module 610.43.02, built from a cmpunlocker fork with a P2P patch. P2P topology reports OK and NCCL passes on all four cards. We do **not** have Morrowmake's TRAP31 patch (see [comparison-morrowmake.md](comparison-morrowmake.md)) |
| Power, clocks | 200 W cap per card, 170tune +200 MHz at 1500 MHz |
| Container | Docker inside an LXC container, GPUs passed with CDI |

The upstream recipe was measured at 180 W per card. Ours ran at 200 W. Keep that in mind when you compare speeds.

### What the machine needs

- **GPUs**: four cards with 64 GB each and working P2P for TP4. The upstream README explains how to check links and P2P (`tools/p2p_check.sh`). Without P2P, set `P2P=0` (slower all-reduces).
- **Disk**: about 430 GiB for the weights. Our download came to 428 GiB. Use a fast local disk.
- **Disk for the image**: the built image is 17.8 GB, plus the build cache and the kernel compile caches. A 120 GB container root file system with 79 GB free looked too tight, so we grew it to 200 GB.
- **RAM**: about 190 GiB for the Engram tables in a tmpfs, plus the engine's own host memory. See [section 6](#6-engram-tables-in-ram-and-container-memory).
- **Software on the Docker host**:
  - Docker with the NVIDIA Container Toolkit.
  - A CDI spec that lists all four GPUs (`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`). Our patched `serve.sh` uses `--device nvidia.com/gpu=all`. Inside an LXC container, check after every boot that the spec still lists all four cards.
  - `curl`, `python3`, `git`, `patch`.
  - `bc`. Upstream `serve.sh` uses it to check that the GPUs are idle, and stops if it is missing.

## 2. Get the upstream recipe at the pinned commit

```bash
git clone https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx.git dsv41-recipe
cd dsv41-recipe
git checkout 2ad5741ab1
```

Clone this repository next to it, so that `../deepseek-v41-flash-4x-cmp170hx` points at it. The commands below assume that layout.

```bash
REPO=../deepseek-v41-flash-4x-cmp170hx   # path to this repository
```

## 3. Apply the patches

The patches are plain unified diffs against upstream `2ad5741ab1`. Apply them from the recipe root with `patch -p1`:

```bash
patch -p1 < $REPO/recipe/serve.sh.patch
patch -p1 < $REPO/recipe/start.sh.patch
patch -p1 < $REPO/recipe/tools/fetch_official_nonexpert.py.patch
```

We checked that all three apply cleanly to `2ad5741ab1` and that the patched scripts parse (`bash -n`, `py_compile`).

What they change:

| File | Change |
|---|---|
| `serve.sh` | GPUs through CDI (`--device nvidia.com/gpu=all`) instead of `--gpus all`. `SPEC_TOKENS=0` drops the `--speculative-config` flag entirely (no DSpark drafter) and skips the drafter's `DSV41_DYNK` default. `ENGRAM_STORAGE` is configurable instead of being hard-coded to `ssd`. `ENGRAM_HOST_DIR` mounts an Engram directory from anywhere (our tmpfs) at `/engram`, read-only. `DSV41_EXPERTS_ONLY_FILES_RE` also matches `of-00044` files, the 3.5 bpw shards used by the graft. `VLLM_EXL3_FAT_THRESHOLD` is passed through to the container |
| `start.sh` | Calls `./engram-ram.sh` before `./serve.sh` when `ENGRAM_HOST_DIR` is under `/opt/engram-ram/` |
| `tools/fetch_official_nonexpert.py` | Downloads from a pinned revision (`OFF_REV`) instead of `main` |

Then copy the extra files:

```bash
cp $REPO/recipe/tools/download-pinned.sh $REPO/recipe/tools/make_hybrid_graft.py tools/
cp $REPO/recipe/engram-ram.sh .
cp $REPO/recipe/env.example .env
chmod +x tools/download-pinned.sh engram-ram.sh
```

`make_hybrid_graft.py` calls upstream `tools/make_hybrid.py` from its own directory, so it has to sit in `tools/`.

## 4. Download the weights (pinned revisions)

`tools/download-pinned.sh` is a variant of upstream `tools/download.sh`. The differences:

- Every Hugging Face download uses a pinned revision. Upstream downloads from `main`.
- It also fetches the five 3.5 bpw shards that hold layers 18-22 at 3 bits.
- It does not build the hybrid. That is a separate step ([section 5](#5-build-the-hybrid-checkpoint-with-the-graft)).

| Step | Source | Revision | Size | Directory under `MODELS_DIR` |
|---|---|---|---|---|
| 1 | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), whole pack | `0be63c33` | 196.3 GiB | `DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` |
| 2 | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw), shards `model-00020` to `00024-of-00044` plus index and configs | `04a492fc` | about 24 GiB (5.24 GB per shard) | `DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-L18-22` |
| 3 | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), non-expert tensors only, fetched with HTTP range requests | `1bae0e10` | about 17.5 GiB | `DeepSeek-V4.1-Flash-1017-official-nonexpert` |
| 4 | Same official repo, Engram tables: shards `00047` and `00048-of-00048` plus index and config | `1bae0e10` | about 189 GiB | `DeepSeek-V4.1-Flash-1017-engram` |
| 5 | Curtis White's `quantization_config.json` from [CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX) | `b8b9e99` | small | `MODELS_DIR/quantization_config.json` |

Like upstream `download.sh`, the script reads `MODELS_DIR` from `.env` in the recipe root, or from the environment. It stops if `MODELS_DIR` is not set anywhere:

```bash
MODELS_DIR=/srv/models PARALLEL=4 tools/download-pinned.sh 2>&1 | tee download.log
```

The script can resume: after an interruption, run it again. `curl -C -` continues partial files, and the official non-expert fetch skips finished files. `HF_TOKEN` is used if set. Our download ran at about 105 MB/s.

The layer 18-22 shards of the 2.9 pack stay on disk. The graft does not delete anything, so you can still build the original 2.9 hybrid for an A/B test.

**Disk tip.** Put a quota on the models directory if it shares a pool with other data. A mistyped revision or path can silently start a second multi-hundred-GiB download.

### Verify the downloads

The script does not check hashes. Hugging Face publishes a SHA-256 for every LFS file (`lfs.oid` in the tree API), so whole-file downloads can be checked against the pinned revision. This standalone script (Python standard library only) does that for the two EXL3 directories and the Engram directory:

```python
#!/usr/bin/env python3
"""verify_lfs.py MODELS_DIR: compare local files with the SHA-256 that Hugging Face publishes at the pinned revisions."""
import hashlib, json, os, sys, urllib.request

M = sys.argv[1]
SETS = [
    ("LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw", "0be63c3337efbb2a0061d5b8bb4006fa186d433f",
     "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"),
    ("LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw", "04a492fc3b4d25ddbab8f2d2f07f235ec77c9010",
     "DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-L18-22"),
    ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170",
     "DeepSeek-V4.1-Flash-1017-engram"),
]
bad = checked = 0
for repo, rev, sub in SETS:
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    for e in json.load(urllib.request.urlopen(url, timeout=60)):
        path = os.path.join(M, sub, e["path"])
        if e.get("type") != "file" or "lfs" not in e or not os.path.exists(path):
            continue                      # only LFS files that were actually downloaded
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        ok = h.hexdigest() == e["lfs"]["oid"]
        checked += 1; bad += not ok
        print("OK " if ok else "BAD", sub, e["path"], flush=True)
print(f"{checked} files checked, {bad} mismatches")
sys.exit(1 if bad else 0)
```

```bash
python3 -I verify_lfs.py /srv/models
```

When we ran our own check, all 55 LFS files matched. Hashing about 410 GiB takes a while; run it in the background.

The `official-nonexpert` directory cannot be checked this way. `fetch_official_nonexpert.py` writes *new* safetensors files assembled from byte ranges of the official shards, so their hashes match nothing published. That script checks the length of every range it reads.

## 5. Build the hybrid checkpoint with the graft

```bash
python3 tools/make_hybrid_graft.py /srv/models DeepSeek-V4.1-Flash-1017-hybrid k3
```

Arguments: `MODELS_DIR OUT_NAME PROFILE`. `serve.sh` expects the output name `DeepSeek-V4.1-Flash-1017-hybrid`.

What it does:

1. It runs upstream `tools/make_hybrid.py`. That builds Curtis White's hybrid as a directory of relative symlinks: routed experts from the 2.9 pack, every other tensor from the official FP8 files, and `config.json` with Curtis's `quantization_config` (see the upstream recipe for the details).
2. For each grafted layer, it rewrites the index entries `layers.N.ffn.experts.*` to point at the 3.5 bpw shard, and links that shard.
3. It checks that:
   - the grafted layer has exactly the same expert tensor names in both packs;
   - all of a layer's experts sit in one shard, and that shard holds no other layer's experts;
   - the bit width K read from the safetensors header (last trellis dimension / 16) is the one expected;
   - no link in the result is broken.
4. It removes the links to 2.9 shards that nothing references any more.

The output directory must be empty or absent; the script refuses otherwise. Nothing is copied, so the build takes seconds. Our build gave 188,241 tensors, of which 23,040 were grafted at 3 bits, and 88 links.

Profiles:

| Profile | Layers changed | Status |
|---|---|---|
| `k3` | 18-22 from 2 to 3 bits (3.5 bpw shards `00020`-`00024`) | **Production.** All 40 layers at 3 bits |
| `k4x4` | `k3` plus layers 0, 1, 3, 6 at 4 bits | Experimental, no measurable gain (see [results.md](results.md)). Needs the four 3.5 bpw shards `00002`, `00003`, `00005` and `00008-of-00044` in `DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-K4`. The download script does not fetch them. 4-bit layers run on the slower cooperative kernel; set `VLLM_EXL3_FAT_THRESHOLD=2048` |

To build the original 2.9 pack for comparison (layers 18-22 at 2 bits), run upstream `tools/make_hybrid.py` directly with another output name. Then point `serve.sh` at it, for example by renaming the directories.

## 6. Engram tables in RAM, and container memory

### Why `ssd` mode reading from a tmpfs

**Do not leave `ENGRAM_STORAGE` empty.** Our first deployment did, intending to keep the tables in RAM through the engine's default "host" mode. On this stack that mode reserves 2 x 23.6 GiB of pinned host memory per rank and **never fills it**:

- the hybrid index does not contain `engram.embed.*`;
- vLLM skips its missing-weights check when a quantization method is set;
- the upstream author never tested this mode.

The server started, passed the smoke test and benchmarked at normal speed, but Engram layers 1 and 14 were not doing their job. It showed up only as copy errors (15/20 exact) and output that changed between runs at temperature 0. This was our configuration mistake, not an upstream bug. Upstream documents `ssd` mode only. Details are in [results.md](results.md).

Putting the tables into the index so that host mode would load them does not work. The loader reads each tensor whole, about 95 GiB per process across 4 processes, and runs out of memory.

The fix keeps the tested `ssd` code path but points it at a tmpfs:

- `engram-ram.sh` mounts a tmpfs (default size `ENGRAM_TMPFS_SIZE=200G`) and copies the two Engram shards plus index and config into it. It copies only files that are missing or whose size differs; the copy took 94 s from NVMe on our machine.
- `start.sh` (patched) runs it before `serve.sh` when `ENGRAM_HOST_DIR` is under `/opt/engram-ram/`. A tmpfs does not survive a reboot of the container, and this re-creates it.
- `serve.sh` (patched) mounts `ENGRAM_HOST_DIR` read-only at `/engram` and sets `DSV41_ENGRAM_STORAGE=ssd`.

**Paths.** `engram-ram.sh` reads the tables from `$MODELS_DIR/DeepSeek-V4.1-Flash-1017-engram` and copies them to `$ENGRAM_HOST_DIR`, mounting the tmpfs on the parent directory of `ENGRAM_HOST_DIR`. Both values come from `.env`, which `start.sh` exports. The `start.sh` patch calls the script only when `ENGRAM_HOST_DIR` is under `/opt/engram-ram/`; to use another mount point, also change that pattern in `start.sh`. The script needs the right to mount a tmpfs: root, or a privileged container.

To read the tables from disk instead, leave `ENGRAM_HOST_DIR` empty and keep `ENGRAM_STORAGE=ssd`. That is the upstream default path. We have not measured it on our machine. Upstream notes that the first pass over new text is about 10 % slower in this mode while the page cache fills. On ZFS with a small ARC it may be slower than upstream measured.

### Sizing the container's memory

- The tables take about 189 GiB of tmpfs.
- tmpfs pages count against the container's memory limit and cannot be reclaimed (our container had no swap).
- After a start with the tables in tmpfs, `free -g` inside the container showed: total 460, used 202, shared 189, available 257 (`data/logs/02-restart-engram-tmpfs-ssd.txt`). So about 203 GiB was not available while the server was idle, almost all of it the tmpfs.
- For comparison, Morrowmake's recipe reports 202-204 GiB of host RAM with the same tables in RAM and suggests about 240 GiB for comfort.
- We ran with a 460 GiB container limit throughout, including the 8 x 100k-token stress runs. We have **not** tested a smaller limit. 240 GiB is a plausible floor (inference), not a measured one.

## 7. The environment file, line by line

`recipe/env.example`, copied to `.env` in the recipe root. Both `start.sh` and `serve.sh` source it with `set -a`, so every value must be valid shell. **Quote any value that contains a space.**

| Line | Meaning |
|---|---|
| `MODELS_DIR=/opt/models` | Directory holding `DeepSeek-V4.1-Flash-1017-hybrid/` and the packs it links to. Mounted read-only at `/models` in the container. The hybrid uses relative links, so the whole directory has to be mounted, not just the hybrid |
| `LAYOUT=tp4` | Tensor parallel over 4 cards. Needs x16 links and P2P. `pp4` (pipeline parallel) exists upstream; we did not test it |
| `BIND=127.0.0.1` | Host address the API is published on. Keep it on localhost until you add a key (see [section 10](#10-security)) |
| `PORT=8011` | API port. The test scripts default to 8011. Upstream default is 8000 |
| `MAX_MODEL_LEN=262144` | Maximum context per request. Upstream supports up to 1,048,576, which shrinks the KV pool; we did not test it |
| `MAX_NUM_SEQS=8` | Maximum concurrent sequences. All our benchmarks and stress tests use 8 |
| `GPU_MEMORY_UTILIZATION=0.98` | Share of each card vLLM may use. 0.95 (upstream default) gives 1,817,233 KV tokens, 0.97 gives 2,417,392, and 0.98 gives 2,717,420. At 0.98, `prompt_logprobs` on 1,024-token windows left only 0.23 GiB per card ([section 9.8](#98-the-098-ceiling-and-prompt_logprobs)) |
| `P2P=1` | Lets NCCL use peer-to-peer (`NCCL_P2P_LEVEL=SYS`). `0` routes through host memory |
| `SPEC_TOKENS=0` | No DSpark drafter. `5` turns it on (upstream default). With it, the KV pool falls to 1,113,781 tokens at 0.97 and single-user speed rises 1.9-3.5x |
| `ENGRAM_STORAGE=ssd` | Engram tables read through the `ssd` row store. Always `ssd` on this stack; empty means host mode, which leaves the tables empty |
| `ENGRAM_HOST_DIR=/opt/engram-ram/DeepSeek-V4.1-Flash-1017-engram` | Where the tables are read from: the tmpfs prepared by `engram-ram.sh`. Empty means `MODELS_DIR/DeepSeek-V4.1-Flash-1017-engram` on disk |
| `EXTRA_ARGS=--enable-prompt-tokens-details` | Extra `vllm serve` arguments, split on whitespace. This one makes `usage.prompt_tokens_details.cached_tokens` visible, so tests can see prefix-cache hits. To add a key: `EXTRA_ARGS="--enable-prompt-tokens-details --api-key CHANGE_ME"`, with the quotes |
| `IMAGE=ds41-cmp170hx:1.2.0-fidelity` | Image tag. `start.sh` builds it with `docker/build.sh` if it does not exist |
| `NAME=ds41-flash` | Container name |
| `CACHE_DIR=/opt/dsv41-gnmac/cache` | Host directory for the compile caches (inductor, Triton, TileLang). Point it at `<recipe dir>/cache`. Keeping it makes restarts faster |
| `VLLM_EXL3_FAT_THRESHOLD=2048` | Turns off exllamav3's "fat expert" path, whose buffers are allocated lazily after the KV pool is sized. Needed with 4-bit layers. With all layers at 3 bits it has no effect; we kept it so that every configuration ran with the same environment |

## 8. Build the image and start

```bash
docker/build.sh ds41-cmp170hx:1.2.0-fidelity    # optional: start.sh builds it if missing
./start.sh                                     # engram-ram.sh, then serve.sh, then waits for /health
```

- **Build**: about 20 minutes on our machine, giving a 17.8 GB image. Upstream says 30-45 minutes. The patches do not touch the Dockerfile; the image is the upstream 1.2.0 image under another tag.
- **First start**: about 10 minutes, mostly kernel compilation and loading. Later starts reuse `CACHE_DIR`.
- **Logs**: `./start.sh logs` or `docker logs -f ds41-flash`.
- **Stop**: upstream `./stop.sh`.
- **One GPU server at a time**: `serve.sh` refuses to start if the GPUs already hold more than 2,000 MiB.

## 9. Validation sequence

Run these in order after each start, from the recipe root (`$REPO` is this repository, as in [section 2](#2-get-the-upstream-recipe-at-the-pinned-commit)). The test scripts talk to `http://127.0.0.1:<port>` and use only the Python standard library. Run them on the Docker host.

### 9.1 Startup log

```bash
docker logs ds41-flash 2>&1 | grep -E "GPU KV cache size|Available KV cache memory|engram token map|offloaded to pinned"
```

Expected:

- `Built engram token map (129280 -> 99092 ids) for layers (1, 14)`, once per rank.
- **No** line containing `offloaded to pinned`. That line means host mode, and host mode means empty tables.
- `GPU KV cache size: 2,717,420 tokens` at 0.98, or 2,417,392 at 0.97 and 1,817,233 at 0.95. A different number means different weights, a different utilisation or something else on the cards.

### 9.2 Smoke test

```bash
./start.sh smoke
```

Expected: `chat : No — 221 = 13 × 17, so it is composite.`, a `get_weather` tool call for Lisbon, `image: Red`, `SMOKE PASS` (`data/logs/08-ceiling-0.98-smoke-stress-nll.txt`).

The smoke test passed even when the Engram tables were empty. It does not validate fidelity.

### 9.3 Exact-copy test

```bash
python3 $REPO/tests/copytest-dsv41.py --port 8011 --conc 8                 # 20 lists of 40 identifiers
python3 $REPO/tests/copytest-dsv41.py --port 8011 --conc 8 --n 100 --seed 777
```

Each task asks the model to copy a list of 40 random identifiers (`word_word_5digits_8hex`) exactly, at temperature 0 with reasoning off. There are three passes:

1. Cold.
2. The same prompts again, served from the prefix cache.
3. A second conversation turn on a cached prefix.

Expected: all exact on all three passes. With `--n 100 --seed 777`, prefix-cache hits come to 92 % of prompt tokens on pass 2 and 31 % on pass 3. Our results: 20/20 and 100/100 on every pass (`data/logs/03-copytest-100-and-determinism.txt`). With the empty Engram tables we got 15/20.

### 9.4 Determinism

```bash
python3 $REPO/tests/copytest-determ.py     # 5 tasks x 5 solo repetitions, outputs compared byte for byte
python3 $REPO/tests/copytest-solo.py       # replays selected copy tasks one at a time (diagnostic)
```

Expected: `5/5 exactes, 1 sortie(s) distincte(s) sur 5` for every task (French output: 5/5 exact, 1 distinct output out of 5).

Both scripts reuse the task generator of `copytest-dsv41.py` from the same directory, and accept `--port`. Their printed messages are in French, as in our logs.

### 9.5 Stress and peak VRAM

The stress test cuts its prompts from the Gutenberg texts that upstream `bench/prefill.py` caches on its first run, so run the prefill benchmark first. It is also the cold-prefill measurement.

```bash
python3 bench/prefill.py --port 8011                     # expected ~2,500 tok/s median (ours: 2,514)
python3 $REPO/tests/stress-dsv41.py --port 8011 --conc 8 --prompt 100000 --maxtok 512 --rounds 2
```

- Each wave sends 8 concurrent requests, each with a unique 100,000-token prompt and 512 output tokens.
- The script records the peak memory per card with `nvidia-smi`, so it must run where `nvidia-smi` sees the cards.
- It reads the Gutenberg cache from `bench/.cache` relative to the current directory (the recipe root here); pass `--corpus-dir` to read it from elsewhere.
- Expected: 8/8 per wave, the server still alive, and a peak around 63.6 GB per card at 0.98 (62.9 GB at 0.97). A wave takes about 390 s.

### 9.6 Decode benchmark (optional)

```bash
python3 bench/mia.py --port 8011 --label mine     # upstream script; downloads MiaAI-Lab's bench_decode.py at a pinned commit
```

Compare with the `engramtmpfs-*` rows in [results.md](results.md). Expect about 105 tok/s for one user and 449-560 tok/s aggregate for 8 users without DSpark at 200 W.

The decode benchmark **cannot detect the Engram problem**: the run without tables decoded at the same speed.

### 9.7 NLL (optional, for comparing weight configurations)

```bash
python3 $REPO/tests/nll_eval.py build --base-url http://127.0.0.1:8011 --cache ./nllcache --code-dir .
python3 $REPO/tests/nll_eval.py build --base-url http://127.0.0.1:8011 --cache ./nllshort --seq-len 40 --n-seqs 400 --code-dir .
python3 $REPO/tests/nll_eval.py run   --base-url http://127.0.0.1:8011 --cache ./nllcache --label k3 --out runs/k3.json
python3 $REPO/tests/nll_eval.py run   --base-url http://127.0.0.1:8011 --cache ./nllshort --seq-len 40 --label k3 --out runs/k3-short.json
python3 $REPO/tests/nll_eval.py selfcheck --base-url http://127.0.0.1:8011 --cache ./nllcache --label k3 --out runs/sc-k3.json
python3 $REPO/tests/nll_eval.py compare $REPO/data/nll/k3.json runs/k3.json
```

- `build` freezes the token windows once: French and English Gutenberg books, and the upstream recipe's own source as the code set (`--code-dir` points at the recipe root).
- Our corpora are identified by their SHA-256 in `data/nll/*.json`. Your rebuilt windows are comparable with ours only if those hashes match.
- For an A/B test, use the A-B-A design described in the script's docstring: baseline, restart on the candidate, restart on the baseline again. Then `compare` all pairs.
- Run nothing else on the server during a run. Batch composition changes the numerics.

### 9.8 The 0.98 ceiling and `prompt_logprobs`

At `GPU_MEMORY_UTILIZATION=0.98` normal serving left about 1.3 GiB per card at peak: 63,580 MiB out of 64,911 MiB usable, during an 8 x 100k stress wave.

Requests that ask for `prompt_logprobs` allocate a logits buffer that vLLM's startup memory profile does not count. Scoring 1,024-token windows peaked at 64,676 MiB, a margin of 0.23 GiB per card. It did not run out of memory, but that is too close.

Before NLL runs or any workload with `prompt_logprobs` on long prompts, either restart at 0.97 or use windows of 512 tokens or fewer. Results at 0.97 and 0.98 were bit-identical (`data/nll/k3.json` vs `data/nll/k3-098.json`), so switching changes nothing but the KV pool.

## 10. Security

- **The API has no authentication by default.** Anyone who can reach the port can use the model, including tool calls and images.
- `BIND=127.0.0.1` publishes the port on the host's loopback only (inside the container the server listens on all interfaces). Keep it that way until a key is set.
- To serve other machines, either forward the port over SSH (`ssh -N -L 127.0.0.1:8011:127.0.0.1:8011 user@host`, as upstream recommends), or add a key and bind a LAN address:

  ```bash
  EXTRA_ARGS="--enable-prompt-tokens-details --api-key CHANGE_ME"
  BIND=<LAN address of the Docker host>
  ```

  Clients then send `Authorization: Bearer CHANGE_ME`.
- The key appears in the container's command line (`docker inspect`, `ps`). Anyone with access to the Docker host can read it.
- A firewall rule in front of the port is a good second layer.
- If you let other people use a server built from the upstream image over a network, AGPL-3.0 section 13 applies to the vllm-exl3 code in it; see [NOTICE.md](../NOTICE.md).

## 11. Troubleshooting notes

- **`serve.sh` stops at once with `bc: command not found`**: install `bc`. Upstream `serve.sh` uses it to check that the GPUs are idle.
- **Docker cannot resolve the CDI device `nvidia.com/gpu=all`, or the container sees fewer than four GPUs**: the CDI spec is missing or stale. `nvidia-ctk cdi list` should show all four GPUs; regenerate the spec with `nvidia-ctk cdi generate`.
- **Copy errors, dropped `_` in identifiers, or output that changes between runs at temperature 0**: check the startup log for `offloaded to pinned` ([9.1](#91-startup-log)). If the tables are loaded and errors remain, the next suspects we had prepared, never needed, were:
  - `DSV41_VARIANTS=` (empty: no runtime variants);
  - `--no-enable-prefix-caching`;
  - the original 2.9 hybrid.
- **The KV pool is smaller than expected**: another process holds GPU memory, or the graft profile is not `k3`. `k4x4` costs about 1.6 GiB per card (KV 1,670,368 at 0.97); DSpark costs 2.78 GiB per card.
- **The tmpfs is gone after a reboot**: expected. `./start.sh` re-creates it, or run `./engram-ram.sh` by hand.
