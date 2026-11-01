# Setup: the current configuration on Morrowmake 1.0.1

This page rebuilds what we serve: [Morrowmake's recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe)
1.0.1, with the routed experts of layers 18-22 grafted to 3 bits and the lossless compact KV cache. Every step
starts from Morrowmake's own pinned files, and each one is checked before the next. The previous
configuration (GNmac engine) is in [setup-gnmac.md](setup-gnmac.md).

## 1. What you need

Morrowmake's requirements ([its how-to-use](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe/blob/main/docs/how-to-use.md)):
4x CMP 170HX exposing 64 GiB each on PCIe Gen2 x16, NVIDIA driver 580 or newer, Docker able to see the four
cards, about 240 GiB of host RAM for the Engram tables, about 680 GiB of disk. Add **25 GiB** for the five
grafted shards.

What we ran on: ASRock Rack ROMED8-2T, EPYC 7F52, 512 GB DDR4-3200 (8 channels), NVIDIA open driver 610.43.02
with a cmpunlocker fork and a P2P patch (Morrowmake's P2P check passes without its TRAP31 patch), 200 W per
card, inside an LXC container.

## 2. Morrowmake 1.0.1 as released

```bash
git clone https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe.git morrowmake
cd morrowmake && git checkout v1.0.1
./install.sh       # the engine image, pinned by digest (ghcr.io/morrowmake/vllm-cmp170hx@sha256:32ae8b0e...)
./download.sh      # the EXL3 export @ c178a012 and the official checkpoint @ 1bae0e10, then the hybrid
```

**If your Docker has no NVIDIA runtime and uses CDI** (our case, Docker inside LXC), replace `--gpus all` with
`--device nvidia.com/gpu=all` in `start.sh`. It changes how the cards reach the container, nothing else.

Start it once as released (`./start.sh`, then `./start.sh smoke`) before changing anything: the first start
compiles the kernels into `./cache`.

## 3. Graft layers 18-22 to 3 bits

Morrowmake serves LibertAI's 2.9 bpw pack, whose layers 18-22 have 2-bit routed experts. LibertAI's 3.5 bpw
pack has 3-bit experts for those layers under the same tensor names.

```bash
# the five donor shards (about 24 GiB) and the donor index, at the pinned revision
hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw --revision 04a492fc3b4d25ddbab8f2d2f07f235ec77c9010 \
  --include 'model-0002[0-4]-of-00044.safetensors' 'model.safetensors.index.json' --local-dir donor-3.5

# graft into a hard-linked copy, so Morrowmake's own tree stays as released (no extra disk space);
# to graft in place instead, use models/ wherever models-k3/ appears below and skip MODELS_DIR
cp -al models models-k3

python3 /path/to/this-repo/morrowmake/tools/graft_layers.py \
  --export models-k3/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw --donor donor-3.5 --layers 18-22
```

The tool checks that both packs have exactly the same tensors for each layer and that the donor experts are
3-bit (read from the shard headers; it stops otherwise), links the donor shards into the export directory and
rewrites the export's index. The old index file is unlinked first, so a hard-linked original is not modified.

Then rebuild the hybrid checkpoint:

```bash
find models-k3/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8 -mindepth 1 -delete   # links and 3 small files, no weights
patch -p1 < /path/to/this-repo/morrowmake/prepare.sh.patch                     # the hybrid index hash of the graft
echo 'MODELS_DIR=./models-k3' >> .env      # set it once: when a key appears twice in .env, the first one wins
./download.sh      # rebuilds the hybrid and checks it against the patched hash
```

`prepare.sh` refuses a hybrid whose index does not have the hash `5d98a2d8…`. That hash holds for this exact
graft on Morrowmake's pinned export; any other layer set gives another hash, and you then edit the line
yourself.

## 4. The compact KV cache

```bash
/path/to/this-repo/morrowmake/compact-kv/build-overlay.sh .   # extracts 6 files from the image, patches them
patch -p1 < /path/to/this-repo/morrowmake/start.sh.patch       # mounts kvc-overlay/ over the image's files
```

If you replaced `--gpus all` for CDI in step 2, `start.sh.patch` still applies (with "fuzz 1": that line is
part of its context).

`build-overlay.sh` takes the six engine files the cache modifies from the pinned image itself, applies
`compact-kv.patch` and adds `sm80_kv_pack.py`. The image is not modified. Without `VLLM_SM80_COMPACT_KV=1` the
patched files behave exactly as the originals.

Run the GPU unit test before the first start. It needs about 1 GiB on one card, with the server stopped (with CDI, write `--device nvidia.com/gpu=all` instead of `--gpus all`):

```bash
IMG=$(grep -oP '^RELEASE_IMAGE=\K.*' start.sh)
M=(); for f in ops/sm80_kv_pack.py ops/sm80_attn_core.py ops/sm80_attn_pipeline.py ops/sm80_sparse_attn.py \
              ops/sm80_attn_fused.py ops/sm80_index_scores.py sm80_attention.py; do
  M+=(-v "$PWD/kvc-overlay/vllm_sm80/models/deepseek_v41/$f:/opt/assembly/plugin/vllm_sm80/models/deepseek_v41/$f:ro"); done
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 "${M[@]}" \
  -v /path/to/this-repo/morrowmake/compact-kv/tests:/t:ro --entrypoint /opt/venv/bin/python "$IMG" /t/test_gpu.py
```

Every line must say `IDENTICAL` and the last one `RESULT OK`. `bench_gpu.py`, in the same directory and run the
same way, times each kernel with BF16 and compact caches.

## 5. Settings and launch

Append [`morrowmake/env.example`](../morrowmake/env.example) to `.env` (it sets `VLLM_SM80_COMPACT_KV=1`, the
listen address, the served name and the API key) and set your own `API_KEY`. Then:

```bash
cp /path/to/this-repo/morrowmake/start-*.sh .
./start-dspark.sh                                   # what we serve: DSpark on, KV pool ~1.47 M tokens
./start-nospec.sh                                   # no DSpark: KV pool ~3.09 M tokens
MORE_ARGS="--max-model-len 1048576" ./start-nospec.sh   # contexts up to 1M (results.md, section 5)
```

At start, `logs/serve.log` must report `GPU KV cache size: 1,466,159 tokens` (DSpark) or `3,090,900 tokens`
(no DSpark), and the boot check must pass. Without the compact cache and without DSpark, the graft gives 1,555,036
tokens.

**Keep the memory setting at 0.95**, Morrowmake's default. At 0.98 our server ran out of memory on ordinary
requests.

## 6. Checking quality against our numbers

The frozen windows are not published, but every run file is, so you can compare runs of your own with each
other the same way:

```bash
python3 tests/nll_eval.py build ...      # see tests/nll_eval.py --help
NLL_API_KEY=<your key> python3 tests/nll_eval.py run --base-url http://127.0.0.1:8011 --model deepseek-v4.1-flash \
  --cache ... --label mine --out runs/mine.json
python3 tests/nll_eval.py compare A.json B.json
python3 tests/identical.py A.json B.json    # bit-level identity, e.g. BF16 cache against compact cache
```

The expected result of the compact cache against BF16 rows is `0 differ` on every set.

## 7. Security

- Morrowmake publishes the API on `127.0.0.1` by default. `env.example` sets `HOST=0.0.0.0` with an
  `API_KEY`: every `/v1` request then needs `Authorization: Bearer <key>`; `/health` stays open.
- Morrowmake's engine (its vLLM fork and the `vllm_sm80` plugin) is Apache-2.0, and so are the patched files of
  `compact-kv/`.
