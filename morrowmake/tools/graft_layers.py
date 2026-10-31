#!/usr/bin/env python3
"""Graft the routed experts of some layers from one EXL3 pack into Morrowmake's export checkpoint.

Morrowmake 1.0.0 serves LibertAI's 2.9 bpw pack (`DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`, 43 shards), whose
layers 18-22 have 2-bit routed experts. LibertAI's 3.5 bpw pack has 3-bit experts for those layers, under the
same tensor names, in shards `model-00020..00024-of-00044.safetensors`. This tool:

1. checks that the donor pack has exactly the export's routed-expert tensors for each grafted layer, and that
   their bit width, read from the trellis shape in the shard headers (the last dimension is 16 x bits), is
   `--bits` (3 by default);
2. links (or copies) the donor shards into the export directory;
3. rewrites the export's `model.safetensors.index.json` so those tensors point at the donor shards. The old
   index is unlinked first, so a hard-linked original elsewhere is not modified.

Then empty the hybrid directory (it holds links and three small files, no weights) and let Morrowmake's
`prepare.sh` rebuild it, with the hybrid index hash of `prepare.sh.patch` for the 18-22 graft.

usage: graft_layers.py --export MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw --donor DIR_WITH_3.5_SHARDS \
                       --layers 18-22 [--copy]
"""
import argparse
import json
import os
import re
import shutil
import struct


def layer_list(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


ap = argparse.ArgumentParser()
ap.add_argument("--export", required=True, help="Morrowmake's export directory (EXL3 2.9 bpw pack)")
ap.add_argument("--donor", required=True, help="directory with the donor shards and the donor index")
ap.add_argument("--layers", required=True, help="e.g. 18-22")
ap.add_argument("--bits", type=int, default=3, help="bit width the donor experts must have (default 3)")
ap.add_argument("--copy", action="store_true", help="copy the donor shards instead of hard-linking them")
a = ap.parse_args()

exp_index_path = os.path.join(a.export, "model.safetensors.index.json")
exp_index = json.load(open(exp_index_path))
donor_map = json.load(open(os.path.join(a.donor, "model.safetensors.index.json")))["weight_map"]
weight_map = exp_index["weight_map"]

moved = 0
for layer in layer_list(a.layers):
    pat = re.compile(rf"^layers\.{layer}\.ffn\.experts\.")
    mine = {k for k in weight_map if pat.match(k)}
    theirs = {k for k in donor_map if pat.match(k)}
    if not mine or mine != theirs:
        raise SystemExit(f"layer {layer}: the two packs do not have the same routed-expert tensors "
                         f"({len(mine)} vs {len(theirs)})")
    shards = sorted({donor_map[k] for k in mine})
    bits = set()
    for shard in shards:
        h = header(os.path.join(a.donor, shard))
        bits |= {h[k]["shape"][-1] // 16 for k in mine if k.endswith(".trellis") and k in h}
    if bits != {a.bits}:
        raise SystemExit(f"layer {layer}: donor experts are {sorted(bits) or 'unreadable'} bits, expected {a.bits}")
    for shard in shards:
        src = os.path.join(a.donor, shard)
        dst = os.path.join(a.export, shard)
        if not os.path.exists(dst):
            (shutil.copy2 if a.copy else os.link)(src, dst)
    for k in mine:
        weight_map[k] = donor_map[k]
    moved += len(mine)
    print(f"layer {layer}: {len(mine)} tensors -> {', '.join(shards)} ({'/'.join(map(str, sorted(bits)))} bits)")

tmp = exp_index_path + ".tmp"
with open(tmp, "w") as f:
    json.dump(exp_index, f, indent=2)
os.unlink(exp_index_path)
os.replace(tmp, exp_index_path)
print(f"{moved} tensors remapped in {exp_index_path}")
print("next: empty the hybrid directory, apply prepare.sh.patch, run ./download.sh")
