#!/usr/bin/env python3
"""Build the GNmac hybrid, then graft the 3-bit routed experts of layers 18-22 (LibertAI 3.5 bpw pack) in place of
the 2-bit ones of the 2.9 bpw pack. Everything else is unchanged (3-bit experts of the other layers, official FP8 for
all other tensors). Nothing is copied: relative symlinks, like make_hybrid.py.

Copyright (c) 2026 alanoo81, MIT licence. This script is our own; it calls
the upstream tools/make_hybrid.py of GNmac-lab/deepseek-v41-flash-1017-cmp170hx @ 2ad5741ab1 (Copyright (c) 2026
GNmac-lab, MIT), which it does not include, and must sit next to it in tools/. The hybrid layout and
quantization_config.json it uses are Curtis White's (CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX).

Usage: make_hybrid_graft.py MODELS_DIR [OUT_NAME] [PROFILE]     (PROFILE: k3 = production, k4x4 = experiment)
  - first builds MODELS_DIR/<OUT_NAME> with tools/make_hybrid.py (2.9 bpw pack, Curtis White's configuration);
  - then rewrites the index for the expert tensors of the grafted layers and links the 3.5 bpw shards.
Checks: same expert tensor names in both packs for those layers, one shard per layer holding no other layer's
experts, the bit width K read from the safetensors header, and every referenced file exists."""
import json
import os
import re
import struct
import subprocess
import sys

M = sys.argv[1]
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "DeepSeek-V4.1-Flash-1017-hybrid"
PROFILE = sys.argv[3] if len(sys.argv) > 3 else "k3"
OUT = os.path.join(M, OUT_NAME)
LIB = "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
# layer -> (source directory under MODELS_DIR, expected K = bits per weight of the routed experts).
# K is read back from the safetensors header (last trellis dimension / 16), not only from exl3_k_map.json.
L35 = "DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-L18-22"
L35K4 = "DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-K4"   # k4x4 only: shards 00002/00003/00005/00008-of-00044, fetched by hand
PROFILES = {
    "k3": {L: (L35, 3) for L in (18, 19, 20, 21, 22)},                         # every layer at 3 bits
    "k4x4": {**{L: (L35, 3) for L in (18, 19, 20, 21, 22)},                     # + the most sensitive layers at 4 bits
             **{L: (L35K4, 4) for L in (0, 1, 6, 3)}},                          #   (alpha sensitivity, 1017 checkpoint)
}
GRAFT = PROFILES[PROFILE]
GRAFT_LAYERS = tuple(sorted(GRAFT))


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))
here = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(OUT) and os.listdir(OUT):
    sys.exit(f"{OUT} is not empty: empty it first (make_hybrid.py does not remove old links)")
subprocess.run([sys.executable, os.path.join(here, "make_hybrid.py"), M, os.path.join(M, "quantization_config.json"),
                LIB, OUT_NAME], check=True)

idx_path = os.path.join(OUT, "model.safetensors.index.json")
idx = json.load(open(idx_path))
wm = idx["weight_map"]
src_maps = {d: json.load(open(os.path.join(M, d, "model.safetensors.index.json")))["weight_map"]
            for d in set(v[0] for v in GRAFT.values())}

grafted, old_files, new_files = 0, set(), set()
for layer in GRAFT_LAYERS:
    pat = re.compile(rf"^layers\.{layer}\.ffn\.experts\.")
    src, kexp = GRAFT[layer]
    old = {k: v for k, v in wm.items() if pat.match(k)}
    new = {k: v for k, v in src_maps[src].items() if pat.match(k)}
    assert old and set(old) == set(new), f"layer {layer}: expert tensor sets differ between the packs"
    files = set(new.values())
    assert len(files) == 1, f"layer {layer}: experts spread over several shards {files}"
    fname = next(iter(files))
    fpath = os.path.join(M, src, fname)
    assert os.path.isfile(fpath), f"missing shard: {src}/{fname}"
    h = header(fpath)
    exp_keys = [k for k in h if re.match(r"^layers\.\d+\.ffn\.experts\.", k)]
    foreign = {re.match(r"^layers\.(\d+)\.", k).group(1) for k in exp_keys} - {str(layer)}
    assert not foreign, f"{fname} also holds the experts of layers {foreign}"
    ks = {h[k]["shape"][-1] // 16 for k in exp_keys if k.endswith(".trellis")}
    assert ks == {kexp}, f"layer {layer}: K read from the header {ks}, expected {kexp}"
    old_files |= set(old.values())
    new_files |= {(src, fname)}
    wm.update(new)
    grafted += len(new)
    print(f"layer {layer}: {len(new)} tensors -> {src}/{fname} (K={kexp}, checked in the header)")

for src, f in sorted(new_files):
    os.symlink(os.path.join("..", src, f), os.path.join(OUT, f))
still_used = set(wm.values())
for f in sorted(old_files - still_used):     # 2-bit shards of the 2.9 pack that this view no longer uses
    os.unlink(os.path.join(OUT, f))
idx["metadata"]["dsv41_hybrid"] += " ; graft profile " + PROFILE + ": " + ", ".join(
    f"L{L}=K{GRAFT[L][1]}" for L in GRAFT_LAYERS) + " from LibertAI EXL3 3.5 bpw @04a492fc"
json.dump(idx, open(idx_path, "w"), indent=1)
missing = [f for f in still_used if not os.path.exists(os.path.join(OUT, f))]
assert not missing, f"broken links: {missing[:5]}"
print(f"{OUT}: {grafted} tensors grafted, {len(still_used)} shards referenced, no broken link")
