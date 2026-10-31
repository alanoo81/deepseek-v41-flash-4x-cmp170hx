#!/usr/bin/env python3
"""Bit-level comparison of two runs of tests/nll_eval.py (plain or .gz JSON).

- Two `run` files (same frozen windows): compares every prompt logprob, token by token.
- Two `selfcheck` files: compares the logprobs of the generated (greedy) tokens.

Prints the number of values compared, how many differ and the largest difference.
usage: identical.py A.json[.gz] B.json[.gz]
"""
import gzip
import json
import sys


def load(path):
    raw = open(path, "rb").read()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def pairs(a, b):
    if "items" in a:  # selfcheck
        other = {i["id"]: i for i in b["items"]}
        for item in a["items"]:
            yield item["decode_logprobs"], other[item["id"]]["decode_logprobs"]
        return
    for name, seqs in a["sets"].items():
        other = {s["id"]: s for s in b["sets"][name]["seqs"]}
        for s in seqs["seqs"]:
            yield s.get("logprobs", []), other[s["id"]].get("logprobs", [])


a, b = load(sys.argv[1]), load(sys.argv[2])
total = differ = 0
worst = 0.0
for x, y in pairs(a, b):
    total += max(len(x), len(y))
    differ += abs(len(x) - len(y))
    for p, q in zip(x, y):
        if p != q:
            differ += 1
            worst = max(worst, abs(p - q))
verdict = "IDENTICAL" if differ == 0 and total else "DIFFERENT"
print(f"{sys.argv[1]} vs {sys.argv[2]}: {total} values, {differ} differ, max |d| {worst:.3g} -> {verdict}")
