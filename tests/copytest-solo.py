#!/usr/bin/env python3
# deepseek-v41-flash-4x-cmp170hx (MIT): replays selected copy-test tasks one at a time (concurrency 1), twice, with
# the same random generation as copytest-dsv41.py (20 lists, default seed). Diagnostic only.
# Output messages are in French, like the logs in data/logs/.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["copytest-dsv41.py", "--n", "20"] + sys.argv[1:]   # e.g. --port 8000
src = open(os.path.join(HERE, "copytest-dsv41.py"), encoding="utf-8").read().split("for kind, label in")[0]
g = {}; exec(compile(src, "copytest", "exec"), g)                  # definitions + task generation only
idx = [0, 2, 10, 17, 19, 1, 3]                                      # the task indices replayed in our diagnosis
for rep in (1, 2):
    for i in idx:
        out, u = g["chat"]([{"role": "user", "content": g["INSTR"] + "\n".join(g["tasks"][i][0])}])
        ok, bad = g["score"](g["tasks"][i][0], out)
        print(f"essai {rep} tâche {i:2d} (seule) : {'OK' if ok else f'{bad} lignes fausses'} — {out.splitlines()[0][:60] if out else ''!r}", flush=True)
