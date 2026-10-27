#!/usr/bin/env python3
# deepseek-v41-flash-4x-cmp170hx (MIT): determinism at temperature 0. 5 tasks x 5 repetitions, sent one at a time,
# outputs compared byte for byte. Reuses the task generator of copytest-dsv41.py (seed 4242).
# Output messages are in French, like the logs in data/logs/ ("sortie(s) distincte(s)" = distinct output(s)).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["copytest-dsv41.py", "--n", "5", "--seed", "4242"] + sys.argv[1:]   # e.g. --port 8000
src = open(os.path.join(HERE, "copytest-dsv41.py"), encoding="utf-8").read().split("for kind, label in")[0]
g = {}; exec(compile(src, "copytest", "exec"), g)
for i, (l1, _) in enumerate(g["tasks"]):
    outs = [g["chat"]([{"role": "user", "content": g["INSTR"] + "\n".join(l1)}])[0] for _ in range(5)]
    oks = [g["score"](l1, o)[0] for o in outs]
    print(f"tâche {i} : {sum(oks)}/5 exactes, {len(set(outs))} sortie(s) distincte(s) sur 5", flush=True)
