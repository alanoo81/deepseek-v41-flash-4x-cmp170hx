#!/usr/bin/env python3
"""deepseek-v41-flash-4x-cmp170hx (MIT): exact-copy test for DeepSeek-V4.1-Flash. Written to probe a suspected bug
(prefix cache => corrupted Engram n-gram history, cf. llama.cpp PR #30817 and GNmac issue #1); it also catches a
model running with missing weights.

Pass 1: N unique prompts (nothing cached). Pass 2: the same prompts (prefix-cache reuse).
Pass 3: a second conversation turn on a cached prefix (turn 1 + its exact answer + a new list to copy).
Each task: copy a list of 40 random identifiers exactly. Temperature 0, reasoning off.
Score: exact copy (yes/no) and number of wrong lines. Usage: copytest-dsv41.py [--port 8011] [--conc 8] [--n 20]
[--seed 20261027]. The prompt and the printed messages are in French, as in our runs (data/logs/):
"passe" = pass, "copies exactes" = exact copies, "lignes fausses" = wrong lines, "jetons" = tokens."""
import argparse, concurrent.futures as cf, json, random, string, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="8011"); ap.add_argument("--conc", type=int, default=8)
ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=20261027)
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"
rng = random.Random(a.seed)

def ident():
    w = lambda: "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8)))
    return f"{w()}_{w()}_{rng.randint(0, 99999):05d}_{''.join(rng.choice('0123456789abcdef') for _ in range(8))}"

def lst(n=40): return [ident() for _ in range(n)]

INSTR = ("Recopie EXACTEMENT la liste suivante, une entrée par ligne, dans le même ordre, sans numéro, sans "
         "commentaire, sans bloc de code, rien d'autre.\n\n")

def chat(messages):
    body = {"model": "deepseek-v4.1-flash", "messages": messages, "temperature": 0, "max_tokens": 2000,
            "reasoning_effort": "none"}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["choices"][0]["message"].get("content") or "", r["usage"]

def score(expected, got):
    lines = [l.strip() for l in got.strip().splitlines() if l.strip()]
    bad = sum(1 for i, e in enumerate(expected) if i >= len(lines) or lines[i] != e) + max(0, len(lines) - len(expected))
    return bad == 0, bad

tasks = [(lst(), lst()) for _ in range(a.n)]

def run(kind, t):
    l1, l2 = t
    m1 = [{"role": "user", "content": INSTR + "\n".join(l1)}]
    if kind in ("p1", "p2"):
        out, u = chat(m1); ok, bad = score(l1, out)
    else:  # 2nd turn: turn 1 (already cached) + its exact answer, then a new list
        m2 = m1 + [{"role": "assistant", "content": "\n".join(l1)},
                   {"role": "user", "content": INSTR + "\n".join(l2)}]
        out, u = chat(m2); ok, bad = score(l2, out)
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return ok, bad, u["prompt_tokens"], cached, out

for kind, label in (("p1", "passe 1 (à froid)"), ("p2", "passe 2 (même prompt, cache de préfixes)"),
                    ("p3", "passe 3 (2e tour sur préfixe en cache)")):
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda t: run(kind, t), tasks))
    oks = sum(r[0] for r in res); bad = sum(r[1] for r in res)
    pt = sum(r[2] for r in res); ct = sum(r[3] for r in res)
    print(f"{label} : {oks}/{len(res)} copies exactes, {bad} lignes fausses, "
          f"cache {ct}/{pt} jetons de prompt ({100 * ct / max(pt, 1):.0f} %), {time.time() - t0:.0f} s", flush=True)
    for i, r in enumerate(res):
        if not r[0]:
            print(f"   tâche {i} : {r[1]} lignes fausses — début de sortie : {r[4][:160]!r}", flush=True)
