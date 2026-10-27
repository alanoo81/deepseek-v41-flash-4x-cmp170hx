#!/usr/bin/env python3
"""deepseek-v41-flash-4x-cmp170hx (MIT): memory / stability stress test for DeepSeek-V4.1-Flash, N concurrent
long-context requests.

Each request: a UNIQUE excerpt (random start + nonce, so no prefix-cache hits) of public-domain texts (the Gutenberg
books that upstream bench/prefill.py caches on its first run), cut to exactly PROMPT tokens with /tokenize, followed
by a question; MAXTOK output tokens, temperature 0, reasoning off. Checks: no HTTP error, non-empty answers, server
still alive, and records the peak VRAM per card during the test (nvidia-smi must see the cards).
Usage: stress-dsv41.py [--port 8011] [--conc 8] [--prompt 100000] [--maxtok 512] [--rounds 2] [--corpus-dir DIR]
The prompt and the printed messages are in French, as in our runs (data/logs/): "vague" = wave, "réponses" =
answers, "jetons de prompt" = prompt tokens, "générés" = generated, "serveur vivant" = server alive."""
import argparse, concurrent.futures as cf, json, random, subprocess, threading, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="8011"); ap.add_argument("--conc", type=int, default=8)
ap.add_argument("--prompt", type=int, default=100000); ap.add_argument("--maxtok", type=int, default=512)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--corpus-dir", default="bench/.cache",
                help="directory of Gutenberg *.txt files (upstream bench/prefill.py cache; run from the recipe root)")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
MODEL = "deepseek-v4.1-flash"

def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

cache = Path(a.corpus_dir)
corpus = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(cache.glob("*.txt")))
assert len(corpus) > 2_000_000, f"Gutenberg corpus missing in {cache} (run bench/prefill.py first)"
rng = random.Random(20261027)

def make_prompt(i):
    start = rng.randrange(0, len(corpus) - a.prompt * 6)
    text = f"[document {i}-{rng.random():.12f}]\n" + corpus[start:start + a.prompt * 6]
    ids = post("/tokenize", {"model": MODEL, "prompt": text})["tokens"][:a.prompt]
    return post("/detokenize", {"model": MODEL, "tokens": ids})["prompt"]

peak = {}
stop = threading.Event()
def sampler():
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            i, m = (int(x) for x in line.split(","))
            peak[i] = max(peak.get(i, 0), m)
        time.sleep(1)

def one(prompt):
    t0 = time.time()
    body = {"model": MODEL, "temperature": 0, "max_tokens": a.maxtok, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt + "\n\nRésume ce passage en cinq phrases, en français."}]}
    try:
        r = post("/v1/chat/completions", body)
        c = r["choices"][0]["message"].get("content") or ""
        return True, r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"], time.time() - t0, c[:80]
    except Exception as e:
        return False, 0, 0, time.time() - t0, repr(e)[:200]

threading.Thread(target=sampler, daemon=True).start()
for rd in range(a.rounds):
    prompts = [make_prompt(rd * 100 + i) for i in range(a.conc)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(one, prompts))
    ok = sum(r[0] and r[2] > 0 for r in res)
    print(f"vague {rd + 1} : {ok}/{a.conc} réponses OK, {sum(r[1] for r in res)} jetons de prompt, "
          f"{sum(r[2] for r in res)} générés, {time.time() - t0:.0f} s", flush=True)
    for r in res:
        if not (r[0] and r[2] > 0):
            print("   ÉCHEC :", r[4], flush=True)
stop.set(); time.sleep(1.5)
alive = urllib.request.urlopen(BASE + "/health", timeout=10).status == 200
print(f"serveur vivant après le test : {alive} ; VRAM max par carte (MiB) : {dict(sorted(peak.items()))}", flush=True)
