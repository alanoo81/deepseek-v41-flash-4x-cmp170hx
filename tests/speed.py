#!/usr/bin/env python3
"""Decode and prefill speed per stream at short and long contexts (3k to 258k tokens), plus aggregate rate at 4 and
8 streams.

Safeguards (docs/agentic-methodology.md): a unique random prefix at the head of every prompt (prefix cache
defeated, hit rate checked on /metrics), discarded warm-up passes, no ignore_eos (natural output, at most 256
tokens), exact counts from usage.completion_tokens, staggered arrivals (0.7 s) under concurrency.
Per-stream decode = (generated tokens - 1) / (end - first token). The prompt is French technical prose.

The "aggregate" figures here are dominated by the 3.5k-token prefills of the concurrent requests and the
staggered arrivals: they are NOT comparable with the MiaAI-protocol aggregates (short prompts, simultaneous).
usage: speed.py --base http://127.0.0.1:8011 --model NAME --label X --out speed-X.json [--key K]"""
import argparse, json, random, statistics, string, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BODY = ("L'administration des systèmes informatiques repose sur quelques principes durables. "
        "Le premier est la reproductibilité : toute opération manuelle finit par être oubliée, "
        "puis perdue. Le deuxième est la vérification : un changement que l'on n'a pas mesuré "
        "n'est pas un changement, c'est une hypothèse. Le troisième est la réversibilité, car "
        "une modification que l'on ne sait pas annuler engage bien au-delà de son auteur. ")
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--key"); ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
H = {"Content-Type": "application/json", **({"Authorization": f"Bearer {a.key}"} if a.key else {})}


def prompt(rep):
    salt = "".join(random.choices(string.ascii_lowercase, k=16))
    return f"Note {salt}. " + BODY * rep + "\n\nRésumé détaillé des principes ci-dessus, point par point :\n1."


def one(rep, out=256):
    p = {"model": a.model, "prompt": prompt(rep), "max_tokens": out, "temperature": 0.0, "stream": True,
         "stream_options": {"include_usage": True}}
    r = urllib.request.Request(a.base + "/v1/completions", data=json.dumps(p).encode(), headers=H)
    t0 = time.perf_counter(); first = None; gen = 0; ptoks = 0
    for line in urllib.request.urlopen(r, timeout=3600):
        s = line.decode().strip()
        if s.startswith("data: ") and s != "data: [DONE]":
            d = json.loads(s[6:])
            if d.get("usage"):
                gen = d["usage"]["completion_tokens"]; ptoks = d["usage"]["prompt_tokens"]
            if d.get("choices") and first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    return dict(ttft=first - t0, gen=gen, prompt=ptoks, wall=end - t0,
                decode=(gen - 1) / (end - first) if gen > 1 and end > first else None)


def prefix_hits():
    m = urllib.request.urlopen(urllib.request.Request(a.base + "/metrics", headers=H), timeout=10).read().decode()
    v = {}
    for l in m.splitlines():
        for k in ("prefix_cache_hits_total", "prefix_cache_queries_total"):
            if l.startswith("vllm:" + k):
                v[k] = v.get(k, 0) + float(l.split()[-1])
    return v


res = {"label": a.label, "ts": time.time()}
h0 = prefix_hits()
w = one(40, 32)  # warm-up; also gives the tokens per repetition of the text
tpr = w["prompt"] / 40.5
# Context steps up to the server limit (262,144 tokens, output included); ~88 tokens per repetition.
# Above 32k: 2 measurements (a 256k request costs minutes of prefill), one warm-up per step.
for name, target, n in (("3k", 3500, 3), ("32k", 32768, 3), ("64k", 65536, 2), ("128k", 131072, 2),
                        ("256k", 258000, 2)):
    rep = max(1, int(target / tpr))
    one(rep, 32)  # warm-up at this length
    runs = [one(rep) for _ in range(n)]
    res[f"c1-{name}"] = dict(prompt_tokens=runs[0]["prompt"],
                            decode_median=statistics.median(r["decode"] for r in runs),
                            ttft_median=statistics.median(r["ttft"] for r in runs),
                            prefill_rate_median=statistics.median(r["prompt"] / r["ttft"] for r in runs),
                            gen=[r["gen"] for r in runs])
    x = res[f"c1-{name}"]
    print(f"{a.label} c1 {name:5s} ({x['prompt_tokens']} tokens): decode {x['decode_median']:.1f} tok/s, "
          f"TTFT {x['ttft_median']:.1f} s (prefill {x['prefill_rate_median']:.0f} tok/s), outputs {x['gen']}", flush=True)
for n in (4, 8):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        rs = list(ex.map(lambda i: (time.sleep(i * 0.7), one(40))[1], range(n)))
    wall = time.perf_counter() - t0
    res[f"c{n}"] = dict(aggregate=sum(r["gen"] for r in rs) / wall,
                        decode_per_stream_median=statistics.median(r["decode"] for r in rs if r["decode"]))
    print(f"{a.label} c{n}: aggregate {res[f'c{n}']['aggregate']:.1f} tok/s, median per-stream decode "
          f"{res[f'c{n}']['decode_per_stream_median']:.1f} tok/s", flush=True)
h1 = prefix_hits()
q = h1.get("prefix_cache_queries_total", 0) - h0.get("prefix_cache_queries_total", 0)
hit = (h1.get("prefix_cache_hits_total", 0) - h0.get("prefix_cache_hits_total", 0)) / q if q else 0
res["prefix_hit_rate"] = hit
print(f"{a.label} prefix cache hit rate during the run: {hit*100:.1f} % (must stay ~0)", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
