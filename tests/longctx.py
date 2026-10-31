#!/usr/bin/env python3
"""Exact recall at very long contexts: 8 random access codes hidden from 5 % to 95 % depth in a haystack of
public-domain books (Project Gutenberg plain-text files), then a final question; score = codes returned exactly.
Also measures the time to the first token and the decode rate. Temperature 0, thinking off.
The question and the needles are in French, like most of the books we used.
usage: longctx.py --base URL --model NAME [--key K] --books 'DIR/*.txt' --sizes 131072,524288,1000000 --out x.json"""
import argparse, glob, json, random, re, string, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--key", default="")
ap.add_argument("--sizes", default="131072,524288,1000000")
ap.add_argument("--needles", type=int, default=8)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--books", required=True, help="glob of plain-text books")
ap.add_argument("--out", required=True)
a = ap.parse_args()
H = {"Content-Type": "application/json", **({"Authorization": f"Bearer {a.key}"} if a.key else {})}


def post(path, body, timeout=7200):
    r = urllib.request.Request(a.base + path, json.dumps(body).encode(), H)
    return urllib.request.urlopen(r, timeout=timeout)


def ntok(text):
    return json.load(post("/tokenize", {"model": a.model, "prompt": text}))["count"]


def book(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", t)
    e = re.search(r"\*\*\* ?END OF", t)
    t = t[m.end() if m else 0: e.start() if e else len(t)]
    return re.sub(r"\n{3,}", "\n\n", t).strip()


books = [book(p) for p in sorted(glob.glob(a.books))]
QUESTION = ("\n\n---\nFin du document. Question : le document ci-dessus contient {n} phrases de la forme "
            "« Le code d'accès du coffre numéro K est ... ». Donne les {n} codes, une ligne par coffre, au format "
            "exact « coffre K : CODE », du coffre 1 au coffre {n}, et rien d'autre.")
results = []
for size in [int(s) for s in a.sizes.split(",")]:
    rng = random.Random(a.seed * 1000003 + size)
    order = books[:]
    rng.shuffle(order)
    text = f"[document {rng.getrandbits(64):016x}]\n\n" + "\n\n".join(order)
    # 3 to 4 characters per token on these books: start too long, the real count only shortens it
    chars = int(size * 5)
    while len(text) < chars:
        text += "\n\n" + "\n\n".join(order)
    text = text[:chars]
    for _ in range(4):
        n = ntok(text)
        if abs(n - size) < size * 0.01:
            break
        text = text[: int(len(text) * size / n)]
    codes = ["".join(rng.choices(string.ascii_uppercase, k=3)) + "-" + "".join(rng.choices(string.digits, k=4))
             + "-" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=3)) for _ in range(a.needles)]
    depths = [0.05 + 0.9 * i / (a.needles - 1) for i in range(a.needles)]
    pieces, last = [], 0
    for k, (d, c) in enumerate(zip(depths, codes), 1):
        cut = text.rfind("\n", 0, int(len(text) * d)) + 1 or int(len(text) * d)
        pieces += [text[last:cut], f"Le code d'accès du coffre numéro {k} est {c}.\n"]
        last = cut
    prompt = "".join(pieces) + text[last:] + QUESTION.format(n=a.needles)
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": 200, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    t0 = time.time(); first = None; out = ""; usage = {}
    try:
        with post("/v1/chat/completions", body) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    piece = ch.get("delta", {}).get("content") or ""
                    if first is None and ch.get("delta"):
                        first = time.time()
                    out += piece
        end = time.time(); err = None
    except Exception as e:  # noqa: BLE001 — keep the trace and go on with the next size
        end = time.time(); err = repr(e)[:300]
    got = dict(re.findall(r"coffre\s*(\d+)\s*:\s*([A-Z0-9-]+)", out))
    hits = [got.get(str(k)) == c for k, c in enumerate(codes, 1)]
    gen = usage.get("completion_tokens", 0)
    res = dict(size=size, prompt_tokens=usage.get("prompt_tokens"), ttft=first and round(first - t0, 1),
               decode_tps=round((gen - 1) / (end - first), 1) if first and gen > 1 else None,
               hits=sum(hits), needles=a.needles, per_depth=[[round(d, 3), h] for d, h in zip(depths, hits)],
               answer=out[:600], error=err)
    results.append(res)
    print(f"{size:>8} tokens (prompt {res['prompt_tokens']}): recall {res['hits']}/{a.needles}, "
          f"TTFT {res['ttft']} s, decode {res['decode_tps']} tok/s" + (f", ERROR {err}" if err else ""), flush=True)
json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
