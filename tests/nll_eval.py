#!/usr/bin/env python3
"""Paired held-out NLL / perplexity evaluation through a running vLLM OpenAI server (python3 stdlib only).

Built for DeepSeek-V4.1-Flash-1017 (GNmac stack, served name 'deepseek-v4.1-flash', --tokenizer-mode deepseek_v41)
to compare weight configurations (all-K3 vs some-K4 routed-expert layers, or the 2.9 pack with K2 layers 18-22) on
IDENTICAL token windows. The model is scored by teacher forcing: /v1/completions with a token-id prompt,
max_tokens=1 and prompt_logprobs=k. vLLM computes the prompt logprobs in the prefill path, and a request that asks
for prompt_logprobs skips reading the prefix cache (vllm/sampling_params.py:539-543, kv_cache_manager.py:213-215,249).

Subcommands
  build      fetch + freeze the corpora: download (Gutenberg) or read (local code), clean, cut fixed windows of
             exactly --seq-len tokens with the SERVER's tokenizer (/tokenize), and save token ids. Done once; every
             later run reads the frozen ids, so all configurations see the same tokens.
  run        score every frozen window with one config and save per-sequence (and per-token) results as JSON.
  compare    paired statistics between two run files (A = baseline, B = candidate): mean per-sequence NLL
             difference, bootstrap + t confidence intervals, sign test, Wilcoxon signed-rank, delta-PPL %, top-1
             accuracy; with --topk runs also a KL(A||B) lower bound and argmax agreement.
  selfcheck  decode-path vs prefill-path consistency (greedy generation with logprobs, then teacher-forced
             rescoring of the same tokens) + a causality canary (random token block repeated once). prompt_logprobs
             only exercises the prefill MoE path; this is the check that also touches the decode path.
  info       print a summary of the frozen corpora.

Typical use (on the Docker host, server bound to 127.0.0.1:<port>): an A-B-A design across restarts
  ./nll_eval.py build     --base-url http://127.0.0.1:8011 --cache ./nllcache                    # 1024 x 64 per set
  ./nll_eval.py build     --base-url ... --cache ./nllshort --seq-len 40 --n-seqs 400             # coop-path set
  ./nll_eval.py run       --base-url ... --cache ./nllcache --label k3 --out runs/k3.json
  ./nll_eval.py run       --base-url ... --cache ./nllshort --seq-len 40 --label k3 --out runs/k3-short.json
  ./nll_eval.py selfcheck --base-url ... --cache ./nllcache --label k3 --out runs/sc-k3.json
  (restart the server with the candidate weights; same env except the weights)
  ... the same three commands with --label k4 ...
  (restart again on the baseline weights)
  ... the same three commands with --label k3bis ...
  ./nll_eval.py compare   runs/k3.json runs/k3bis.json          # NULL: cross-restart noise, must be ~0
  ./nll_eval.py compare   runs/k3.json runs/k4.json             # effect; also k3bis vs k4
Why A-B-A: two runs in one server process are bit-exact (GNmac docs/results.md:76-78), so an in-process A/A pair
only proves determinism and says nothing about the noise a restart or a kernel change adds (TP4 is not bit-
reproducible across processes, docs/troubleshooting.md:14; a 1-ulp change moves short-context logprobs by nats).
Window length decides which MoE kernel is scored: prompts of <= 42 tokens (BOS + 41) run the routed experts through
the decode-shaped kernels (coop for 4-bit layers, exl3.py tokens*topk <= 256); longer prompts run the prefill
kernels. The default 1,024-token windows fit one 2,048-token prefill chunk and keep the prompt-logprob logits spike
to ~1.4 GiB/GPU (2,048 rows: ~2.7 GiB, not covered by vLLM's startup memory profile).
Run nothing else on the server during a run (batch composition changes the numerics), keep the server env identical
between configs except the weights (e.g. VLLM_EXL3_FAT_THRESHOLD), and keep --seq-len, --skip, --bos-id identical
between the runs you compare (compare refuses mismatches).
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

VERSION = "nll_eval/1.1"

# ---------------------------------------------------------------------------------------------------- corpora
# Project Gutenberg plain-text files (public domain in the US; Gutenberg licence for the wrapper text, which is
# stripped). Titles/languages verified against the file headers on 2026-10-26; build() re-checks 'Language:'.
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
FR_BOOKS = [
    (14155, "Madame Bovary (Flaubert)"),
    (5711, "Germinal (Zola)"),
    (2650, "Du cote de chez Swann (Proust)"),
    (5097, "Vingt mille lieues sous les mers (Verne)"),
    (13951, "Les trois mousquetaires (Dumas)"),
    (798, "Le rouge et le noir (Stendhal)"),
    (17989, "Le comte de Monte-Cristo, Tome I (Dumas)"),
    (17489, "Les miserables, Tome I (Hugo)"),
]
EN_BOOKS = [
    (145, "Middlemarch (Eliot)"),
    (4276, "North and South (Gaskell)"),
    (6593, "Tom Jones (Fielding)"),
    (8492, "The King in Yellow (Chambers)"),
    (3207, "Leviathan (Hobbes)"),
    (768, "Wuthering Heights (Bronte)"),
    (174, "The Picture of Dorian Gray (Wilde)"),
    (2600, "War and Peace (Tolstoy, tr. Maude)"),
]
# Code: the GNmac recipe (MIT, commit 2ad5741ab1, 2026-10-26: written after the model's release, so it cannot be
# in its training data). Whitelist only; files that this repository patches or adds are excluded so the set is the
# upstream text. Never add a pattern that could match .env (credentials).
CODE_INCLUDE = ["patches/**/*.py", "patches/**/*.cu", "patches/**/*.cuh", "bench/*.py", "bench/*.cu",
                "tools/*.py", "docker/Dockerfile", "start.sh", "stop.sh"]
CODE_EXCLUDE = {"serve.sh", "serve.sh.orig", "engram-ram.sh", "tools/make_hybrid_graft.py",
                "tools/fetch_official_nonexpert.py", "tools/download-pinned.sh"}
CODE_DIR_CANDIDATES = ["/opt/dsv41-gnmac"]   # where our recipe copy lived; otherwise pass --code-dir
PROSE_EXT = {".txt", ".md", ".rst"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def sha256(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def ids_sha(ids):
    return sha256(",".join(map(str, ids)))


# ---------------------------------------------------------------------------------------------------- HTTP
class Api:
    def __init__(self, base, model, timeout=1800):
        self.base, self.model, self.timeout = base.rstrip("/"), model, timeout

    def _req(self, path, body=None, tries=3):
        data = None if body is None else json.dumps(body).encode()
        for i in range(tries):
            req = urllib.request.Request(self.base + path, data, {"Content-Type": "application/json",
                                         **({"Authorization": "Bearer " + os.environ["NLL_API_KEY"]} if os.environ.get("NLL_API_KEY") else {})})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:2000]
                if e.code < 500 or i == tries - 1:
                    raise SystemExit(f"HTTP {e.code} on {path}: {msg}")
                log(f"  HTTP {e.code} on {path}, retry {i + 1}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if i == tries - 1:
                    raise SystemExit(f"cannot reach {self.base}{path}: {e}")
                log(f"  {e} on {path}, retry {i + 1}")
            time.sleep(5 * (i + 1))

    def get(self, path):
        return self._req(path)

    def post(self, path, body):
        return self._req(path, body)

    def tokenize(self, text):
        r = self.post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})
        return r["tokens"]

    def detokenize(self, ids):
        return self.post("/detokenize", {"model": self.model, "tokens": ids})["prompt"]

    def server_info(self):
        info = {"base_url": self.base}
        models = self.get("/v1/models")
        info["models"] = [{k: m.get(k) for k in ("id", "root", "max_model_len")} for m in models.get("data", [])]
        if self.model not in [m["id"] for m in info["models"]]:
            raise SystemExit(f"model {self.model!r} not served here: {info['models']}")
        try:
            info["version"] = self.get("/version")
        except SystemExit:
            info["version"] = None
        return info


def fetch_url(url, dest):
    """Download once into the cache; later builds reuse the cached bytes (Gutenberg files get re-edited)."""
    if dest.exists():
        return dest.read_bytes()
    log(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(req, timeout=300) as r:
        b = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(b)
    tmp.replace(dest)
    return b


# ---------------------------------------------------------------------------------------------------- cleaning
def gutenberg_clean(raw, want_lang):
    t = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    s = re.search(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK.*$", t, re.M | re.I)
    e = re.search(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK", t, re.M | re.I)
    if not s or not e:
        raise SystemExit("Gutenberg START/END markers not found")
    m = re.search(r"^Language:\s*(.+)$", t[:s.start()], re.M)
    lang = m.group(1).strip() if m else "?"
    if want_lang and not lang.lower().startswith(want_lang.lower()):
        raise SystemExit(f"language is {lang!r}, expected {want_lang!r}")
    title = re.search(r"^Title:\s*(.+)$", t[:s.start()], re.M)
    body = t[s.end():e.start()]
    # Unwrap the ~70-column hard wrap: one line per paragraph, paragraphs separated by a blank line.
    paras = []
    for p in re.split(r"\n[ \t]*\n+", body):
        p = re.sub(r"[ \t]+", " ", " ".join(x.strip() for x in p.split("\n"))).strip()
        if p and not re.fullmatch(r"\[?Illustration[^\]]*\]?", p, re.I):
            paras.append(p)
    return "\n\n".join(paras), {"language": lang, "title": title.group(1).strip() if title else "?",
                                "replacement_chars": t.count("�")}


def code_stream(code_dir):
    root = Path(code_dir)
    files = set()
    for pat in CODE_INCLUDE:
        for f in root.glob(pat):
            rel = f.relative_to(root).as_posix()
            if f.is_file() and rel not in CODE_EXCLUDE and not f.name.startswith(".") and "__pycache__" not in rel \
                    and not rel.endswith((".orig", ".env")):
                files.add(rel)
    parts, manifest, seen = [], [], set()
    for rel in sorted(files):
        b = (root / rel).read_bytes()
        h = sha256(b)
        if h in seen:
            continue
        seen.add(h)
        try:
            txt = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        parts.append(f"# ===== {rel} =====\n" + txt.replace("\r\n", "\n").rstrip("\n") + "\n")
        manifest.append({"path": rel, "sha256": h, "bytes": len(b)})
    if not parts:
        raise SystemExit(f"no code files under {code_dir}")
    return "\n".join(parts), manifest


# ---------------------------------------------------------------------------------------------------- windows
def cut_windows(api, text, n, seq_len, kind, lo=0.05, hi=0.95):
    """n windows of exactly seq_len tokens, at evenly spaced char positions in [lo, hi] of the text, each starting
    at a paragraph (prose) or line (code) boundary. Deterministic for a given text + tokenizer."""
    out, L = [], len(text)
    sep = "\n\n" if kind == "prose" else "\n"
    prev_end = 0
    for k in range(n):
        p = max(int(L * (lo + (hi - lo) * (k + 0.5) / n)), prev_end)   # never start inside the previous window
        j = text.find(sep, p)
        if j < 0:
            raise SystemExit(f"window {k}: text too short for {n} windows")
        st = j + len(sep)
        chars = seq_len * (6 if kind == "prose" else 5)
        while True:
            excerpt = text[st:st + chars]
            ids = api.tokenize(excerpt)
            if len(ids) >= seq_len + 8 or st + chars >= L:
                break
            chars *= 2
        if len(ids) < seq_len:
            raise SystemExit(f"window {k}: only {len(ids)} tokens left at char {st}: use fewer windows")
        ids = ids[:seq_len]
        dt = api.detokenize(ids)
        # exact char span when the detokenized ids are a prefix of the excerpt (lossless byte-level BPE);
        # otherwise fall back to a conservative bound so the next window still cannot overlap
        end = st + len(dt) if excerpt.startswith(dt) else st + len(excerpt)
        prev_end = end
        out.append({"char_start": st, "char_end": end, "ids": ids, "ids_sha256": ids_sha(ids), "text": dt})
    return out


def build_set(api, a, name, cache):
    path = cache / f"corpus-{name}.json"
    if path.exists() and not a.rebuild:
        log(f"[{name}] frozen corpus exists: {path} (use --rebuild to redo)")
        return
    seqs, sources = [], []
    if name in ("fr", "en"):
        books = FR_BOOKS if name == "fr" else EN_BOOKS
        want = "French" if name == "fr" else "English"
        per = [a.n_seqs // len(books) + (1 if i < a.n_seqs % len(books) else 0) for i in range(len(books))]
        for (bid, label), w in zip(books, per):
            url = GUTENBERG_URL.format(id=bid)
            raw = fetch_url(url, cache / "raw" / f"pg{bid}.txt")
            text, meta = gutenberg_clean(raw, want)
            sources.append({"key": f"pg{bid}", "url": url, "raw_sha256": sha256(raw), "clean_sha256": sha256(text),
                            "label": label, **meta, "license": "public domain (Project Gutenberg)"})
            log(f"[{name}] pg{bid} {meta['title']!r} {len(text)} chars -> {w} windows")
            for k, win in enumerate(cut_windows(api, text, w, a.seq_len, "prose")):
                seqs.append({"id": f"{name}/pg{bid}/{k}", "source": f"pg{bid}", **win})
    elif name == "code":
        code_dir = a.code_dir or next((d for d in CODE_DIR_CANDIDATES if Path(d).is_dir()), None)
        if not code_dir:
            raise SystemExit("no GNmac recipe dir found; pass --code-dir")
        text, manifest = code_stream(code_dir)
        sources.append({"key": "gnmac", "dir": str(code_dir), "files": manifest, "clean_sha256": sha256(text),
                        "license": "MIT (GNmac-lab, deepseek-v41-flash-1017-cmp170hx @ 2ad5741ab1)"})
        log(f"[{name}] {len(manifest)} files, {len(text)} chars -> {a.n_seqs} windows")
        for k, win in enumerate(cut_windows(api, text, a.n_seqs, a.seq_len, "code")):
            seqs.append({"id": f"{name}/gnmac/{k}", "source": "gnmac", **win})
    elif name in a.local_specs:
        files = sorted(set(Path("/").glob(a.local_specs[name].lstrip("/"))) if a.local_specs[name].startswith("/")
                       else set(Path(".").glob(a.local_specs[name])))
        files = [f for f in files if f.is_file() and not f.name.startswith(".env")]
        if not files:
            raise SystemExit(f"--local {name}: no files match {a.local_specs[name]}")
        kind = "prose" if all(f.suffix in PROSE_EXT for f in files) else "code"
        texts, manifest = [], []
        for f in files:
            b = f.read_bytes()
            texts.append(b.decode("utf-8", errors="replace").replace("\r\n", "\n").strip("\n"))
            manifest.append({"path": str(f), "sha256": sha256(b), "bytes": len(b)})
        text = "\n\n".join(texts)
        sources.append({"key": name, "files": manifest, "clean_sha256": sha256(text), "kind": kind})
        log(f"[{name}] {len(files)} local files ({kind}), {len(text)} chars -> {a.n_seqs} windows")
        for k, win in enumerate(cut_windows(api, text, a.n_seqs, a.seq_len, kind)):
            seqs.append({"id": f"{name}/{name}/{k}", "source": name, **win})
    else:
        raise SystemExit(f"unknown set {name!r} (fr, en, code, or one given with --local NAME=GLOB)")
    corpus = {"format": "nll_eval.corpus/1", "tool": VERSION, "set": name, "seq_len": a.seq_len,
              "n_seqs": len(seqs), "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "server": api.server_info(),
              "sources": sources, "seqs": seqs,
              "corpus_sha256": sha256("|".join(s["ids_sha256"] for s in seqs))}
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(corpus, ensure_ascii=False))
    tmp.replace(path)
    log(f"[{name}] frozen {len(seqs)} x {a.seq_len} tokens -> {path} (sha {corpus['corpus_sha256'][:12]})")


def load_corpus(cache, name):
    p = cache / f"corpus-{name}.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run build first")
    c = json.loads(p.read_text())
    if c["corpus_sha256"] != sha256("|".join(s["ids_sha256"] for s in c["seqs"])) or \
            any(ids_sha(s["ids"]) != s["ids_sha256"] for s in c["seqs"]):
        raise SystemExit(f"{p} fails its own checksums")
    return c


# ---------------------------------------------------------------------------------------------------- scoring
def score(api, ids, topk, salt):
    """Teacher-forced logprobs of ids[1:] given the prefix. Returns (logprobs, ranks, topk lists, seconds)."""
    body = {"model": api.model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": topk,
            "return_token_ids": True, "add_special_tokens": False}
    if salt:
        body["cache_salt"] = salt
    t0 = time.time()
    r = api.post("/v1/completions", body)
    dt = time.time() - t0
    ch = r["choices"][0]
    if ch.get("prompt_token_ids") is None:
        raise SystemExit("server did not echo prompt_token_ids (return_token_ids ignored?): cannot verify the prompt")
    if ch["prompt_token_ids"] != ids:
        raise SystemExit("server re-tokenized the prompt: prompt_token_ids differ from the ids sent")
    plp = ch.get("prompt_logprobs")
    if plp is None or len(plp) != len(ids):
        raise SystemExit(f"prompt_logprobs missing or wrong length ({None if plp is None else len(plp)} vs {len(ids)})")
    lps, ranks, tops = [], [], []
    for j in range(1, len(ids)):
        d = plp[j]
        e = d.get(str(ids[j]))
        if e is None:
            raise SystemExit(f"position {j}: actual token {ids[j]} missing from prompt_logprobs")
        v = float(e["logprob"])
        if not math.isfinite(v) or v <= -9998:     # vLLM serializes -inf as -9999 (generate/base/serving.py)
            raise SystemExit(f"position {j}: non-finite logprob {v} for token {ids[j]}")
        lps.append(v)
        ranks.append(e.get("rank"))
        if topk:
            items = sorted(((int(k), float(v["logprob"])) for k, v in d.items()), key=lambda x: -x[1])
            tops.append([[i for i, _ in items], [round(l, 6) for _, l in items]])
    return lps, ranks, tops, dt


def cmd_run(a):
    cache = Path(a.cache)
    api = Api(a.base_url, a.model, a.timeout)
    sets = [load_corpus(cache, s) for s in a.sets.split(",")]
    for c in sets:
        if c["seq_len"] != a.seq_len:
            raise SystemExit(f"corpus {c['set']} was built with seq_len {c['seq_len']}, not {a.seq_len}")
    first = sets[0]["seqs"][0]["ids"]
    req = {"model": a.model, "prompt": [a.bos_id] + first, "max_tokens": 1, "temperature": 0,
           "prompt_logprobs": a.topk, "return_token_ids": True}
    if a.dry_run:
        print(json.dumps({**req, "prompt": req["prompt"][:16] + ["..."]}, indent=1))
        print(f"{sum(len(c['seqs']) for c in sets)} requests of {a.seq_len + 1} prompt tokens", file=sys.stderr)
        return
    info = api.server_info()
    bos = api.detokenize([a.bos_id])
    log(f"server ok: {info['models']}; bos {a.bos_id} = {bos!r}")
    # Warm-up (not recorded): first prompt_logprobs call compiles batched_count_greater_than and grows the
    # allocator; a short prompt keeps it cheap.
    score(api, [a.bos_id] + first[:256], a.topk, None if a.no_cache_salt else f"nll-warm-{uuid.uuid4().hex}")
    run = {"format": "nll_eval.run/1", "tool": VERSION, "label": a.label, "notes": a.notes,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "server": info,
           "params": {"seq_len": a.seq_len, "skip": a.skip, "bos_id": a.bos_id, "topk": a.topk,
                      "cache_salt": not a.no_cache_salt, "store_tokens": not a.no_store_tokens}, "sets": {}}
    total = sum(len(c["seqs"]) for c in sets)
    done, t_all = 0, time.time()
    for c in sets:
        rows = []
        for s in c["seqs"]:
            ids = [a.bos_id] + s["ids"]
            salt = None if a.no_cache_salt else f"nll-{a.label}-{uuid.uuid4().hex}"
            lps, ranks, tops, dt = score(api, ids, a.topk, salt)
            sc = lps[a.skip:]                       # lps[i] scores ids[i+1] = window token i
            row = {"id": s["id"], "ids_sha256": s["ids_sha256"], "n_scored": len(sc),
                   "sum_nll": -math.fsum(sc), "mean_nll": -math.fsum(sc) / len(sc),
                   "top1_acc": sum(1 for r in ranks[a.skip:] if r == 1) / len(sc), "latency_s": round(dt, 3)}
            if not a.no_store_tokens:
                row["logprobs"] = [round(x, 7) for x in sc]
                row["ranks"] = ranks[a.skip:]
            if a.topk:
                row["topk"] = tops[a.skip:]
            rows.append(row)
            done += 1
            log(f"  [{done}/{total}] {s['id']}: nll {row['mean_nll']:.4f} ppl {math.exp(row['mean_nll']):.3f} "
                f"({dt:.1f}s)")
        run["sets"][c["set"]] = {"corpus_sha256": c["corpus_sha256"], "seqs": rows}
    if a.repeat:
        # Determinism probe: rescore the first --repeat windows of each set and record the largest drift.
        rep = []
        for c in sets:
            for s, row in zip(c["seqs"][:a.repeat], run["sets"][c["set"]]["seqs"]):
                lps, _, _, _ = score(api, [a.bos_id] + s["ids"], 0,
                                     None if a.no_cache_salt else f"nll-rep-{uuid.uuid4().hex}")
                sc = lps[a.skip:]
                d_seq = -math.fsum(sc) / len(sc) - row["mean_nll"]
                d_tok = max(abs(x - y) for x, y in zip(sc, row["logprobs"])) if "logprobs" in row else None
                rep.append({"id": s["id"], "d_mean_nll": d_seq, "max_abs_d_token_logprob": d_tok})
        run["repeat"] = rep
        log("repeat: max |d mean nll| = %.3g" % max(abs(r["d_mean_nll"]) for r in rep))
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run["wall_s"] = round(time.time() - t_all, 1)
    for name, st in run["sets"].items():
        m = sum(r["sum_nll"] for r in st["seqs"]) / sum(r["n_scored"] for r in st["seqs"])
        log(f"{name}: mean NLL {m:.5f}  PPL {math.exp(m):.4f}  ({len(st['seqs'])} seqs)")
    write_json(Path(a.out), run)
    log(f"saved {a.out}")


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False).encode()
    tmp = p.with_name(p.name + ".part")
    tmp.write_bytes(gzip.compress(data) if p.suffix == ".gz" else data)
    tmp.replace(p)


def read_json(p):
    b = Path(p).read_bytes()
    return json.loads(gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b)


# ---------------------------------------------------------------------------------------------------- statistics
def mean(x):
    return math.fsum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(math.fsum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float("nan")


def t975(df):
    """0.975 quantile of Student t (Cornish-Fisher expansion; < 0.3 % error for df >= 5)."""
    z = 1.959963985
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def sign_test(d):
    pos, neg = sum(1 for v in d if v > 0), sum(1 for v in d if v < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n   # exact integers (a float fsum overflowed for n ~ 400)
    return pos, neg, min(1.0, p)


def wilcoxon(d):
    """Two-sided Wilcoxon signed-rank, normal approximation with tie and continuity corrections."""
    x = [v for v in d if v != 0]
    n = len(x)
    if n < 6:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(x[i]))
    ranks = [0.0] * n
    i, ties = 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(x[order[j + 1]]) == abs(x[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    wp = math.fsum(r for r, v in zip(ranks, x) if v > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    z = (abs(wp - mu) - 0.5) / math.sqrt(var)
    return math.erfc(max(z, 0) / math.sqrt(2))


def bootstrap_ci(strata, B, seed):
    """Percentile 95 % CI of the pooled mean of paired differences, resampling sequences within each stratum (set)
    so every resample keeps the set sizes."""
    rng = random.Random(seed)
    tot = sum(len(s) for s in strata)
    stats = []
    for _ in range(B):
        acc = 0.0
        for s in strata:
            n = len(s)
            acc += math.fsum(s[rng.randrange(n)] for _ in range(n))
        stats.append(acc / tot)
    stats.sort()
    return stats[int(0.025 * B)], stats[int(0.975 * B) - 1]


def kl_lower_bound(ta, tb):
    """KL(A||B) on the partition {tokens listed by both} + {everything else}: by the data-processing inequality
    this is a lower bound of the full-vocabulary KL. ta/tb = [ids, logprobs] for one position."""
    da, db = dict(zip(*ta)), dict(zip(*tb))
    common = da.keys() & db.keys()
    kl, pa_s, pb_s = 0.0, 0.0, 0.0
    for t in common:
        pa, pb = math.exp(da[t]), math.exp(db[t])
        pa_s += pa
        pb_s += pb
        if pa > 0:
            kl += pa * (da[t] - db[t])
    ra, rb = max(1.0 - pa_s, 0.0), max(1.0 - pb_s, 1e-12)
    if ra > 1e-12:
        kl += ra * math.log(ra / rb)
    return max(kl, 0.0)


def paired_rows(A, B, n):
    sa, sb = A["sets"][n], B["sets"][n]
    if sa["corpus_sha256"] != sb["corpus_sha256"]:
        raise SystemExit(f"set {n}: the two runs used different corpora")
    ra = {r["id"]: r for r in sa["seqs"]}
    rb = {r["id"]: r for r in sb["seqs"]}
    ids = [i for i in ra if i in rb]
    if len(ids) != len(ra) or len(ids) != len(rb):
        log(f"warning: set {n}: {len(ra)} vs {len(rb)} sequences, comparing the {len(ids)} common ones")
    for i in ids:
        if ra[i]["ids_sha256"] != rb[i]["ids_sha256"] or ra[i]["n_scored"] != rb[i]["n_scored"]:
            raise SystemExit(f"{i}: token windows differ between the runs")
    return [ra[i] for i in ids], [rb[i] for i in ids]


def token_extras(xa, xb):
    tok = {}
    if all("logprobs" in r for r in xa + xb):
        ad = sorted(abs(y - x) for p, q in zip(xa, xb) for x, y in zip(p["logprobs"], q["logprobs"]))
        tok = {"mean_abs_d_logprob": mean(ad), "p99_abs_d_logprob": ad[int(0.99 * (len(ad) - 1))],
               "frac_abs_d_gt_0.1": sum(1 for v in ad if v > 0.1) / len(ad)}
    if all("topk" in r for r in xa + xb):
        kls, agree = [], 0
        for p, q in zip(xa, xb):
            for ta, tb in zip(p["topk"], q["topk"]):
                kls.append(kl_lower_bound(ta, tb))
                agree += ta[0][0] == tb[0][0]
        tok.update({"kl_a_b_lower_bound_mean": mean(kls), "argmax_agreement": agree / len(kls)})
    return tok


def summarize(xa, xb, strata, B, seed):
    """xa/xb: paired per-sequence rows (all strata concatenated); strata: list of per-set d lists."""
    d = [x for s in strata for x in s]
    pa = math.fsum(r["sum_nll"] for r in xa) / sum(r["n_scored"] for r in xa)
    pb = math.fsum(r["sum_nll"] for r in xb) / sum(r["n_scored"] for r in xb)
    m, s = mean(d), sd(d)
    se = s / math.sqrt(len(d))
    tci = (m - t975(len(d) - 1) * se, m + t975(len(d) - 1) * se)
    ci = bootstrap_ci(strata, B, seed)
    pos, neg, sp = sign_test(d)
    dacc = mean([q["top1_acc"] - p["top1_acc"] for p, q in zip(xa, xb)])
    return {"n": len(d), "nll_a": pa, "nll_b": pb, "ppl_a": math.exp(pa), "ppl_b": math.exp(pb),
            "dppl_pct": 100 * (math.exp(pb - pa) - 1),
            "dppl_pct_boot_ci95": [100 * (math.exp(ci[0]) - 1), 100 * (math.exp(ci[1]) - 1)],
            "mean_d": m, "sd_d": s, "se_d": se, "boot_ci95_d": list(ci), "t_ci95_d": list(tci),
            "b_better": neg, "b_worse": pos, "sign_p": sp, "wilcoxon_p": wilcoxon(d),
            "mde80_pct": 100 * (math.exp((t975(len(d) - 1) + 0.8416) * se) - 1), "d_top1_acc_pct": 100 * dacc}


def cmd_compare(a):
    A, B = read_json(a.a), read_json(a.b)
    if "nll_eval.selfcheck/1" in (A.get("format"), B.get("format")):
        for X, f in ((A, a.a), (B, a.b)):
            print(f"== {X.get('label')}  ({f})")
            print(json.dumps(X["summary"], indent=1))
        return
    for k in ("seq_len", "skip", "bos_id"):
        if A["params"][k] != B["params"][k]:
            raise SystemExit(f"param {k} differs: {A['params'][k]} vs {B['params'][k]}")
    names = [n for n in A["sets"] if n in B["sets"]]
    if not names:
        raise SystemExit("no set in common")
    out = {"a": {"file": a.a, "label": A["label"], "notes": A.get("notes")},
           "b": {"file": a.b, "label": B["label"], "notes": B.get("notes")}, "sets": {}}
    print(f"A = {A['label']}  ({a.a})\nB = {B['label']}  ({a.b})\n"
          "d = NLL_B - NLL_A per sequence, nats/token (< 0: B predicts the held-out text better)\n")
    print(f"{'set':<6}{'n':>4} {'PPL_A':>8} {'PPL_B':>8} {'dPPL%':>8}  {'boot 95% CI dPPL%':>19} {'mean d':>10} "
          f"{'sd d':>8} {'B<A/B>A':>8} {'sign p':>8} {'wilc p':>8} {'MDE%':>6} {'dTop1%':>7}")
    all_a, all_b, strata = [], [], []
    rows = []
    for n in names:
        xa, xb = paired_rows(A, B, n)
        d = [q["mean_nll"] - p["mean_nll"] for p, q in zip(xa, xb)]
        rec = summarize(xa, xb, [d], a.boot, a.seed)
        rec["token"] = token_extras(xa, xb)
        out["sets"][n] = rec
        rows.append((n, rec))
        all_a += xa
        all_b += xb
        strata.append(d)
    if len(names) > 1:
        rec = summarize(all_a, all_b, strata, a.boot, a.seed)   # stratified bootstrap over the sets
        out["sets"]["all"] = rec
        rows.append(("all", rec))
    for n, r in rows:
        lo, hi = r["dppl_pct_boot_ci95"]
        print(f"{n:<6}{r['n']:>4} {r['ppl_a']:>8.4f} {r['ppl_b']:>8.4f} {r['dppl_pct']:>+8.3f}  "
              f"[{lo:>+8.3f},{hi:>+8.3f}] {r['mean_d']:>+10.5f} {r['sd_d']:>8.5f} "
              f"{str(r['b_better']) + '/' + str(r['b_worse']):>8} {r['sign_p']:>8.3g} {r['wilcoxon_p']:>8.3g} "
              f"{r['mde80_pct']:>6.3f} {r['d_top1_acc_pct']:>+7.3f}")
    for n in names:
        if out["sets"][n]["token"]:
            print(f"  {n} token-level: " + ", ".join(f"{k} {v:.4g}" for k, v in out["sets"][n]["token"].items()))
    for X, lab in ((A, "A"), (B, "B")):
        if X.get("repeat"):
            print(f"  {lab} repeat drift (same request twice): max |d mean nll| "
                  f"{max(abs(r['d_mean_nll']) for r in X['repeat']):.3g}")
    print("\nResolved effect = bootstrap CI excludes 0 AND |mean d| is well above the cross-restart null pair (same"
          "\nweights, server restarted). An A/A pair inside one process is bit-exact and is NOT a noise floor."
          "\nMDE% = smallest dPPL detectable with 80 % power (two-sided 5 %) at this n and spread of d.")
    if a.json:
        write_json(Path(a.json), out)


# ---------------------------------------------------------------------------------------------------- selfcheck
def cmd_selfcheck(a):
    cache = Path(a.cache)
    api = Api(a.base_url, a.model, a.timeout)
    info = api.server_info()
    salt = lambda: None if a.no_cache_salt else f"nll-sc-{uuid.uuid4().hex}"
    res = {"format": "nll_eval.selfcheck/1", "tool": VERSION, "label": a.label, "notes": a.notes,
           "server": info, "params": {"prompt_len": a.prompt_len, "gen": a.gen, "n": a.n, "bos_id": a.bos_id,
                      "canary_len": a.canary_len},
           "items": []}
    absd, rank_miss, n_tok = [], 0, 0
    for name in a.sets.split(","):
        c = load_corpus(cache, name)
        for s in c["seqs"][:a.n]:
            prompt = [a.bos_id] + s["ids"][:a.prompt_len]
            g = api.post("/v1/completions", {
                "model": a.model, "prompt": prompt, "max_tokens": a.gen, "temperature": 0, "logprobs": 1,
                "return_token_ids": True, "return_tokens_as_token_ids": True, "add_special_tokens": False,
                **({"cache_salt": salt()} if not a.no_cache_salt else {})})["choices"][0]
            gen_ids = g.get("token_ids") or [int(t.split(":", 1)[1]) for t in g["logprobs"]["tokens"]]
            dec = g["logprobs"]["token_logprobs"]
            lps, ranks, _, _ = score(api, prompt + gen_ids, 0, salt())
            pre = lps[len(prompt) - 1:]                  # logprobs of the generated tokens, prefill path
            pr = ranks[len(prompt) - 1:]
            d = [x - y for x, y in zip(dec, pre)]
            absd += [abs(v) for v in d]
            rank_miss += sum(1 for r in pr if r != 1)
            n_tok += len(d)
            res["items"].append({"id": s["id"], "n_gen": len(gen_ids), "finish": g.get("finish_reason"),
                                 "decode_logprobs": dec, "prefill_logprobs": pre, "prefill_ranks": pr,
                                 "max_abs_d": max(abs(v) for v in d) if d else None})
            log(f"  {s['id']}: {len(d)} tokens, mean |decode - prefill| {mean([abs(v) for v in d]):.4f}, "
                f"prefill rank != 1: {sum(1 for r in pr if r != 1)}")
    # Causality canary: a random block repeated once. The first copy must stay at its causal NLL (high); a kernel
    # that leaks later rows into earlier ones drops it (GNmac saw -12 % with a broken slice, how-it-works.md:95-97).
    rng = random.Random(1234)
    L = a.canary_len
    r = [rng.randrange(1000, 127000) for _ in range(L)]
    lps, _, _, _ = score(api, [a.bos_id] + r + r, 0, salt())
    first, second = lps[1:L], lps[L + 1:]                    # lps[i] scores ids[i+1]; both skip r[0]
    canary = {"block": L, "first_copy_nll": -mean(first), "second_copy_nll": -mean(second)}
    absd.sort()
    res["summary"] = {"tokens": n_tok, "mean_abs_decode_minus_prefill": mean(absd) if absd else None,
                      "p99_abs": absd[int(0.99 * (len(absd) - 1))] if absd else None,
                      "max_abs": absd[-1] if absd else None,
                      "frac_prefill_rank_not_1": rank_miss / n_tok if n_tok else None, "canary": canary}
    print(json.dumps(res["summary"], indent=1))
    write_json(Path(a.out), res)
    log(f"saved {a.out}")


# ---------------------------------------------------------------------------------------------------- main
def cmd_build(a):
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    api = Api(a.base_url, a.model, a.timeout)
    info = api.server_info()
    log(f"server ok: {info['models']}")
    probe = api.tokenize("Bonjour le monde")
    if not probe or probe[0] == a.bos_id:
        raise SystemExit(f"/tokenize with add_special_tokens=false returned {probe}: unexpected BOS handling")
    for name in a.sets.split(","):
        build_set(api, a, name, cache)


def cmd_info(a):
    for p in sorted(Path(a.cache).glob("corpus-*.json")):
        c = json.loads(p.read_text())
        print(f"{c['set']:<8} {c['n_seqs']:>4} x {c['seq_len']} tokens  sha {c['corpus_sha256'][:16]}  "
              f"sources: {', '.join(s['key'] for s in c['sources'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, server=True):
        if server:
            p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8011")
            p.add_argument("--model", default="deepseek-v4.1-flash")
            p.add_argument("--timeout", type=float, default=1800)
        p.add_argument("--cache", default="nllcache", help="corpus cache directory")
        p.add_argument("--sets", default="fr,en,code")
        p.add_argument("--seq-len", type=int, default=1024,
                   help="window tokens (BOS added on top); <= 41 scores the decode-shaped (coop) MoE kernels")
        p.add_argument("--bos-id", type=int, default=0, help="<｜begin▁of▁sentence｜> (config.json bos_token_id)")
        p.add_argument("--no-cache-salt", action="store_true",
                       help="do not send a per-request cache_salt (prompt_logprobs already skips cache reads)")

    p = sub.add_parser("build")
    common(p)
    p.add_argument("--n-seqs", type=int, default=64)
    p.add_argument("--code-dir", default=None)
    p.add_argument("--local", action="append", default=[], metavar="NAME=GLOB",
                   help="extra set from local files (.txt/.md -> prose windows, else code)")
    p.add_argument("--rebuild", action="store_true")
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--label", required=True)
    p.add_argument("--notes", default="", help="free text stored in the run (weights, K map, util, ...)")
    p.add_argument("--out", required=True, help=".json or .json.gz")
    p.add_argument("--skip", type=int, default=1,
                   help="window tokens not scored at the start (1 = score tokens 2..seq_len, like LibertAI)")
    p.add_argument("--topk", type=int, default=0, help="also keep the top-k prompt logprobs (<= --max-logprobs, 20)")
    p.add_argument("--repeat", type=int, default=2, help="rescore the first N windows per set (determinism probe)")
    p.add_argument("--no-store-tokens", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the first request and exit; sends nothing")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p = sub.add_parser("selfcheck")
    common(p)
    p.add_argument("--label", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4, help="windows per set")
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--canary-len", type=int, default=1000,
                   help="random block length, repeated once (must exceed one prefill slice to test slice causality)")
    p = sub.add_parser("info")
    p.add_argument("--cache", default="nllcache")
    a = ap.parse_args()
    if a.cmd == "build":
        a.local_specs = {}
        for spec in a.local:
            k, _, v = spec.partition("=")
            if not k or not v or k in ("fr", "en", "code"):
                raise SystemExit(f"bad --local {spec!r}")
            a.local_specs[k] = v
    {"build": cmd_build, "run": cmd_run, "compare": cmd_compare, "selfcheck": cmd_selfcheck,
     "info": cmd_info}[a.cmd](a)


if __name__ == "__main__":
    main()
