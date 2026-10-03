#!/usr/bin/env python3
"""Exact replies through the running server: every drafted reply (MTP and copy drafts, the default) must equal its
serial reference ("draft": false: one token a round, a fresh prefill), and sending the same requests at once (they
queue: one request at a time) must not change any reply. Greedy and seeded sampling, thinking on and off, prose, code
and a quote-and-edit task (copy drafts). Replies are compared by the server's token_sha (a hash of the reply's token
ids), and the text too.

Usage: tools/exact.py [label] [--tokens 400] [--no-concurrent]   (API_URL / PORT, MODEL as in client.py)
Exit 0 when every reply equals its reference, 1 otherwise."""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

sys.dont_write_bytecode = True
from client import URL, open_url  # noqa: E402

MODEL = os.environ.get("MODEL", "GLM-5.3-EXL3")
EDIT = ("def area(r):\n    return 3.14159 * r * r\n\n\ndef circumference(r):\n    return 2 * 3.14159 * r\n\n\n"
        "def describe(r):\n    return f'radius {r}: area {area(r):.2f}, circumference {circumference(r):.2f}'\n")
CASES = [
    ("code", "Write a Python function that merges two sorted lists, with a docstring.", 0.0, False),
    ("prose", "Explain how a hash map handles collisions, in two paragraphs.", 0.0, False),
    ("haiku", "Write a haiku about rivers.", 1.0, False),
    ("think", "List five uses of binary search.", 1.0, True),
    ("bash", "Write a bash one-liner that counts lines in every .py file under a directory.", 0.0, True),
    ("edit", "Rename the parameter r to radius everywhere and repeat the whole file:\n\n" + EDIT, 0.0, False),
]


def ask(prompt: str, temperature: float, thinking: bool, draft: bool, tokens: int) -> tuple[str, str]:
    body = {"model": MODEL, "max_tokens": tokens, "seed": 7, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": thinking}, "messages": [{"role": "user", "content": prompt}]}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(open_url(req, 3600))
    m = r["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    sha = (r.get("tensorfold") or {}).get("token_sha") or hashlib.sha256(text.encode()).hexdigest()[:12]
    return sha, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default="exact")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--no-concurrent", action="store_true")
    a = ap.parse_args()
    ref, bad = {}, 0
    t0 = time.time()
    for name, prompt, temp, thinking in CASES:
        drafted = ask(prompt, temp, thinking, True, a.tokens)
        serial = ask(prompt, temp, thinking, False, a.tokens)
        ref[name] = serial
        same = drafted == serial
        bad += not same
        print(f"  {'same' if same else 'DIFFERENT'}  drafted vs serial  {name:6s} T={temp} thinking={thinking} "
              f"{serial[0]}", flush=True)
    if not a.no_concurrent:
        got: dict = {}

        def run(case):
            name, prompt, temp, thinking = case
            got[name] = ask(prompt, temp, thinking, True, a.tokens)

        threads = [threading.Thread(target=run, args=(c,)) for c in CASES]
        for t in threads:
            t.start()
            time.sleep(0.2)
        for t in threads:
            t.join()
        for name, _, _, _ in CASES:
            same = got.get(name) == ref[name]
            bad += not same
            print(f"  {'same' if same else 'DIFFERENT'}  concurrent vs serial  {name}", flush=True)
    n = len(CASES) * (1 if a.no_concurrent else 2)
    print(f"{a.label}: {n - bad}/{n} replies equal their serial references ({time.time() - t0:.0f} s)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
