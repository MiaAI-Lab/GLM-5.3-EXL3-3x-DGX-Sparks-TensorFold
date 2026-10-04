"""Live A/B for copy drafts (TF_GLM_COPY_HYBRID): edit/quote-heavy prompts plus two controls, on :8888.
   python3 tools/copy_ab.py <label>          one line per request: tok/s, tokens/round, round kinds, reply sha
Run once with the hybrid off and once on (two boots); greedy replies must have the same sha in both (exactness)."""
import collections, json, pathlib, sys, urllib.request
URL = "http://127.0.0.1:8888/v1/chat/completions"
R = pathlib.Path(__file__).resolve().parents[1]
code = "\n".join((R / "tools/quality.py").read_text().splitlines()[:140])
prose = (R / "README.md").read_text().split("### Memory")[0].split("### Quality")[0][-3000:]
cfg = json.dumps({"server": {"host": "0.0.0.0", "port": 8888, "workers": 3, "timeout_s": 600},
                  "model": {"name": "GLM-5.3-EXL3", "context": 499712, "kv": "fp4", "prefill_rows": 3072},
                  "drafter": {"kind": "dspark", "block": 8, "copy": True},
                  "cache": {"disk": "/cache/pcache", "budget_gib": 64, "keep_free_gib": 100},
                  "users": [{"name": f"user{i}", "quota": 1000 * i, "admin": i == 0} for i in range(12)]}, indent=2)
P = {
    "edit-code": f"Add a short docstring to every function that lacks one in this Python code. Change nothing else "
                 f"and return the whole file in one code block.\n\n```python\n{code}\n```",
    "edit-prose": f"Fix any spelling or grammar mistakes in this text and return the full corrected text, keeping the "
                  f"Markdown exactly as it is.\n\n{prose}",
    "edit-json": f"In this JSON, set the port to 9000 and every user's quota to double its value. Return the complete "
                 f"JSON.\n\n```json\n{cfg}\n```",
    "ctl-prose": "Write a long, detailed essay about the history of lighthouses and the people who kept them.",
    "ctl-code": "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, with docstrings.",
}
for kind, prompt in P.items():
    for temp, seed in ((0.0, 1), (1.0, 11), (1.0, 22)):
        body = {"model": "GLM-5.3-EXL3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024,
                "temperature": temp, "top_p": 0.95, "seed": seed, "chat_template_kwargs": {"thinking": False}}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=1200))
        tf = d.get("tensorfold", {})
        n, s = d["usage"]["completion_tokens"], tf.get("decode_s") or float("nan")
        arms = dict(collections.Counter(tf.get("drafters") or []))
        print(f"{sys.argv[1]} {kind:10s} T={temp} seed {seed}: {n} tok, {n / s:5.1f} tok/s, "
              f"tok/round {tf.get('tokens_per_round')}, copy {tf.get('copy_accepted')}/{tf.get('copy_drafted')}, "
              f"kinds {arms}, sha {tf.get('sha256')}", flush=True)
