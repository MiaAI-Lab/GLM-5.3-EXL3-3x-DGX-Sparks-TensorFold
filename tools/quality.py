#!/usr/bin/env python3
"""Quality suite for an OpenAI-compatible chat server: fixed, seeded items with checkable answers, greedy (temperature
0), so two runs (two models, quantizations, KV-cache formats, engine versions) can be compared item by item.

    tools/quality.py run <label> [--url http://127.0.0.1:8888] [--model ID] [--api-key KEY]
                     [--only qa,reason,arith,track,code,long,gen] [--quick] [--long-sizes 32768,131072] [--conc 4]
    tools/quality.py compare <label A> <label B>
    tools/quality.py show <label>

Results go to ./quality/<label>.json (QUALITY_DIR to change). ``run`` resumes nothing: it re-runs the categories asked
for and replaces them in the label's file, keeping the others.

Categories (default: all but long; --quick: qa and code only):
  qa      150 short questions (30 fixed + 120 seeded: arithmetic, strings, letter counts, sorting, binary, dates, LCM),
          thinking off
  reason  40 seeded multi-step word problems, thinking ON (long generations: small errors accumulate)
  arith   40 seeded chains of 10 exact integer operations, thinking ON
  track   40 seeded ledgers of 25 token transfers between 5 people, one balance asked, thinking ON
  code    25 Python function tasks, thinking off; the reply's code is run against hidden tests in a subprocess
          (10 s timeout, isolated mode). This executes code the model wrote: run it on a machine you trust it on.
  long    recall of 16 keys, 4 of them corrected later (the latest value counts), at --long-sizes tokens (approximate),
          sizes x items: 32768 x3, 131072 x2, 262144 x2 by default; keep them under the server's context window
  gen     20 open prompts, 400 tokens: only reply agreement between two runs (no score)

``compare`` pairs the items of two runs: identical replies, items only one run got right, and an exact two-sided
McNemar p-value (p >= 0.05: the difference is within noise for that many items).
"""
import concurrent.futures as cf
import datetime
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import os

URL = "http://127.0.0.1:8888"
MODEL = None
API_KEY = os.environ.get("API_KEY", "")
OUT = Path(os.environ.get("QUALITY_DIR", "quality"))

# question, accepted answers (any substring match, case-insensitive), max tokens
QUESTIONS = [
    ("What is 17 * 23? Answer with the number only.", ["391"], 16),
    ("What is 1234 + 8766? Answer with the number only.", ["10000"], 16),
    ("What is 2 to the power of 20? Answer with the number only.", ["1048576", "1,048,576"], 16),
    ("A train travels 180 km in 2.5 hours. What is its average speed in km/h? Number only.", ["72"], 16),
    ("If x + 2x + 3x = 48, what is x? Number only.", ["8"], 16),
    ("What is the remainder when 1000 is divided by 7? Number only.", ["6"], 16),
    ("How many minutes are in 3.5 hours? Number only.", ["210"], 16),
    ("What is the square root of 1764? Number only.", ["42"], 16),
    ("What is 15% of 240? Number only.", ["36"], 16),
    ("A rectangle is 12 by 7. What is its area? Number only.", ["84"], 16),
    ("Reverse the string 'tensorfold'. Answer with the reversed string only.", ["dlofrosnet"], 16),
    ("How many letters 'r' are in the word 'strawberry'? Number only.", ["3"], 16),
    ("Sort these numbers ascending: 9, 2, 7, 4, 1. Answer as a comma-separated list.", ["1, 2, 4, 7, 9", "1,2,4,7,9"], 24),
    ("What comes next: 2, 6, 12, 20, 30, ? Number only.", ["42"], 16),
    ("All bloops are razzies and all razzies are lazzies. Are all bloops lazzies? Answer yes or no.", ["yes"], 8),
    ("If today is Wednesday, what day is it 10 days from now? One word.", ["saturday"], 8),
    ("What is the capital of Australia? One word.", ["canberra"], 8),
    ("What is the chemical symbol for gold? Symbol only.", ["au"], 8),
    ("Who wrote 'Pride and Prejudice'? Name only.", ["austen"], 12),
    ("What is the boiling point of water at sea level in degrees Celsius? Number only.", ["100"], 8),
    ("How many sides does a hexagon have? Number only.", ["6"], 8),
    ("What is the largest planet in our solar system? One word.", ["jupiter"], 8),
    ("In Python, what does len([1, [2, 3], 4]) return? Number only.", ["3"], 8),
    ("In Python, what is the value of 7 // 2? Number only.", ["3"], 8),
    ("What is the binary representation of 13? Digits only.", ["1101"], 12),
    ("What is 0.1 + 0.2 rounded to one decimal place? Number only.", ["0.3"], 8),
    ("Convert 5 kilometers to meters. Number only.", ["5000", "5,000"], 12),
    ("What is the greatest common divisor of 84 and 36? Number only.", ["12"], 8),
    ("Spell the word 'necessary' backwards. Letters only.", ["yrassecen"], 16),
    ("What is 999 * 999? Number only.", ["998001", "998,001"], 16),
]

WORDS = ("amber basil cedar delta ember fable garnet harbor indigo jasper kettle lantern meadow nectar orbit pepper "
         "quartz raven saffron timber umber velvet willow xenon yarrow zephyr").split()

WORDS = ("amber basil cedar delta ember fable garnet harbor indigo jasper kettle lantern meadow nectar orbit pepper "
         "quartz raven saffron timber umber velvet willow xenon yarrow zephyr").split()


def _request(path: str, body: dict | None, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL.rstrip("/") + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def model_id() -> str:
    global MODEL
    if MODEL is None:
        MODEL = _request("/v1/models", None, 30)["data"][0]["id"]
    return MODEL


def chat(content: str, max_tokens: int, thinking: bool = False, timeout: float = 7200.0) -> dict:
    # thinking: GLM's chat template flag and the common enable_thinking (templates ignore the one they do not use)
    body = {"model": model_id(), "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 1234,
            "chat_template_kwargs": {"thinking": thinking, "enable_thinking": thinking}}
    t0 = time.time()
    d = _request("/v1/chat/completions", body, timeout)
    m = d["choices"][0]["message"]
    return {"text": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "finish": d["choices"][0].get("finish_reason"), "tokens": d.get("usage", {}).get("completion_tokens"),
            "prompt_tokens": d.get("usage", {}).get("prompt_tokens"), "s": round(time.time() - t0, 1)}


# -- qa ---------------------------------------------------------------------------------------------------------------
def qa_items() -> list[dict]:
    items = [{"id": f"fixed{i}", "q": q, "ok": a, "mt": mt} for i, (q, a, mt) in enumerate(QUESTIONS)]
    rng = random.Random(53)
    for i in range(20):
        a, b = rng.randint(100, 999), rng.randint(11, 99)
        items.append({"id": f"mul{i}", "q": f"What is {a} * {b}? Answer with the number only.", "ok": [str(a * b)], "mt": 16})
    for i in range(15):
        a, b, c = rng.randint(1000, 9999), rng.randint(1000, 9999), rng.randint(100, 999)
        items.append({"id": f"add{i}", "q": f"What is {a} + {b} - {c}? Answer with the number only.",
                      "ok": [str(a + b - c)], "mt": 16})
    for i in range(15):
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(6, 10)))
        items.append({"id": f"rev{i}", "q": f"Reverse the string '{w}'. Answer with the reversed string only.",
                      "ok": [w[::-1]], "mt": 24})
    for i in range(15):
        w = rng.choice(["mississippi", "bookkeeper", "banana", "committee", "parallel", "assessment", "successful",
                        "occurrence", "tattoo", "referee", "balloon", "coffee", "letter", "address", "giraffe"])
        ch = rng.choice(sorted(set(w)))
        items.append({"id": f"cnt{i}", "q": f"How many times does the letter '{ch}' appear in the word '{w}'? Number only.",
                      "ok": [str(w.count(ch))], "mt": 8, "exact": True})
    for i in range(15):
        xs = rng.sample(range(1, 100), 7)
        items.append({"id": f"sort{i}", "q": "Sort these numbers ascending: " + ", ".join(map(str, xs)) +
                      ". Answer as a comma-separated list only.", "ok": [", ".join(map(str, sorted(xs))),
                                                                       ",".join(map(str, sorted(xs)))], "mt": 40})
    for i in range(10):
        n = rng.randint(20, 255)
        items.append({"id": f"bin{i}", "q": f"What is {n} in binary? Digits only.", "ok": [bin(n)[2:]], "mt": 16,
                      "exact": True})
    for i in range(15):
        d0 = datetime.date(2024, 1, 1) + datetime.timedelta(days=rng.randint(0, 700))
        k = rng.randint(3, 60)
        d1 = d0 + datetime.timedelta(days=k)
        items.append({"id": f"date{i}", "q": f"What date is {k} days after {d0.isoformat()}? Answer in YYYY-MM-DD only.",
                      "ok": [d1.isoformat()], "mt": 16})
    for i in range(15):
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        items.append({"id": f"lcm{i}", "q": f"What is the least common multiple of {a} and {b}? Number only.",
                      "ok": [str(a * b // math.gcd(a, b))], "mt": 12, "exact": True})
    return items


def qa_score(it: dict, text: str) -> bool:
    t = text.strip().lower().rstrip(".")
    if it.get("exact"):
        nums = re.findall(r"[0-9]+", t)
        return bool(nums) and nums[-1] == it["ok"][0]
    return any(a.lower() in t for a in it["ok"])


# -- reason -----------------------------------------------------------------------------------------------------------
def reason_items() -> list[dict]:
    rng = random.Random(97)
    names = ["Ava", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun"]
    things = ["apples", "marbles", "stickers", "coins", "books", "cards", "shells", "pencils"]
    items = []
    for i in range(40):
        a, b, c = rng.sample(names, 3)
        th = rng.choice(things)
        x = rng.randint(20, 90)
        k = rng.randint(2, 5)
        g = rng.randint(3, 15)
        p = rng.choice([2, 3, 4])
        y = x * k                       # b has k times a
        z = y - g                       # b gives g to c ... c has z? build a chain
        cnt_c = rng.randint(5, 40)
        h = rng.randint(2, x // 2)
        b_after = y - g
        c_after = 2 * (cnt_c + g)
        a_after = x - h
        total = a_after + b_after + c_after
        share = total // p
        rem = total - share * p
        q = (f"{a} has {x} {th}. {b} has {k} times as many {th} as {a}. {c} has {cnt_c} {th}. {b} gives {g} {th} to "
             f"{c}. Then {a} loses {h} {th}, and {c} doubles the number of {th} they have. Then all three put their {th} together and split them as evenly as possible into {p} boxes, "
             f"putting any leftover {th} in a jar. How many {th} are in the jar, and how many are in each box? "
             f"End your reply with a line 'Answer: <jar>, <per box>'.")
        items.append({"id": f"r{i}", "q": q, "jar": rem, "box": share})
    return items


def reason_score(it: dict, text: str) -> bool:
    m = re.findall(r"answer:\s*\**\s*(\d+)\s*\**\s*,\s*\**\s*(\d+)", text.lower())
    return bool(m) and (int(m[-1][0]), int(m[-1][1])) == (it["jar"], it["box"])


# -- arith / track (harder, thinking on: long exact reasoning where small slips change the answer) -------------------
def arith_items() -> list[dict]:
    rng = random.Random(211)
    items = []
    for i in range(40):
        v0 = v = rng.randint(100, 999)
        text = []
        for _ in range(10):
            op = rng.choice(["mul", "add", "sub", "mod", "div"])
            if op == "mul":
                k = rng.randint(3, 19)
                v *= k
                text.append(f"multiply it by {k}")
            elif op == "add":
                k = rng.randint(100, 9999)
                v += k
                text.append(f"add {k}")
            elif op == "sub":
                k = rng.randint(10, max(11, v // 2))
                v -= k
                text.append(f"subtract {k}")
            elif op == "mod":
                k = rng.randint(1000, 9999)
                v = v % k + 1000
                text.append(f"replace it by its remainder when divided by {k}, then add 1000")
            else:
                k = rng.randint(2, 9)
                v //= k
                text.append(f"divide it by {k}, rounding down")
        q = (f"Start with {v0}. Then, in order: " + "; ".join(f"({j + 1}) {t}" for j, t in enumerate(text)) +
             ". What is the final number? End your reply with a line 'Answer: <number>'.")
        items.append({"id": f"a{i}", "q": q, "answer": v})
    return items


def track_items() -> list[dict]:
    rng = random.Random(307)
    people = ["Ana", "Bo", "Cy", "Di", "Ed"]
    items = []
    for i in range(40):
        have = {p: rng.randint(20, 60) for p in people}
        lines = [f"{p} starts with {have[p]} tokens." for p in people]
        for _ in range(25):
            a, b = rng.sample(people, 2)
            kind = rng.random()
            if kind < 0.6 and have[a] > 0:
                k = rng.randint(1, max(1, have[a] // 2))
                have[a] -= k
                have[b] += k
                lines.append(f"{a} gives {k} to {b}.")
            elif kind < 0.8:
                k = rng.randint(1, 9)
                have[a] += k
                lines.append(f"{a} finds {k}.")
            else:
                g = have[a] // 2
                have[a] -= g
                have[b] += g
                lines.append(f"{a} gives half of their tokens (rounded down) to {b}.")
        who = rng.choice(people)
        q = ("Track the tokens carefully. " + " ".join(lines) +
             f" How many tokens does {who} have at the end? End your reply with a line 'Answer: <number>'.")
        items.append({"id": f"t{i}", "q": q, "answer": have[who]})
    return items


def answer_score(it: dict, text: str) -> bool:
    m = re.findall(r"answer:\s*\**\s*(-?[\d,]+)", text.lower())
    return bool(m) and int(m[-1].replace(",", "")) == it["answer"]


# -- code -------------------------------------------------------------------------------------------------------------
CODE = [
    ("is_palindrome(s)", "returns True if the string s reads the same forwards and backwards ignoring case and "
     "non-alphanumeric characters", "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('abc')\nassert is_palindrome('')"),
    ("fib(n)", "returns the n-th Fibonacci number with fib(0) = 0 and fib(1) = 1, for n up to 90",
     "assert fib(0)==0 and fib(1)==1 and fib(10)==55 and fib(90)==2880067194370816120"),
    ("flatten(xs)", "flattens arbitrarily nested lists into one list, keeping order",
     "assert flatten([1,[2,[3,[4]],5],[]])==[1,2,3,4,5]\nassert flatten([])==[]"),
    ("roman(n)", "converts an integer 1..3999 to a Roman numeral string",
     "assert roman(1994)=='MCMXCIV' and roman(3999)=='MMMCMXCIX' and roman(4)=='IV'"),
    ("anagrams(words)", "groups words that are anagrams of each other; returns a list of groups, each group sorted, "
     "groups sorted by their first word", "assert anagrams(['eat','tea','tan','ate','nat','bat'])==[['ate','eat','tea'],['bat'],['nat','tan']]"),
    ("merge_intervals(iv)", "merges overlapping [start, end] intervals (lists) and returns them sorted by start",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]"),
    ("primes_upto(n)", "returns the list of primes <= n", "assert primes_upto(30)==[2,3,5,7,11,13,17,19,23,29]\nassert primes_upto(1)==[]"),
    ("rle(s)", "run-length encodes a string as character followed by count, e.g. 'aaab' -> 'a3b1'",
     "assert rle('aaabccdddd')=='a3b1c2d4' and rle('')==''"),
    ("balanced(s)", "returns True if the brackets ()[]{} in s are balanced (other characters ignored)",
     "assert balanced('{[()()]}x') and not balanced('([)]') and not balanced('((')"),
    ("two_sum(nums, target)", "returns the indices [i, j] with i < j of the two numbers adding to target (exactly one "
     "solution exists)", "assert two_sum([2,7,11,15],9)==[0,1] and two_sum([3,2,4],6)==[1,2]"),
    ("word_freq(text)", "returns a dict of lowercase word -> count, words being runs of letters",
     "assert word_freq('The cat and the hat.')=={'the':2,'cat':1,'and':1,'hat':1}"),
    ("transpose(m)", "transposes a rectangular matrix given as a list of lists",
     "assert transpose([[1,2,3],[4,5,6]])==[[1,4],[2,5],[3,6]]"),
    ("binary_search(xs, x)", "returns the index of x in the sorted list xs, or -1",
     "assert binary_search([1,3,5,7,9],7)==3 and binary_search([1,3,5],4)==-1 and binary_search([],1)==-1"),
    ("lcs(a, b)", "returns the length of the longest common subsequence of strings a and b",
     "assert lcs('ABCBDAB','BDCABA')==4 and lcs('','x')==0"),
    ("to_base(n, b)", "converts a non-negative integer n to a string in base b (2..16, digits 0-9a-f)",
     "assert to_base(255,16)=='ff' and to_base(0,2)=='0' and to_base(10,2)=='1010'"),
    ("caesar(s, k)", "shifts letters by k positions (wrapping, keeping case), other characters unchanged",
     "assert caesar('Hello, World!',3)=='Khoor, Zruog!' and caesar('abc',-1)=='zab'"),
    ("dedupe(xs)", "removes duplicates keeping the first occurrence order",
     "assert dedupe([3,1,3,2,1])==[3,1,2]"),
    ("chunks(xs, n)", "splits a list into consecutive chunks of size n (the last may be shorter)",
     "assert chunks([1,2,3,4,5],2)==[[1,2],[3,4],[5]] and chunks([],3)==[]"),
    ("median(xs)", "returns the median of a non-empty list of numbers (average of the middle two for even length)",
     "assert median([3,1,2])==2 and median([4,1,3,2])==2.5"),
    ("count_islands(grid)", "counts 4-connected groups of 1s in a grid (list of lists of 0/1)",
     "assert count_islands([[1,1,0,0],[0,1,0,1],[1,0,0,1]])==3 and count_islands([])==0"),
    ("matrix_mult(a, b)", "multiplies two matrices given as lists of lists",
     "assert matrix_mult([[1,2],[3,4]],[[5,6],[7,8]])==[[19,22],[43,50]]"),
    ("is_valid_ipv4(s)", "returns True for dotted-quad IPv4 addresses with parts 0..255 and no leading zeros",
     "assert is_valid_ipv4('192.168.0.1') and not is_valid_ipv4('256.1.1.1') and not is_valid_ipv4('01.2.3.4') and not is_valid_ipv4('1.2.3')"),
    ("longest_word(s)", "returns the longest word (split on whitespace), the first one on ties",
     "assert longest_word('a bb ccc dd eee')=='ccc'"),
    ("digit_sum(n)", "returns the repeated digit sum of a non-negative integer until one digit remains",
     "assert digit_sum(9875)==2 and digit_sum(0)==0"),
    ("spiral(m)", "returns the elements of a matrix in clockwise spiral order",
     "assert spiral([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5] and spiral([])==[]"),
]


def code_items() -> list[dict]:
    return [{"id": f"c{i}", "q": f"Write a Python function `{sig}` that {desc}. Reply with one ```python code block "
             f"containing only the function (and any imports it needs), no explanation, no tests.", "tests": tests}
            for i, (sig, desc, tests) in enumerate(CODE)]


def code_score(it: dict, text: str) -> bool:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    src = (m[0] if m else text) + "\n\n" + it["tests"] + "\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text(src)
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=d, capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


# -- long -------------------------------------------------------------------------------------------------------------
ANIMALS = ["falcon", "otter", "bison", "lynx", "heron", "badger", "marmot", "ibis", "walrus", "gecko", "puffin",
           "tapir", "koala", "oriole", "viper", "egret", "jackal", "newt", "osprey", "quokka"]


def long_item(tokens: int, seed: int, keys: int = 16, updates: int = 4) -> dict:
    rng = random.Random(seed)
    names = rng.sample(ANIMALS, keys)
    first = {nm: str(rng.randint(10000, 99999)) for nm in names}
    lines = [" ".join(rng.choice(WORDS) for _ in range(12)) + "." for _ in range(int(tokens * 0.72) // 12)]
    for i, nm in enumerate(names):                         # first values in the first 70%
        lines.insert(int(len(lines) * 0.7 * (i + 0.5) / keys), f"Remember this: the code for {nm} is {first[nm]}.")
    final = dict(first)
    for j, nm in enumerate(rng.sample(names, updates)):    # overwritten later, in the last 30%
        final[nm] = str(rng.randint(10000, 99999))
        lines.insert(int(len(lines) * (0.72 + 0.27 * (j + 0.5) / updates)),
                     f"Correction: the code for {nm} has changed, it is now {final[nm]}.")
    ask = ("\n\nSome codes were corrected later in the text; use the latest value for each. List the current code "
           "for each of these, one per line as 'name: code': " + ", ".join(names) + ".")
    return {"prompt": " ".join(lines) + ask, "final": final, "first": first}


def long_score(it: dict, text: str) -> dict:
    t = text.lower()
    got = {}
    for nm in it["final"]:
        m = re.search(rf"{nm}\W+(\d{{5}})", t)
        got[nm] = m.group(1) if m else None
    ok = sum(got[nm] == v for nm, v in it["final"].items())
    stale = sum(got[nm] == it["first"][nm] != it["final"][nm] for nm in it["final"])
    return {"found": ok, "keys": len(it["final"]), "stale": stale}


LONG_SIZES = [(32768, 3), (131072, 2), (262144, 2)]

# -- gen --------------------------------------------------------------------------------------------------------------
GEN = ["Explain how a hash table handles collisions, with an example.",
       "Write a short story about a lighthouse keeper who finds a message in a bottle.",
       "Describe the water cycle to a ten-year-old.",
       "Write a Python class implementing an LRU cache with get and put, and explain it.",
       "What are the trade-offs between microservices and a monolith?",
       "Summarize the causes of the French Revolution.",
       "Write a bash script that finds the ten largest files under a directory.",
       "Explain gradient descent and the role of the learning rate.",
       "Write a haiku sequence about the four seasons.",
       "How does TCP congestion control work?",
       "Compare quicksort and mergesort.",
       "Write a SQL query to find the second highest salary per department, and explain it.",
       "Explain what a transformer's attention mechanism computes.",
       "Plan a three-day trip to Kyoto.",
       "Write a Rust function that parses a CSV line with quoted fields.",
       "Why is the sky blue?",
       "Describe how public-key cryptography works.",
       "Write a limerick about a cat who learns to code.",
       "Explain the difference between processes and threads.",
       "What is the Monty Hall problem and why is switching better?"]


# -- run / compare ----------------------------------------------------------------------------------------------------
def run(label: str, only: set[str], conc: int) -> None:
    path = OUT / f"{label}.json"
    res = json.loads(path.read_text()) if path.exists() else {"label": label}
    res.update({"model": model_id(), "url": URL})

    def save():
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1))

    def batch(name, items, ask, score):
        t0 = time.time()
        with cf.ThreadPoolExecutor(conc) as ex:
            replies = list(ex.map(ask, items))
        out = []
        for it, r in zip(items, replies):
            out.append({"id": it["id"], **r, "ok": score(it, r["text"])})
        res[name] = out
        save()
        n = sum(o["ok"] for o in out)
        print(f"{label} {name}: {n}/{len(out)} in {time.time() - t0:.0f}s", flush=True)

    if "qa" in only:
        batch("qa", qa_items(), lambda it: chat(it["q"], it["mt"]), qa_score)
    if "reason" in only:
        batch("reason", reason_items(), lambda it: chat(it["q"], 8192, thinking=True), reason_score)
    if "arith" in only:
        batch("arith", arith_items(), lambda it: chat(it["q"], 12288, thinking=True), answer_score)
    if "track" in only:
        batch("track", track_items(), lambda it: chat(it["q"], 12288, thinking=True), answer_score)
    if "code" in only:
        batch("code", code_items(), lambda it: chat(it["q"], 1024), code_score)
    if "gen" in only:
        items = [{"id": f"g{i}", "q": q} for i, q in enumerate(GEN)]
        batch("gen", items, lambda it: chat(it["q"], 400), lambda it, t: True)
    if "long" in only:
        out = []
        for size, n in LONG_SIZES:
            for k in range(n):
                it = long_item(size, seed=7000 + 97 * size + k)
                r = chat(it["prompt"], 400)
                sc = long_score(it, r["text"])
                out.append({"id": f"L{size}-{k}", "size": size, **r, **sc, "ok": sc["found"] == sc["keys"]})
                res["long"] = out
                save()
                print(f"{label} long {size} #{k}: {sc['found']}/{sc['keys']} (stale {sc['stale']}), "
                      f"{r['prompt_tokens']} tok, {r['s']:.0f}s", flush=True)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(a: str, b: str) -> None:
    A = json.loads((OUT / f"{a}.json").read_text())
    B = json.loads((OUT / f"{b}.json").read_text())
    print(f"{'category':>8} | {a:>12} | {b:>12} | identical replies | only {a} right | only {b} right | McNemar p")
    for cat in ["qa", "reason", "arith", "track", "code", "long"]:
        if cat not in A or cat not in B:
            continue
        x, y = {o["id"]: o for o in A[cat]}, {o["id"]: o for o in B[cat]}
        ids = [i for i in x if i in y]
        na, nb = sum(x[i]["ok"] for i in ids), sum(y[i]["ok"] for i in ids)
        same = sum(x[i]["text"] == y[i]["text"] for i in ids)
        oa = sum(x[i]["ok"] and not y[i]["ok"] for i in ids)
        ob = sum(y[i]["ok"] and not x[i]["ok"] for i in ids)
        extra = ""
        if cat == "long":
            ka, kb = sum(x[i]["found"] for i in ids), sum(y[i]["found"] for i in ids)
            kt = sum(x[i]["keys"] for i in ids)
            extra = f"  keys {ka}/{kt} vs {kb}/{kt}"
        print(f"{cat:>8} | {na:>5}/{len(ids):<6} | {nb:>5}/{len(ids):<6} | {same:>17} | {oa:>13} | {ob:>13} | "
              f"{mcnemar(oa, ob):.3f}{extra}")
    if "gen" in A and "gen" in B:
        pre, same = [], 0
        for x, y in zip(A["gen"], B["gen"]):
            s, t = x["text"], y["text"]
            n = 0
            while n < min(len(s), len(t)) and s[n] == t[n]:
                n += 1
            pre.append(n / max(1, len(s)))
            same += s == t
        print(f"     gen | identical {same}/{len(pre)}, mean common prefix {sum(pre) / len(pre):.2f} of the reply")
    if "reason" in A and "reason" in B:
        ta = sum(o["tokens"] or 0 for o in A["reason"]) / len(A["reason"])
        tb = sum(o["tokens"] or 0 for o in B["reason"]) / len(B["reason"])
        fa = sum(o["finish"] == "length" for o in A["reason"])
        fb = sum(o["finish"] == "length" for o in B["reason"])
        print(f"  reason | mean tokens {ta:.0f} vs {tb:.0f}; cut at max_tokens {fa} vs {fb}")


def show(label: str) -> None:
    d = json.loads((OUT / f"{label}.json").read_text())
    print(f"{label}: {d.get('model')} at {d.get('url')}")
    pct = []
    for cat in ["qa", "reason", "arith", "track", "code", "long"]:
        if cat in d:
            n, t = sum(o["ok"] for o in d[cat]), len(d[cat])
            pct.append(100 * n / t)
            extra = ""
            if cat == "long":
                extra = (f", keys {sum(o['found'] for o in d[cat])}/{sum(o['keys'] for o in d[cat])}, "
                         f"stale {sum(o['stale'] for o in d[cat])}")
            print(f"  {cat:>6}: {n}/{t} ({100 * n / t:.1f}%){extra}")
    if pct:
        print(f"  overall (mean of categories): {sum(pct) / len(pct):.1f}%")


def _opt(args: list[str], name: str, default=None):
    return args[args.index(name) + 1] if name in args else default


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[3:]
    if cmd == "run":
        URL = _opt(args, "--url", os.environ.get("QUALITY_URL", URL))
        MODEL = _opt(args, "--model")
        API_KEY = _opt(args, "--api-key", API_KEY)
        if "--long-sizes" in args:
            sizes = [int(x) for x in _opt(args, "--long-sizes").split(",")]
            LONG_SIZES = [(s, int(_opt(args, "--long-items", 2))) for s in sizes]
        only = ({"qa", "code"} if "--quick" in args else {"qa", "reason", "arith", "track", "code", "gen"})
        if "--only" in args:
            only = set(_opt(args, "--only").split(","))
        run(sys.argv[2], only, int(_opt(args, "--conc", 4)))
        show(sys.argv[2])
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif cmd == "show":
        show(sys.argv[2])
    else:
        sys.exit(__doc__)
