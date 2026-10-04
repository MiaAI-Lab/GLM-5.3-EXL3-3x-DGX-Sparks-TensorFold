one expert call a chunk) and 0135's smaller first/last CP parts, 499,712-token window | 448 tok/s at 9.9k, 434 at 94k tokens (needles correct); exact 12/12 | cp500b (`PREFILL_ROWS=3072`, `TF_GLM_CP_RAW_Q=0`); with raw queries on (cp500a): 386 / 436 |
| Long-context prefill with 0131 (no copies, exchanges overlapped on a side stream, candidates merged by row owners) | 378 tok/s at 9.9k, 410 at 94k tokens (needles correct); exact 12/12 | fp4g627pipe; `TF_GLM_CP_PIPE=0` gives back 0129's path |
| Long-context prefill with 0129 (context-parallel prompt attention through our MSA kernel) | 294 tok/s at 9.9k, 297 at 94k, 276 at 314k tokens (needles correct; was 244 / 236 / 220); exact 12/12 | fp4g627p; spark3 lowest 5.41 GiB |
| Long-context mode with 0128 (expert kernels: L2 prefetch, dependent launches), 626,688-token window | greedy: prose 31.7-32.5, code 41.8-42.3 tok/s (3 runs); sampled: prose 22.9-24.0, code 25.7-29.4; exact 12/12 | fp4g627x; the same reply hashes |
| Long-context mode with 0125 (faster draft selection), 626,688-token window | greedy: prose 30.1-31.8, code 39.9-41.3 tok/s (3 runs); sampled (T 1.0, top_p 0.95): prose 22.0-23.4, code 25.1-28.7 | fp4g627m; same reply hashes as before 0125 |

Sampled replies (temperature 1.0, top_p 0.95: a chat client's default) decode slower than greedy ones: fewer drafts
survive the target's sampled token (~2 tokens a round on prose against ~3-4 greedy). sparkDash's decode bench is
greedy; its live gauge on a chat shows the sampled rate.

Decode speed depends on the text: speculative drafts land more often on predictable text. On worked arithmetic
(thinking on) the drafter's acceptance was 87% and a request averaged ~45 tok/s, with bursts above 70.

### Concurrent requests (0134)

`PARALLEL=2` (to 4; not with `CP=1`) decodes that many requests together, each with its own `CONTEXT` window. Boot par2
(`PARALLEL=2 CONTEXT=65536`, FP8 KV, eager batched verify, no graphs yet): exact 12/12 with requests really concurrent
(a request's reply hash alone equals its hash beside another); greedy decode 26.9 tok/s for one request, 35.5 / 35.9
tok/s together for two (+33%; 17.7-19.1 each).

### Prompt cache on NVMe (0132)

With `TF_GLM_DISK_CACHE=<dir>` every kept prompt state is also written, in the background, to each Spark's own NVMe
(each rank its own rows; checksummed; a size budget, 64 GiB by default, never leaving under 100 GB free). A later
request whose prompt starts with a saved state resumes from it, also after a restart. Boots pcacheA / pcacheB (fp4,
`CP=1`): a 94,317-token needle prefilled in 236.1 s and saved 1.26 GB a Spark; after a restart the same request
resumed from disk in 3.0 s (prefill 0.002 s), answer correct; exact 12/12.

### Exactness and long context

- Drafted and concurrent replies equal serial ones: 12/12 on every boot listed here (`tools/exact.py`).
- Needle (`tools/needle.py`): correct at 9.9k, 94k and 314k tokens (fp4g655); to 235,660 tokens (p19, `CP=1`, FP8 KV,
  450k window).

