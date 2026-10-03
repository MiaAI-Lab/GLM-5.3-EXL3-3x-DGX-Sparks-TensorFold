"""The startup memory plan of every rank, offline: TensorFold's own estimate (the patched engine's weight transform and
full GLM-5.3 geometry, from the checkpoint's headers and config only; no GPU, no weights read) for each rank of a
TP=N start, against each Spark's MemAvailable, with what TensorFold does not count (the CUDA context, NCCL, compiled
kernels, the Python process: OVERHEAD_GIB, measured at the first boots) on top.

    python memplan.py <checkpoint folder> --tp 3 --context 131072 --kv bf16 --dense q4 --mtp 1 \
        --avail 0=114.2,1=115.0,2=116.1 [--floor 10] [--overhead 6] [--reserve 16] [--json]

Prints a table a rank, and exits 1 (naming the rank and the shortfall) when a rank's lowest free memory would fall
below --floor GiB, or TensorFold's own admission (MemAvailable - TENSORFOLD_MEMORY_RESERVE_GIB) would refuse the
window. scripts/start.sh runs it in the image before it starts anything. --avail empty: the plan without the check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

GIB = 2 ** 30


def plan(model_dir: Path, tp: int, context: int, kv: str, dense: str, mtp: bool, prefill_rows: int = 2048,
         max_rows: int = 16, mtp_rows: int = 8, split: bool = False, cp: int = 1) -> list[dict]:
    os.environ["TF_GLM_DENSE"] = dense
    from tensorfold.cuda.capacity import config, estimate_weights
    from tensorfold.cuda.geometry import mla_full_geometry, full_token_bytes
    from tensorfold.families.glm5_next.cuda import tp as tp_mod
    from tensorfold.families.glm5_next.cuda.engine import DENSE_CAPACITY, full_weights, without_mtp
    from tensorfold.families.glm5_next.cuda.qmm import fp8_weights, q4_weights
    from tensorfold.families.glm5_next.cuda.split import rank_cut, rule
    from tensorfold.families.glm5_next.cuda.weights import Config
    from tensorfold.cuda.geometry import split_weights

    if split:                            # the row split's pad rows (hcsplit.buffer_rows: one at three ranks)
        prefill_rows += -prefill_rows % tp
    cfg = Config.read(model_dir)
    if not cfg.full:
        raise SystemExit(f"{model_dir}: not a full GLM-5.3 checkpoint (model_type glm_moe_dsa)")
    text = config(model_dir)
    out = []
    for rank in range(tp):
        lay = tp_mod.Layout.from_config(cfg, rank, tp)

        def cut(name, shape, kind, lay=lay):
            if kind == "vocab":
                return lay.vocab_offset, lay.vocab_offset + lay.vocab
            return rank_cut(lay, name, shape, kind)

        share = lambda family, total, lay=lay: lay.size(family) if family in lay.totals else int(total) // tp  # noqa
        tr = split_weights(rule, tp, cut)
        tr = q4_weights(tr) if dense == "q4" else fp8_weights(tr) if dense == "fp8" else tr
        tr = full_weights(tr, cfg, lay)
        if not mtp:
            tr = without_mtp(tr, cfg.layers, cfg.prefix)
        w = estimate_weights(model_dir, tr)
        g = mla_full_geometry(text, tp, max_rows, share=share, decode_rows=max_rows, mtp_rows=min(mtp_rows, max_rows),
                              prefill_rows=prefill_rows, kv=kv, mtp=mtp, prompt_split_k=dense != "q4",
                              minimum_slots=DENSE_CAPACITY, **({"cp": cp} if cp > 1 else {}))
        slots = max(DENSE_CAPACITY, context + max_rows)
        local = -(-slots // cp) + (1 if cp > 1 else 0)          # context parallelism: every cp-th token a rank
        caches = local * full_token_bytes(text, kv, mtp)
        geometry = g.needed(context)
        out.append({"rank": rank, "weights": w.resident, "staging": w.staging, "geometry": geometry,
                    "caches": caches, "buffers": geometry - caches, "slots": slots,
                    "serving": w.resident + geometry, "startup_peak": w.resident + w.staging,
                    "total": w.resident + max(w.staging, geometry)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--tp", type=int, default=3)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--kv", default="bf16", choices=("bf16", "fp8", "fp4"))
    ap.add_argument("--dense", default="q4", choices=("bf16", "fp8", "q4"))
    ap.add_argument("--mtp", type=int, default=1, choices=(0, 1))
    ap.add_argument("--prefill-rows", type=int, default=2048)
    ap.add_argument("--split", type=int, default=0, choices=(0, 1), help="TF_GLM_HC_SPLIT (the row split's pad rows)")
    ap.add_argument("--cp", type=int, default=1, help="TF_GLM_CP: ranks the token caches are split over (1: none)")
    ap.add_argument("--kept-gib", type=float, default=0.0, help="TF_GLM_CACHE_GIB: kept prompts' saved rows")
    ap.add_argument("--avail", default="", help="rank=GiB MemAvailable at start, comma separated")
    ap.add_argument("--floor", type=float, default=10.0, help="the lowest free GiB a rank may reach")
    ap.add_argument("--overhead", type=float, default=6.0, help="GiB a rank uses past TensorFold's estimate")
    ap.add_argument("--reserve", type=float, default=16.0, help="TENSORFOLD_MEMORY_RESERVE_GIB")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.floor < 4:
        print("memplan: --floor below 4 GiB is not allowed (a GB10 that runs out of memory freezes)", file=sys.stderr)
        return 2
    avail = {}
    for item in filter(None, a.avail.split(",")):
        r, v = item.split("=")
        avail[int(r)] = float(v)
    rows = plan(a.model_dir, a.tp, a.context, a.kv, a.dense, bool(a.mtp), a.prefill_rows, split=bool(a.split),
                cp=a.cp)
    bad = []
    for row in rows:
        row["kept"] = int(a.kept_gib * GIB)
        row["need"] = row["total"] + row["kept"] + int(a.overhead * GIB)
        if row["rank"] in avail:
            have = avail[row["rank"]] * GIB
            row["avail"] = have
            row["lowest_free"] = have - row["need"]
            # TensorFold's own admission: weights + caches/buffers within MemAvailable - its reserve
            row["admits"] = row["total"] + row["kept"] <= have - a.reserve * GIB
            if row["lowest_free"] < a.floor * GIB:
                bad.append(f"rank {row['rank']}: lowest free {row['lowest_free'] / GIB:.1f} GiB < floor {a.floor:g}")
            if not row["admits"]:
                bad.append(f"rank {row['rank']}: TensorFold would refuse (needs {(row['total'] + row['kept']) / GIB:.1f}"
                           f" GiB within {avail[row['rank']]:.1f} - reserve {a.reserve:g})")
    if a.json:
        print(json.dumps({"settings": vars(a) | {"model_dir": str(a.model_dir)}, "ranks": rows, "refused": bad},
                         default=str))
    else:
        g = lambda x: f"{x / GIB:7.2f}"   # noqa: E731
        print(f"memplan: TP={a.tp}{f', CP={a.cp}' if a.cp > 1 else ''}, window {a.context}, KV {a.kv}, dense {a.dense}, "
              f"MTP {'on' if a.mtp else 'off'}, "
              f"prefill rows {a.prefill_rows}, overhead {a.overhead:g} GiB, floor {a.floor:g} GiB (GiB below)")
        print("rank  weights  buffers   caches  staging  TF-total  +kept+ovh  avail  lowest-free")
        for row in rows:
            extra = (f"  {row['avail'] / GIB:6.1f}  {row['lowest_free'] / GIB:7.1f}" if "avail" in row else "")
            print(f"{row['rank']:4d}  {g(row['weights'])}  {g(row['buffers'])}  {g(row['caches'])}  {g(row['staging'])}"
                  f"  {g(row['total'])}  {g(row['need'])}{extra}")
        for line in bad:
            print("memplan: REFUSED: " + line, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:                    # noqa: BLE001  a crash is not a refusal (exit 1): start.sh tells them apart
        import traceback

        traceback.print_exc()
        print(f"memplan: FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
