"""Summarise LeRobot LIBERO eval_info.json files next to published numbers.

Published references (success %, 10 trials per task):
  SmolVLA (0.45B): SmolVLA paper, arXiv 2506.01844, Table 2.
  pi0.5 (LeRobot port, finetuned): LeRobot docs "Reproducing pi0.5 results" (libero.mdx);
  OpenPI reference numbers are quoted in the same page.
"""
import argparse
import json
from pathlib import Path

from rebot_sim_rl.layout import wilson

PUBLISHED = {
    "smolvla": {"source": "SmolVLA paper (arXiv 2506.01844) Table 2, SmolVLA 0.45B",
                "libero_spatial": 90, "libero_object": 96, "libero_goal": 92, "libero_10": 71},
    "pi05": {"source": "LeRobot docs, pi0.5 LIBERO reproduction (lerobot/pi05_libero_finetuned)",
             "libero_spatial": 97.0, "libero_object": 99.0, "libero_goal": 98.0, "libero_10": 96.0},
}
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/libero"))
    ap.add_argument("--out", type=Path, default=Path("results/libero/summary.md"))
    args = ap.parse_args()
    rows, js = [], {}
    for model in ("smolvla", "pi05"):
        for s in SUITES:
            f = args.dir / f"{model}_{s}.eval_info.json"
            shards = sorted(args.dir.glob(f"{model}_{s}_shard*.eval_info.json"))
            if f.exists():
                d = json.loads(f.read_text())
            elif shards:  # the suite was split by task id over several jobs
                d = {"per_task": [t for sh in shards for t in json.loads(sh.read_text())["per_task"]]}
                d["per_task"].sort(key=lambda t: t["task_id"])
            else:
                rows.append(f"| {model} | {s} | not run / pending | | {PUBLISHED[model][s]} |")
                continue
            k = sum(sum(t["metrics"]["successes"]) for t in d["per_task"])
            n = sum(len(t["metrics"]["successes"]) for t in d["per_task"])
            lo, hi = wilson(k, n)
            per_task = [sum(t["metrics"]["successes"]) for t in d["per_task"]]
            js[f"{model}/{s}"] = {"successes": k, "episodes": n, "wilson95": [round(lo, 3), round(hi, 3)],
                                  "per_task_successes": per_task, "published_pct": PUBLISHED[model][s]}
            rows.append(f"| {model} | {s} | {k}/{n} ({100 * k / n:.0f}%) | {100 * lo:.0f}–{100 * hi:.0f}% | "
                        f"{PUBLISHED[model][s]} |")
    md = ["| policy | suite | ours n/N | 95% Wilson | published % |", "|---|---|---|---|---|", *rows, "",
          *[f"- {m}: {PUBLISHED[m]['source']}" for m in PUBLISHED]]
    args.out.write_text("\n".join(md) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(js, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
