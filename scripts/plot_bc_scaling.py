"""BC success vs number of scripted demonstrations (fixed 54-episode eval, Wilson 95% CI)."""
import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rebot_sim_rl.layout import wilson  # noqa: E402

BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/main/bc_n*_s*")
    ap.add_argument("--out", type=Path, default=Path("results/figures/bc_scaling.png"))
    args = ap.parse_args()
    rows = []
    for d in glob.glob(args.runs):
        m = re.search(r"bc_n(\d+)_s(\d+)", d)
        s = json.loads((Path(d) / "eval.json").read_text())["summary"]
        rows.append({"n_demos": int(m.group(1)), "seed": int(m.group(2)), "k": s["successes"], "N": s["episodes"],
                     "stages": s["max_stage_hist"]})
    rows.sort(key=lambda r: (r["n_demos"], r["seed"]))
    ns = sorted({r["n_demos"] for r in rows})
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    pooled = []
    for i, n in enumerate(ns):
        rs = [r for r in rows if r["n_demos"] == n]
        for j, r in enumerate(rs):
            lo, hi = wilson(r["k"], r["N"])
            x = i + (j - 1) * 0.12
            ax.plot([x, x], [lo, hi], color=BLUE, linewidth=1.2, alpha=0.5)
            ax.plot(x, r["k"] / r["N"], "o", color=BLUE, markersize=5, markeredgecolor="white", markeredgewidth=1)
        k, N = sum(r["k"] for r in rs), sum(r["N"] for r in rs)
        pooled.append({"n_demos": n, "successes": k, "episodes": N, "rate": k / N,
                       "wilson95": [round(v, 3) for v in wilson(k, N)]})
        ax.annotate(f"{k}/{N}", (i, max(r["k"] / r["N"] for r in rs)), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=INK2, fontsize=8)
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("scripted demonstrations used for BC", color=INK2)
    ax.set_ylabel("success rate (54 fixed starts)", color=INK2)
    ax.set_ylim(-0.03, 1.12)
    ax.set_title("Behaviour cloning vs demo count (3 training seeds, 95% Wilson CI)", color=INK, loc="left", fontsize=9)
    ax.grid(True, axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    (args.out.with_suffix(".json")).write_text(json.dumps({"per_run": rows, "pooled": pooled}, indent=1))
    print(json.dumps(pooled, indent=1))


if __name__ == "__main__":
    main()
