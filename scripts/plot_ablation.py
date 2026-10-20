"""Single-factor ablation: success vs evaluation-time cube-position noise.

Reads results/eval/<group>_s<seed>.json written by scripts/evaluate.py and pools the
seeds of each group (same 54 starts and noise sequences per setting).
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rebot_sim_rl.layout import wilson  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help="label=glob of eval json files")
    ap.add_argument("--out", type=Path, default=Path("results/figures/ablation_obs_noise.png"))
    args = ap.parse_args()
    table = {}
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for gi, g in enumerate(args.group):
        label, pat = g.rsplit("=", 1)
        files = sorted(glob.glob(pat))
        agg = {}
        per_seed = {}
        for f in files:
            d = json.loads(Path(f).read_text())
            for st in d["settings"]:
                if "friction" in st:
                    continue
                key = st["obs_noise"]
                k, n = st["summary"]["successes"], st["summary"]["episodes"]
                a = agg.setdefault(key, [0, 0])
                a[0] += k
                a[1] += n
                per_seed.setdefault(key, []).append(k / n)
        xs = sorted(agg)
        rate = [agg[x][0] / agg[x][1] for x in xs]
        ci = [wilson(*agg[x]) for x in xs]
        c = SERIES[gi % len(SERIES)]
        off = (gi - (len(args.group) - 1) / 2) * 0.06
        xcm = np.array(xs) * 100 + off
        ax.plot(xcm, rate, "-o", color=c, linewidth=2, markersize=6, markeredgecolor="white",
                label=f"{label} ({len(files)} seeds pooled)")
        for x, (lo, hi) in zip(xcm, ci):
            ax.plot([x, x], [lo, hi], color=c, linewidth=1.2, alpha=0.6)
        for x, xv in zip(xcm, xs):
            ax.annotate(f"{agg[xv][0]}/{agg[xv][1]}", (x, agg[xv][0] / agg[xv][1]), xytext=(6, -3 if gi else 6),
                        textcoords="offset points", color=INK2, fontsize=7)
        table[label] = {f"{x * 100:g}cm": {"successes": agg[x][0], "episodes": agg[x][1],
                                           "wilson95": [round(v, 3) for v in wilson(*agg[x])],
                                           "per_seed_rate": [round(r, 3) for r in per_seed[x]]} for x in xs}
    ax.set_xlabel("cube-position observation noise at evaluation, σ (cm)", color=INK2)
    ax.set_ylabel("success rate", color=INK2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Robustness to perception noise: train σ = 0 vs 1 cm", color=INK, loc="left", fontsize=9)
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
