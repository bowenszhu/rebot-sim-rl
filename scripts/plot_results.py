"""Aggregate training runs and draw the learning curves (return and success kept separate).

  python scripts/plot_results.py --group "PPO from scratch=runs/main/ppo_scratch_s*" \
      --group "BC(10 demos) -> PPO=runs/main/bcppo_n10_s*" --bc-ref runs/main/bc_n10_s* \
      --out results/figures

Training curves come from the SB3 Monitor files (stochastic policy, training start
distribution); evaluation curves from eval_curve.csv (deterministic policy, the fixed
54-episode T1 start set). Seeds are shown as thin lines, the mean as a thick line.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical slots 1-3 (light)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, xlabel, ylabel):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def monitor_curve(run: Path, bin_steps: int):
    frames = []
    for f in sorted((run / "monitor").glob("*.monitor.csv*")):
        df = pd.read_csv(f, skiprows=1)
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames).sort_values("t")
    df["steps"] = df["l"].cumsum()
    df["bin"] = (df["steps"] // bin_steps + 1) * bin_steps
    g = df.groupby("bin").agg(ret=("r", "mean"), succ=("is_success", "mean"))
    return g[g.index <= df["steps"].iloc[-1]]  # drop the last, incomplete bin


def load_group(pattern, bin_steps):
    runs = sorted(Path(p) for p in glob.glob(pattern))
    out = []
    for r in runs:
        ev = pd.read_csv(r / "eval_curve.csv") if (r / "eval_curve.csv").exists() else None
        out.append({"run": r, "eval": ev, "train": monitor_curve(r, bin_steps)})
    return out


def plot_lines(ax, groups, getter, color_of):
    for name, runs in groups.items():
        c = color_of[name]
        xs, ys = [], []
        for r in runs:
            x, y = getter(r)
            if x is None:
                continue
            ax.plot(x, y, color=c, linewidth=0.8, alpha=0.35)
            xs.append(np.asarray(x))
            ys.append(np.asarray(y))
        if not ys:
            continue
        n = min(len(y) for y in ys)
        mean = np.mean([y[:n] for y in ys], axis=0)
        ax.plot(xs[0][:n], mean, color=c, linewidth=2.0, label=f"{name} ({len(ys)} seed{'s' if len(ys) > 1 else ''}{', mean' if len(ys) > 1 else ''})")
        ax.annotate(f"{mean[-1]:.2f}", (xs[0][n - 1], mean[-1]), xytext=(4, 0),
                    textcoords="offset points", color=INK2, fontsize=8, va="center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help="label=glob")
    ap.add_argument("--bc-ref", default=None, help="glob of BC run dirs (eval.json) drawn as a reference line")
    ap.add_argument("--bc-label", default="BC alone")
    ap.add_argument("--bin-steps", type=int, default=100_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for g in args.group:
        label, pat = g.rsplit("=", 1)
        groups[label] = load_group(pat, args.bin_steps)
    color_of = {k: SERIES[i % len(SERIES)] for i, k in enumerate(groups)}
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})

    bc_rate = None
    if args.bc_ref:
        rates = [json.loads((Path(p) / "eval.json").read_text())["summary"]["rate"] for p in glob.glob(args.bc_ref)]
        bc_rate = (float(np.mean(rates)), len(rates)) if rates else None

    figs = {
        "eval_success": ("Fixed-seed evaluation success (54 T1 starts, deterministic)", "success rate",
                         lambda r: (None, None) if r["eval"] is None else (r["eval"]["timesteps"], r["eval"]["rate"])),
        "train_return": ("Training episode return (stochastic policy)", "mean episode return",
                         lambda r: (None, None) if r["train"] is None else (r["train"].index, r["train"]["ret"])),
        "train_success": ("Training episode success (stochastic policy)", "success rate",
                          lambda r: (None, None) if r["train"] is None else (r["train"].index, r["train"]["succ"])),
    }
    for key, (title, ylabel, getter) in figs.items():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        plot_lines(ax, groups, getter, color_of)
        if key == "eval_success" and bc_rate is not None:
            ax.axhline(bc_rate[0], color=INK2, linewidth=1.2, linestyle="--",
                       label=f"{args.bc_label} ({bc_rate[1]} seeds, mean)")
        if "success" in key:
            ax.set_ylim(-0.02, 1.02)
        style(ax, "environment steps", ylabel)
        ax.set_title(title, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(args.out / f"{args.prefix}{key}.png")
        plt.close(fig)
    summary = {}
    for name, runs in groups.items():
        finals = [r["eval"].iloc[-1].to_dict() for r in runs if r["eval"] is not None]
        summary[name] = {"runs": [str(r["run"]) for r in runs], "final_eval": finals}
    (args.out / f"{args.prefix}curves_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float)[:2000])


if __name__ == "__main__":
    main()
