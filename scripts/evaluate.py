"""Fixed-seed evaluation of a saved policy, optionally under perturbations, with videos.

  # nominal + robustness sweep over cube-position observation noise
  python scripts/evaluate.py --model runs/x/model_last.zip --obs-noise 0 0.005 0.01 0.02 \
      --out results/eval/x.json
  # save failure (and a few success) videos
  python scripts/evaluate.py --model ... --videos results/videos/x --max-fail-videos 4

Every setting uses the same 54 deterministic start states (9 grid cells x yaw {0, 45}
x 3 placement jitters, seed 12345); the observation noise uses the env seed of each
episode, so all policies see identical noise sequences.
"""
import argparse
import json
from pathlib import Path

import rebot_sim_rl  # noqa: F401
import imageio
from stable_baselines3 import PPO, SAC

from rebot_sim_rl.env import B601T1Env
from rebot_sim_rl.evaluation import EVAL_REPS, run_eval, summarize


def load(path: Path):
    for cls in (PPO, SAC):
        try:
            return cls.load(path, device="cpu")
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"cannot load {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--obs-noise", type=float, nargs="+", default=[0.0])
    ap.add_argument("--friction", type=float, nargs="*", default=None,
                    help="optional cube friction values to sweep at zero obs noise")
    ap.add_argument("--reps", type=int, default=EVAL_REPS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--videos", type=Path, default=None)
    ap.add_argument("--max-fail-videos", type=int, default=4)
    ap.add_argument("--max-success-videos", type=int, default=1)
    args = ap.parse_args()
    model = load(args.model)
    policy = lambda o: model.predict(o, deterministic=True)[0]  # noqa: E731
    results = {"model": str(args.model), "settings": []}
    settings = [{"obs_noise": n} for n in args.obs_noise]
    if args.friction:
        settings += [{"obs_noise": 0.0, "friction": f} for f in args.friction]
    for st in settings:
        env = B601T1Env(obs_noise_cube=st["obs_noise"], render_mode="rgb_array" if args.videos else None)
        if "friction" in st:
            env.base_friction = st["friction"]
        saved = {"fail": 0, "success": 0}

        def record(i, row, frames, st=st, saved=saved):
            kind = "success" if row["success"] else "fail"
            limit = args.max_success_videos if row["success"] else args.max_fail_videos
            if saved[kind] >= limit:
                return
            saved[kind] += 1
            tag = f"noise{st['obs_noise']:g}" + (f"_mu{st['friction']:g}" if "friction" in st else "")
            args.videos.mkdir(parents=True, exist_ok=True)
            fn = args.videos / f"{tag}_{kind}_{row['cell']}_yaw{row['yaw_deg']:.0f}_ep{i:02d}_stage{row['max_stage']}.mp4"
            imageio.mimsave(fn, frames, fps=10, macro_block_size=8)

        rows = run_eval(policy, env, reps=args.reps, record=record if args.videos else None)
        env.close()
        s = summarize(rows)
        results["settings"].append({**st, "summary": s, "episodes": rows})
        print(json.dumps({**st, **{k: s[k] for k in ("successes", "episodes", "wilson95", "max_stage_hist")}}))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
