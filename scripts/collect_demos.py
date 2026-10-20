"""Collect scripted-IK demonstrations on the training start distribution.

The expert sees the true cube pose. Labels are the clean expert actions; the
executed action can be perturbed (DART-style) to widen state coverage. Only
successful episodes are kept, like the accepted real T1 demos.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rebot_sim_rl.env import B601T1Env
from rebot_sim_rl.expert import ScriptedExpert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="accepted demonstrations to keep")
    ap.add_argument("--exec-noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/demos/expert_n50.npz"))
    args = ap.parse_args()
    env = B601T1Env()
    rng = np.random.default_rng(args.seed)
    ex = ScriptedExpert(env)
    obs_l, act_l, ep_l = [], [], []
    tried = kept = 0
    while kept < args.n:
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        ex.reset()
        O, A = [], []
        for t in range(env.max_steps):
            a = ex.act()
            O.append(obs)
            A.append(a)
            a_exec = np.clip(a + rng.normal(0, args.exec_noise, 5), -1, 1) if args.exec_noise else a
            obs, r, term, trunc, info = env.step(a_exec)
            if term or trunc:
                break
        tried += 1
        if info["is_success"]:
            obs_l += O
            act_l += A
            ep_l += [kept] * len(O)
            kept += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=np.array(obs_l, np.float32), act=np.array(act_l, np.float32),
                        episode=np.array(ep_l, np.int32))
    meta = {"kept": kept, "tried": tried, "transitions": len(obs_l), "exec_noise": args.exec_noise,
            "seed": args.seed}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
