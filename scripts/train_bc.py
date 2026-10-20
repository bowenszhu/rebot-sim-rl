"""Behaviour cloning (MSE on actions) into the PPO actor network.

The BC policy has exactly the PPO actor architecture, so it can be evaluated on
its own (pure BC) and used to initialise PPO fine-tuning.
"""
import argparse
import csv
import json
from pathlib import Path

import rebot_sim_rl  # noqa: F401  (sets BLAS thread env before numpy is imported)
import numpy as np
import torch

from rebot_sim_rl.agents import make_ppo, make_vec_env
from rebot_sim_rl.env import B601T1Env
from rebot_sim_rl.evaluation import dump, run_eval


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    z = np.load(args.demos)
    obs, act, ep = z["obs"], z["act"], z["episode"]
    eps = np.unique(ep)
    val_eps = rng.choice(eps, max(1, int(len(eps) * args.val_frac)), replace=False)
    vm = np.isin(ep, val_eps)
    O, A = torch.tensor(obs[~vm]), torch.tensor(act[~vm])
    Ov, Av = torch.tensor(obs[vm]), torch.tensor(act[vm])

    venv = make_vec_env(1, args.seed, subproc=False)
    model = make_ppo(venv, args.seed)
    pol = model.policy
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    def predict(o):
        feat = pol.extract_features(o, pol.pi_features_extractor)
        return pol.action_net(pol.mlp_extractor.forward_actor(feat))

    log = []
    for e in range(args.epochs):
        perm = torch.randperm(len(O))
        tot = 0.0
        for i in range(0, len(O), args.batch):
            idx = perm[i:i + args.batch]
            loss = ((predict(O[idx]) - A[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        with torch.no_grad():
            vl = ((predict(Ov) - Av) ** 2).mean().item()
        log.append((e + 1, tot / len(O), vl))
    with open(args.out / "bc_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_mse"])
        w.writerows(log)
    model.save(args.out / "bc_policy.zip")

    env = B601T1Env()
    s = dump(args.out / "eval.json",
             run_eval(lambda o: model.predict(o, deterministic=True)[0], env),
             {"demos": str(args.demos), "seed": args.seed, "epochs": args.epochs})
    print(json.dumps({"final_train_mse": log[-1][1], "final_val_mse": log[-1][2], **s}))


if __name__ == "__main__":
    main()
