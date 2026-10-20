"""Train PPO (from scratch or BC-initialised) or SAC on B601T1-v0.

Outputs in --out:
  monitor/env*.monitor.csv  per-episode training return / length / success
  eval_curve.csv            fixed-seed deterministic eval (54 episodes) every --eval-every steps
  model_last.zip, config.json
"""
import argparse
import csv
import json
import time
from pathlib import Path

import rebot_sim_rl  # noqa: F401  (sets BLAS thread env before numpy is imported)
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rebot_sim_rl.agents import actor_parameters, make_ppo, make_sac, make_vec_env
from rebot_sim_rl.env import B601T1Env
from rebot_sim_rl.evaluation import run_eval, summarize


class EvalAndWarmup(BaseCallback):
    def __init__(self, out: Path, every: int, eval_env, warmup_steps: int = 0):
        super().__init__()
        self.out, self.every, self.env, self.warmup = out, every, eval_env, warmup_steps
        self.next_eval = 0
        self.frozen = False
        self.f = open(out / "eval_curve.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["timesteps", "successes", "episodes", "rate", "mean_return", "wall_s"])
        self.t0 = time.time()

    def _set_actor_frozen(self, frozen: bool):
        for p in actor_parameters(self.model):
            p.requires_grad_(not frozen)
        self.frozen = frozen

    def _evaluate(self):
        rows = run_eval(lambda o: self.model.predict(o, deterministic=True)[0], self.env)
        s = summarize(rows)
        self.w.writerow([self.num_timesteps, s["successes"], s["episodes"], round(s["rate"], 4),
                         round(s["mean_return"], 3), round(time.time() - self.t0, 1)])
        self.f.flush()
        print(f"[{self.num_timesteps:>9d}] eval {s['successes']}/{s['episodes']} "
              f"stages={s['max_stage_hist']} t={time.time() - self.t0:.0f}s", flush=True)

    def _on_training_start(self):
        if self.warmup > 0 and isinstance(self.model, PPO):
            self._set_actor_frozen(True)
        self._evaluate()
        self.next_eval = self.every

    def _on_rollout_start(self):
        if self.frozen and self.num_timesteps >= self.warmup:
            self._set_actor_frozen(False)

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
            self.next_eval += self.every
        return True

    def _on_training_end(self):
        self._evaluate()
        self.f.close()


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--init", type=Path, default=None, help="BC policy zip (PPO only)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-std-init", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=0, help="critic-only updates before the actor trains")
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--obs-noise", type=float, default=0.0, help="train-time cube position obs noise (m)")
    ap.add_argument("--reward-version", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    venv = make_vec_env(args.n_envs, args.seed, args.out / "monitor", {"obs_noise_cube": args.obs_noise,
                                                                 "reward_version": args.reward_version})
    if args.algo == "ppo":
        dev = "cpu" if args.device == "auto" else args.device
        model = make_ppo(venv, args.seed, lr=args.lr, log_std_init=args.log_std_init,
                         ent_coef=args.ent_coef, device=dev)
        if args.init is not None:
            bc = PPO.load(args.init, device=dev)
            sd = {k: v for k, v in bc.policy.state_dict().items()
                  if k.startswith(("mlp_extractor.policy_net", "action_net"))}
            missing = model.policy.load_state_dict(sd, strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys
            model.policy.log_std.data[:] = args.log_std_init
    else:
        dev = "cuda" if args.device == "auto" else args.device
        model = make_sac(venv, args.seed, lr=args.lr, device=dev)
    # evaluation always at nominal settings (no obs noise); robustness is evaluated separately
    cb = EvalAndWarmup(args.out, args.eval_every, B601T1Env(reward_version=args.reward_version),
                       args.warmup_steps)
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(args.out / "model_last.zip")
    venv.close()


if __name__ == "__main__":
    main()
