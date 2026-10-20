# Checkpoints

Final SB3 PPO checkpoints of every run in the README tables (`model_last.zip`; BC policies are
saved in the same PPO format as `bc_policy.zip`). Load with
`stable_baselines3.PPO.load(path, device="cpu")`; observations come from `B601T1Env`
(`rebot_sim_rl/env.py`). Re-evaluate, for example:

    python scripts/evaluate.py --model checkpoints/bcppo_n10_s0/model_last.zip \
        --obs-noise 0 0.005 0.01 0.02 --videos /tmp/videos

Training configs and curves: `results/runs/main/<run>/`.
