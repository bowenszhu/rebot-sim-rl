# LIBERO run provenance

LeRobot 0.4.4 (`lerobot.scripts.lerobot_eval`), hf-libero 0.1.4, robosuite 1.4.0,
MuJoCo 3.8.1, torch 2.7.1+cu128, `--eval.batch_size=1 --eval.n_episodes=10`
(10 episodes per task), `--seed=1000`, one GPU per job. Evaluation time is
LeRobot's `eval_s` (excludes model download and environment setup).

| Policy | Suite | GPU | Eval time | File |
|---|---|---|---|---|
| `HuggingFaceVLA/smolvla_libero` | Spatial | H100 80GB | 58 min | `smolvla_libero_spatial.eval_info.json` |
| | Object | L40S | 57 min | `smolvla_libero_object.eval_info.json` |
| | Goal | H200 | 92 min | `smolvla_libero_goal.eval_info.json` |
| | Long, tasks 0–1 | L40S | 41 min | `smolvla_libero_10_shard0.eval_info.json` |
| | Long, tasks 2–3 | H200 | 32 min | `smolvla_libero_10_shard1.eval_info.json` |
| | Long, tasks 4–5 | H200 | 39 min | `smolvla_libero_10_shard2.eval_info.json` |
| | Long, tasks 6–7 | H200 | 46 min | `smolvla_libero_10_shard3.eval_info.json` |
| | Long, tasks 8–9 | H200 | 53 min | `smolvla_libero_10_shard4.eval_info.json` |
| `lerobot/pi05_libero_finetuned` (`n_action_steps=10`, `compile_model=false`) | Spatial | H200 | 63 min | `pi05_libero_spatial.eval_info.json` |
| | Object | H200 | 73 min | `pi05_libero_object.eval_info.json` |
| | Goal | H200 | 59 min | `pi05_libero_goal.eval_info.json` |
| | Long | H200 | 127 min | `pi05_libero_10.eval_info.json` |

Notes

- SmolVLA ran with the checkpoint's own `n_action_steps`; the first Goal/Long attempt
  failed because the SmolVLM2 backbone had not been prefetched for the offline run.
- π0.5 with `compile_model=true` (the checkpoint default) spent most of a 2-episode
  smoke test in `torch.compile`; it was disabled for the full runs.
- `summary.md` / `summary.json` are produced by `scripts/summarize_libero.py`.
