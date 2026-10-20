"""Fixed-seed evaluation on the T1 start set."""
import json

import numpy as np

from .env import B601T1Env
from .layout import eval_starts, wilson

EVAL_REPS = 3  # 9 cells x 2 yaws x 3 placement jitters = 54 episodes
JITTER_CM = 1.0  # T1 placement tolerance


def eval_episodes(reps: int = EVAL_REPS, seed: int = 12345):
    """Deterministic list of (cell, yaw_deg, jitter_xy, reset_seed)."""
    rng = np.random.default_rng(seed)
    out = []
    for rep in range(reps):
        for cell, yaw in eval_starts():
            jit = rng.uniform(-JITTER_CM, JITTER_CM, 2) / 100.0 if rep > 0 else np.zeros(2)
            out.append((cell, yaw, jit.tolist(), int(rng.integers(1 << 30))))
    return out


def run_eval(policy, env: B601T1Env, reps: int = EVAL_REPS, record=None, seed: int = 12345):
    """policy(obs) -> action. record(i, frames) is called with rendered frames if given."""
    rows = []
    for i, (cell, yaw, jit, s) in enumerate(eval_episodes(reps, seed)):
        obs, info = env.reset(seed=s, options={"cell": cell, "yaw_deg": yaw, "jitter": jit})
        ret, frames = 0.0, []
        for t in range(env.max_steps):
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            if record is not None:
                frames.append(env.render())
            if term or trunc:
                break
        rows.append({"i": i, "cell": cell, "yaw_deg": yaw, "jitter_m": jit, "seed": s,
                     "success": bool(info["is_success"]), "max_stage": int(info["stage"]),
                     "return": round(ret, 3), "length": t + 1,
                     "release_z_m": None if env.release is None else round(env.release[0], 4),
                     "release_hspeed_mps": None if env.release is None else round(env.release[1], 3),
                     "careful_success": bool(info["is_success"]) and env.careful_release()})
        if record is not None:
            record(i, rows[-1], frames)
    return rows


def summarize(rows):
    k = sum(r["success"] for r in rows)
    n = len(rows)
    lo, hi = wilson(k, n)
    kg = sum(r.get("careful_success", False) for r in rows)
    ok = [r for r in rows if r.get("success") and r.get("release_z_m") is not None]
    stages = np.bincount([r["max_stage"] for r in rows], minlength=5).tolist()
    return {"successes": k, "episodes": n, "rate": k / n if n else float("nan"),
            "wilson95": [round(lo, 3), round(hi, 3)],
            "max_stage_hist": dict(zip(["none", "grasp_low", "lifted", "in_box_held", "success"], stages)),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "careful_successes": kg, "careful_wilson95": [round(v, 3) for v in wilson(kg, n)],
            "release_z_m_median_on_success": float(np.median([r["release_z_m"] for r in ok])) if ok else None,
            "release_hspeed_mps_median_on_success":
                float(np.median([r["release_hspeed_mps"] for r in ok])) if ok else None}


def dump(path, rows, extra=None):
    payload = {"summary": summarize(rows), "episodes": rows}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=1))
    return payload["summary"]
