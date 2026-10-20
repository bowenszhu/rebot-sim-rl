"""Offline analysis for the sim-to-real kinematic check (see sim2real/PROTOCOL.md).

Input: one reBot mirror recording (<episode>/mirror_trace.json + meta.json) in which
the operator touched known table points with the fingertip and held still, plus a
JSON file with the tape-measured point coordinates in the base frame.

The script never talks to the robot. It
  1. converts raw motor angles to profile-relative angles (q = raw - reference),
  2. finds dwell segments (all joints slower than --vel-thresh for >= --min-dwell s),
  3. computes the simulated fingertip position for each dwell with the sim model,
  4. matches dwells to the listed points in order and reports xyz errors,
  5. fits a planar base correction (dx, dy, dyaw) that best explains the errors.

  python sim2real/analyze_touch_points.py --episode <dir> --rebot-root ~/Projects/reBot \
      --points sim2real/touch_points_template.json --out results/sim2real/touch_check.json
"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from rebot_sim_rl import SCENE_XML


def load_trace(ep: Path, rebot_root: Path):
    meta = json.loads((ep / "meta.json").read_text())
    trace = json.loads((ep / "mirror_trace.json").read_text())
    prof = json.loads((rebot_root / meta["profile"]).read_text())
    ref = np.array([prof["joints"][str(j)]["reference_raw_rad"] for j in range(1, 8)])
    t, q = [], []
    for s in trace["states"]:
        st = s["states"]
        if all(str(j) in st for j in range(1, 8)):
            t.append(s["requested_at"])
            q.append([st[str(j)]["position_rad"] for j in range(1, 8)])
    return np.array(t), np.array(q) - ref, meta


def dwells(t, q, vel_thresh, min_dwell):
    dt = np.diff(t)
    v = np.abs(np.diff(q[:, :6], axis=0)).max(axis=1) / np.maximum(dt, 1e-3)
    still = np.concatenate([[False], v < vel_thresh])
    segs, start = [], None
    for i, s in enumerate(still):
        if s and start is None:
            start = i
        if (not s or i == len(still) - 1) and start is not None:
            end = i if not s else i + 1
            if t[end - 1] - t[start] >= min_dwell:
                segs.append((start, end))
            start = None
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--rebot-root", type=Path, required=True)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--vel-thresh", type=float, default=0.02, help="rad/s")
    ap.add_argument("--min-dwell", type=float, default=1.5, help="s")
    ap.add_argument("--max-tip-z", type=float, default=0.05, help="ignore dwells above this fingertip height (m)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    tip = m.site("fingertip").id
    t, q, meta = load_trace(args.episode, args.rebot_root)
    found = []
    for a, b in dwells(t, q, args.vel_thresh, args.min_dwell):
        d.qpos[:6] = np.median(q[a:b, :6], axis=0)
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[tip].copy()
        if p[2] <= args.max_tip_z:
            found.append({"t0": float(t[a] - t[0]), "dur_s": float(t[b - 1] - t[a]), "tip_sim_m": p.round(4).tolist()})
    pts = json.loads(args.points.read_text())["points"]
    rows = []
    for dw, pt in zip(found, pts):
        meas = np.array(pt["xyz_m"])
        sim = np.array(dw["tip_sim_m"])
        rows.append({**dw, "name": pt["name"], "measured_m": meas.tolist(),
                     "err_cm": ((sim - meas) * 100).round(2).tolist(),
                     "err_xy_cm": round(float(np.linalg.norm((sim - meas)[:2])) * 100, 2)})
    fit = None
    if len(rows) >= 3:
        S = np.array([r["tip_sim_m"][:2] for r in rows])
        M = np.array([r["measured_m"][:2] for r in rows])
        # 2D rigid fit M ~ R S + t (Kabsch)
        sc, mc = S.mean(0), M.mean(0)
        U, _, Vt = np.linalg.svd((S - sc).T @ (M - mc))
        R = (U @ Vt).T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = (U @ Vt).T
        tvec = mc - R @ sc
        resid = np.linalg.norm((S @ R.T + tvec) - M, axis=1)
        fit = {"dyaw_deg": round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2),
               "dxy_cm": (tvec * 100).round(2).tolist(),
               "residual_xy_cm_median": round(float(np.median(resid)) * 100, 2),
               "residual_xy_cm_max": round(float(resid.max()) * 100, 2)}
    out = {"episode": str(args.episode), "profile": meta["profile"], "dwells_found": len(found),
           "points_listed": len(pts), "matched": rows, "planar_fit": fit,
           "note": "dwells are matched to points in listed order; check counts before trusting the match"}
    print(json.dumps({k: v for k, v in out.items() if k != "matched"}, indent=1))
    for r in rows:
        print(f"{r['name']:>8s} err_xy {r['err_xy_cm']:5.2f} cm  err_z {r['err_cm'][2]:5.2f} cm")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
