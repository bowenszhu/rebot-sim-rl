"""Check the simulated B601 against real-robot records (offline, no hardware access).

1. Joint-limit table: upstream MJCF vs real software limits (profile 08) vs sim.
2. FK consistency: recorded T1 joint traces (q = raw - reference) are replayed
   kinematically in the sim model. At each grasp (gripper starts closing after its
   largest opening) the simulated fingertip position is compared with the planned
   cube cell, and the tool axis with the vertical.
3. Gripper mapping: the J7 reading while holding the 6 cm die is converted to a
   finger gap with the linear profile-08 mapping and compared with the die size.
4. Fraction of recorded samples outside the sim joint ranges.

Writes results/model_check/{model_check.json, model_check.md}.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rebot_sim_rl import ROOT, SCENE_XML
from rebot_sim_rl.layout import cell_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=ROOT / "data/real_t1_joint_traces.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/model_check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lim = json.loads((ROOT / "configs/real_limits.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    tip_id = m.site("fingertip").id
    tcp_id = m.site("tcp").id
    g = lim["gripper"]
    gap_per_rad = g["manual_max_gap_m"] / (g["manual_max_raw_rad"] - g["j7_reference_raw_rad"])

    eps = json.loads(args.traces.read_text())
    rows, outside, total = [], 0, 0
    lo = np.array([lim["joints"][f"joint{i}"]["sim_range_rad"][0] for i in range(1, 7)])
    hi = np.array([lim["joints"][f"joint{i}"]["sim_range_rad"][1] for i in range(1, 7)])
    for e in eps:
        Q = np.array(e["q_rel"])
        if len(Q) < 20:
            continue
        total += len(Q)
        outside += int(np.any((Q[:, :6] < lo - 0.02) | (Q[:, :6] > hi + 0.02), axis=1).sum())
        j7 = Q[:, 6]
        half = len(Q) // 2
        peak = int(np.argmax(j7[:half]))
        if j7[peak] < 2.0:
            continue
        after = np.nonzero(j7[peak:] < j7[peak] - 0.5)[0]
        if len(after) == 0:
            continue
        gi = peak + int(after[0])
        # hold value: median J7 over the next ~1 s once closing has settled
        hold = float(np.median(j7[gi + 10: gi + 25])) if gi + 25 < len(j7) else float("nan")
        d.qpos[:6] = Q[gi, :6]
        mujoco.mj_kinematics(m, d)
        tcp = d.site_xpos[tcp_id].copy()
        tip = d.site_xpos[tip_id].copy()
        axis = d.site_xmat[tcp_id].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))
        target = cell_xy(e["cell"])
        rows.append({
            "episode": e["episode"], "cell": e["cell"], "yaw_deg": e["yaw_deg"],
            "tcp_xy": tcp[:2].round(4).tolist(), "cell_xy": target.round(4).tolist(),
            "xy_error_cm": round(float(np.linalg.norm(tcp[:2] - target)) * 100, 2),
            "fingertip_z_cm": round(float(tip[2]) * 100, 2),
            "tool_tilt_deg": round(tilt, 1),
            "hold_gap_cm": round(hold * gap_per_rad * 100, 2),
        })
    err = np.array([r["xy_error_cm"] for r in rows])
    tipz = np.array([r["fingertip_z_cm"] for r in rows])
    gap = np.array([r["hold_gap_cm"] for r in rows if np.isfinite(r["hold_gap_cm"])])
    tilt = np.array([r["tool_tilt_deg"] for r in rows])
    summary = {
        "episodes_used": len(rows),
        "samples_total": total,
        "samples_outside_sim_range_by_gt_0.02rad": outside,
        "grasp_xy_error_cm": {"median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
                              "max": float(err.max())},
        "grasp_fingertip_z_cm": {"median": float(np.median(tipz)), "min": float(tipz.min()),
                                 "max": float(tipz.max())},
        "grasp_tool_tilt_deg": {"median": float(np.median(tilt)), "max": float(tilt.max())},
        "hold_gap_cm": {"median": float(np.median(gap)), "min": float(gap.min()), "max": float(gap.max()),
                        "expected": "~6 cm die (soft, can be squeezed)"},
    }
    (args.out / "model_check.json").write_text(json.dumps({"summary": summary, "grasps": rows}, indent=1))

    lines = ["# Model check: simulated B601 vs real records", "",
             f"Source profile: `{lim['source_profile']}` (sha256 `{lim['source_profile_sha256'][:12]}`)", "",
             "## Joint ranges (rad, relative to the profile reference zero)", "",
             "| joint | upstream MJCF | real software limit | sim range |", "|---|---|---|---|"]
    for k, v in lim["joints"].items():
        lines.append(f"| {k} | {v['upstream_mjcf_range_rad']} | {v['real_limits_relative_rad']} | "
                     f"{[round(x, 3) for x in v['sim_range_rad']]} |")
    lines += ["", "## FK consistency against recorded T1 grasps", "",
              f"- Episodes with a detectable grasp: {len(rows)}",
              f"- TCP-to-planned-cell xy error: median {summary['grasp_xy_error_cm']['median']:.1f} cm, "
              f"p90 {summary['grasp_xy_error_cm']['p90']:.1f} cm, max {summary['grasp_xy_error_cm']['max']:.1f} cm "
              "(cell positions themselves are +-2 cm; the die is placed +-1 cm)",
              f"- Fingertip height at grasp: median {summary['grasp_fingertip_z_cm']['median']:.1f} cm "
              f"(range {summary['grasp_fingertip_z_cm']['min']:.1f} to {summary['grasp_fingertip_z_cm']['max']:.1f}); "
              "the table is z = 0",
              f"- Tool axis tilt from vertical at grasp: median {summary['grasp_tool_tilt_deg']['median']:.0f} deg",
              f"- Finger gap while holding the 6 cm die (linear J7 map): median {summary['hold_gap_cm']['median']:.1f} cm "
              f"(range {summary['hold_gap_cm']['min']:.1f} to {summary['hold_gap_cm']['max']:.1f})",
              f"- Recorded samples outside the sim joint ranges by > 0.02 rad: {outside} / {total}", "",
              "| episode | cell | yaw | xy error cm | fingertip z cm | tilt deg | hold gap cm |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['episode']} | {r['cell']} | {r['yaw_deg']:.0f} | {r['xy_error_cm']} | "
                     f"{r['fingertip_z_cm']} | {r['tool_tilt_deg']} | {r['hold_gap_cm']} |")
    (args.out / "model_check.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
