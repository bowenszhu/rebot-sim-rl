# Sim-to-real kinematic check (prepared, NOT executed)

Status: scripts and protocol only. Nothing in this directory has been run on the
real B601-RS. Claude does not run commands that enable or move the arm; the
operator runs the recording step at the arm.

## Goal

Check that the simulated joint space (`assets/b601/b601_t1_scene.xml`, joint angles
relative to the profile-08 reference zero) puts the fingertip where the real
fingertip is. This is the precondition for any claim that a sim policy's joint
targets mean the same thing on the real arm. It does not test dynamics, contact,
the gripper force, or a learned policy.

What is already known offline (`results/model_check/model_check.md`): replaying 40
recorded T1 grasps through the sim kinematics puts the TCP within a median 2.3 cm
(p90 3.4 cm) of the planned cube cell, with the fingertip 0.7–2.6 cm above the table.
The planned cells themselves are only known to about ±2 cm, so this bounds but does
not measure the model error. The touch-point check below replaces the planned cells
with tape-measured points.

## Step 1 — mark points (no power needed)

1. Use the T1 setup unchanged (mat offset as recorded, box in place).
2. Mark 6 points with small tape crosses: grid A1, A3, B2, C1, C3 and the box centre.
3. Measure each point's x (forward) and y (left) from the J1 axis with a ruler
   (±0.5 cm). Write them into a copy of `touch_points_template.json`
   (keep the order; the analysis matches dwells to points in that order).

## Step 2 — record one mirror episode (operator at the arm)

Use the existing supervised mirror recorder; no new motion path, limits or gates:

```bash
python3 -B experiments/il_vla_benchmark/episode.py \
    --session datasets/il_vla_benchmark/sim2real_touch_<date> --episode 0 \
    --cell B2 --yaw 0 --duration 90 \
    --tactile-status <current W2L status.json>
```

(run from the reBot root; `--cell/--yaw` only fill the metadata.)

During the episode, with the gripper near-closed:

- move the fingertip (lowest point of the W2L fingers) onto each marked point in
  the listed order, touching the tape lightly, with the tool roughly vertical;
- hold still for at least 2 s at each point;
- lift at least 5 cm between points (dwells above 5 cm are ignored);
- return to neutral and end as usual.

Stop rules are the usual ones in reBot `docs/SAFETY.md`. Do not press into the
table; the check needs contact position, not force.

## Step 3 — offline analysis (Claude or the user, no hardware)

```bash
conda run -n rebot-sim-rl python sim2real/analyze_touch_points.py \
    --episode ~/Projects/reBot/datasets/il_vla_benchmark/sim2real_touch_<date>/episodes/ep000_* \
    --rebot-root ~/Projects/reBot --points sim2real/touch_points_<date>.json \
    --out results/sim2real/touch_check_<date>.json
```

Report: dwells found vs points listed, per-point xy and z error, and the planar
(dx, dy, dyaw) fit with its residual. Acceptance suggestion: per-point xy error
≤ 1.5 cm and z error ≤ 1.0 cm after a fitted base offset of ≤ 2 cm / 3°. Larger
errors point to the tool length (24 cm tape value vs. CAD), the J2/J3 zero, or the
table height, in that order.

## Optional step 4 — gripper gap map

The offline check found a median 4.75 cm finger gap (linear J7 map) while holding
the 6 cm foam cube. To separate foam compression from map error, close the fingers
on rigid blocks of 2, 4 and 6 cm and record J7 raw for each (read-only, gripper
only, inside a normal mirror episode). Fit `gap = a (raw - ref) + b` and update
`configs/real_limits.json` if it differs.

## Self-test of the analysis script (done, offline)

`analyze_touch_points.py` was run on the existing T1 recording
`ep004_20261016_011328` (cell A1) only to check that it parses mirror traces and
detects dwells: the first low dwell (the grasp) lands 1.4 cm from the A1 estimate;
the second (the box release) is, as expected, not a touch point. This is a parser
test, not a sim-to-real result.
