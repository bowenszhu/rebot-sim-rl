# Recorded real-robot joint traces

`real_t1_joint_traces.json`: joint angles of 41 teleoperated T1 demonstrations recorded on the
real B601-RS on 2026-10-16 (the episodes that completed or ended with an operator stop and a safe
return). Each entry has the episode name, the planned cube cell and yaw, the mat offset, the
control profile and `q_rel`: J1–J7 positions relative to the profile reference zero, every 5th
feedback sample (~10 Hz). Produced by `tools/extract_real_joint_traces.py` from the reBot
recordings (`<episode>/mirror_trace.json`, `meta.json`); used only by `scripts/check_model.py`
and as test input for `sim2real/analyze_touch_points.py`. No images or sensor data.
