"""Derive simulation joint ranges from a reBot control profile (read-only).

Real software limits are stored in raw encoder coordinates; the recorder and
LeRobot datasets use q = raw - reference_raw_rad. The simulation uses the same
relative coordinates (verified with FK against recorded T1 grasps, see
results/model_check/). The sim range is the intersection of the upstream MJCF
range and the real relative range.
"""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "configs/real_limits.json")
    args = ap.parse_args()
    prof = json.loads(args.profile.read_text())
    mj = {j.get("name"): list(map(float, j.get("range").split()))
          for j in ET.parse(ROOT / "assets/b601/rs_arm_upstream.xml").getroot().iter("joint")
          if j.get("range")}
    out = {
        "source_profile": prof["profile_id"],
        "source_profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "note": "relative = raw - reference_raw_rad; sim_range = intersection(upstream MJCF, real relative)",
        "joints": {},
    }
    for i in range(1, 7):
        pj = prof["joints"][str(i)]
        ref = pj["reference_raw_rad"]
        lo, hi = pj["limits_raw_rad"]
        rel = [round(lo - ref, 4), round(hi - ref, 4)]
        up = mj[f"joint{i}"]
        out["joints"][f"joint{i}"] = {
            "reference_raw_rad": ref,
            "real_limits_raw_rad": [lo, hi],
            "real_limits_relative_rad": rel,
            "upstream_mjcf_range_rad": up,
            "sim_range_rad": [max(rel[0], up[0]), min(rel[1], up[1])],
            "limit_basis": pj.get("limit_basis"),
        }
    j7 = prof["joints"]["7"]
    out["gripper"] = {
        "j7_reference_raw_rad": j7["reference_raw_rad"],
        "j7_limits_raw_rad": j7["limits_raw_rad"],
        "manual_max_raw_rad": 6.06739,
        "manual_max_gap_m": 0.084,
        "mapping": "gap_m ~= 0.084 * (raw - reference) / (6.06739 - reference); linear estimate from profile 08 limit_basis",
        "sim_gap_range_m": [0.0, 0.084],
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["sim_range_rad"] for k, v in out["joints"].items()}))


if __name__ == "__main__":
    main()
