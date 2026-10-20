"""Extract joint traces from reBot T1 mirror recordings (read-only) for the FK consistency check.

Reads <episode>/mirror_trace.json and meta.json, converts raw motor angles to
profile-relative angles (q = raw - reference_raw_rad), and writes a compact JSON
with the planned cube cell, mat offset and a downsampled joint trace.
Never touches hardware; only reads recorded files.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", type=Path, required=True)
    ap.add_argument("--rebot-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    out = []
    for ep in sorted(args.episodes_dir.iterdir())[: args.max_episodes]:
        meta_p, trace_p = ep / "meta.json", ep / "mirror_trace.json"
        if not (meta_p.exists() and trace_p.exists()):
            continue
        meta = json.loads(meta_p.read_text())
        trace = json.loads(trace_p.read_text())
        if trace.get("outcome") not in ("completed", "operator_stop_returned"):
            continue
        profile = json.loads((args.rebot_root / meta["profile"]).read_text())
        ref = {j: profile["joints"][j]["reference_raw_rad"] for j in profile["joints"]}
        samples = []
        for s in trace["states"][:: args.stride]:
            st = s["states"]
            if not all(str(j) in st for j in range(1, 8)):
                continue
            samples.append(
                [round(st[str(j)]["position_rad"] - ref[str(j)], 5) for j in range(1, 8)]
            )
        out.append(
            {
                "episode": ep.name,
                "cell": meta["plan"]["cell"],
                "yaw_deg": meta["plan"]["yaw_deg"],
                "mat_offset_cm": meta.get("mat_offset_cm"),
                "profile": meta["profile"],
                "q_rel": samples,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {len(out)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
