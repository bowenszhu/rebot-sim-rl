"""Build the T1-aligned MuJoCo scene from the upstream B601-RS arm model.

Input:  assets/b601/rs_arm_upstream.xml (copied verbatim from reBot
        ReBot_Arm_web_RS/rebotarm_ros2/src/rebotarm_mujoco_rs/models/rs_arm.xml)
Output: assets/b601/b601_t1_scene.xml

Changes relative to upstream (everything else, i.e. link kinematics, inertias,
J1-J6 axes and meshes, is kept unchanged):
  1. The stock Seeed gripper (gripper_end, 16.6 cm from the J6 flange) is replaced
     by a primitive model of the installed PX6D + R8 + DM-Tac W2L tool:
     flange-to-fingertip 24 cm (user tape measurement, t1_protocol.json) and a
     parallel finger gap of 0-8.4 cm (manual maximum opening, profile 08).
  2. Joint ranges are narrowed to the intersection of the upstream MJCF range and
     the real software limits of control profile b601_px6d_w2l_wide_open_20261016_08,
     expressed relative to the profile reference zero (see configs/real_limits.json).
  3. The torque-motor actuators driven by a Python PD loop upstream are replaced by
     MuJoCo position actuators (PD, force-limited to the upstream ctrlrange) with
     actuator gravity compensation, i.e. the same structure as the upstream
     physics-mode sync (tau = bias + kp*(q*-q) - kd*qd, clipped).
  4. A table (base mounted on the table surface, z = 0), the 6 cm foam cube and the
     open lunch box are placed at the positions measured for the real T1 setup.
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Upstream-derived PD gains (rebotarm_mujoco_rs/mujoco_sync.py arm_kp / arm_kd).
ARM_KP = [80.0, 100.0, 100.0, 35.0, 25.0, 18.0]
ARM_KD = [8.0, 10.0, 10.0, 4.0, 3.0, 2.5]
ARM_FORCE = [36, 36, 36, 14, 14, 14]  # upstream ctrlrange (N*m)

TOOL_LEN = 0.24  # J6 flange to fingertip (m)
MAX_GAP = 0.084  # inner finger gap at full opening (m)


def build(limits: dict, layout: dict) -> str:
    tree = ET.parse(ROOT / "assets/b601/rs_arm_upstream.xml")
    root = tree.getroot()
    root.set("model", "b601_t1_scene")

    # --- joint limits --------------------------------------------------------
    for j in root.iter("joint"):
        name = j.get("name")
        if name in limits:
            lo_u, hi_u = map(float, j.get("range").split())
            lo_r, hi_r = limits[name]
            j.set("range", f"{max(lo_u, lo_r):.4f} {min(hi_u, hi_r):.4f}")
            j.set("actuatorgravcomp", "true")
    # gravity compensation routed through the actuators (force limits still apply)
    for b in root.iter("body"):
        if b.get("name") not in (None, "base_link"):
            b.set("gravcomp", "1")
    # --- replace the stock gripper -------------------------------------------
    link6 = next(b for b in root.iter("body") if b.get("name") == "link6")
    old = next(b for b in link6.findall("body") if b.get("name") == "gripper_end")
    link6.remove(old)
    tool = ET.SubElement(link6, "body", name="tool", pos="0 0 0", gravcomp="1")
    ET.SubElement(tool, "inertial", pos="0 0 0.09", mass="0.60",
                  diaginertia="0.0012 0.0012 0.0004")
    # PX6D force sensor + R8 adapter (visual + coarse collision)
    ET.SubElement(tool, "geom", name="px6d", type="cylinder", size="0.0375 0.02",
                  pos="0 0 0.03", rgba="0.15 0.15 0.15 1", group="1")
    ET.SubElement(tool, "geom", name="r8_adapter", type="box", size="0.045 0.03 0.03",
                  pos="0 0 0.08", rgba="0.25 0.25 0.25 1", group="1")
    # finger rail housing
    ET.SubElement(tool, "geom", name="rail", type="box", size="0.02 0.065 0.012",
                  pos="0 0 0.122", rgba="0.6 0.6 0.62 1", group="1")
    finger_len = TOOL_LEN - 0.134
    fz = 0.134 + finger_len / 2
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        fb = ET.SubElement(tool, "body", name=f"finger_{side}", pos=f"0 {sgn * 0.009:.4f} 0",
                           gravcomp="1")
        ET.SubElement(fb, "joint", name=f"finger_{side}_slide", type="slide",
                      axis=f"0 {sgn:.0f} 0", range=f"0 {MAX_GAP / 2:.4f}",
                      damping="5", armature="0.01", actuatorgravcomp="true")
        ET.SubElement(fb, "inertial", pos=f"0 0 {fz:.4f}", mass="0.08",
                      diaginertia="2e-5 2e-5 5e-6")
        # W2L fingertip pad: 2.5 cm wide, 1.8 cm thick, inner face at y = +-0 gap/2
        ET.SubElement(fb, "geom", name=f"finger_{side}_geom", type="box",
                      size=f"0.0125 0.009 {finger_len / 2:.4f}", pos=f"0 0 {fz:.4f}",
                      rgba="0.85 0.85 0.85 1", friction="1.2 0.01 0.001",
                      condim="4", solref="0.008 1", solimp="0.95 0.99 0.001")
    ET.SubElement(tool, "site", name="tcp", pos=f"0 0 {TOOL_LEN - 0.02:.4f}",
                  size="0.006", rgba="1 0 0 0.6", group="2")
    ET.SubElement(tool, "site", name="fingertip", pos=f"0 0 {TOOL_LEN:.4f}",
                  size="0.004", rgba="0 1 0 0.6", group="2")
    # make the arm visual meshes visible in group 1 only (upstream already does this)
    # --- equality: the two fingers move symmetrically ------------------------
    for e in list(root.findall("equality")):
        root.remove(e)
    eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "joint", joint1="finger_left_slide", joint2="finger_right_slide",
                  polycoef="0 1 0 0 0", solref="0.008 1")
    # --- actuators --------------------------------------------------------------
    for a in list(root.findall("actuator")):
        root.remove(a)
    act = ET.SubElement(root, "actuator")
    for i in range(6):
        jn = f"joint{i + 1}"
        lo, hi = limits[jn]
        ET.SubElement(act, "position", name=f"{jn}_pos", joint=jn, kp=str(ARM_KP[i]),
                      kv=str(ARM_KD[i]), forcerange=f"-{ARM_FORCE[i]} {ARM_FORCE[i]}",
                      ctrlrange=f"{lo:.4f} {hi:.4f}")
    # gripper: position servo on half-gap, ~30 N grip force limit
    ET.SubElement(act, "position", name="gripper_pos", joint="finger_left_slide",
                  kp="1500", kv="40", forcerange="-30 30",
                  ctrlrange=f"0 {MAX_GAP / 2:.4f}")

    # --- scene ------------------------------------------------------------------
    ET.SubElement(root, "option", timestep="0.004", iterations="50", solver="Newton",
                  integrator="implicitfast", cone="elliptic", impratio="10")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", diffuse="0.6 0.6 0.6", ambient="0.35 0.35 0.35",
                  specular="0.1 0.1 0.1")
    ET.SubElement(vis, "global", azimuth="150", elevation="-25", offwidth="960",
                  offheight="720")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="2d", name="wood", builtin="flat",
                  rgb1="0.55 0.42 0.30", width="64", height="64")
    ET.SubElement(asset, "material", name="table_mat", texture="wood", rgba="0.75 0.62 0.48 1")
    ET.SubElement(asset, "material", name="mat_white", rgba="0.95 0.95 0.95 1")
    ET.SubElement(asset, "material", name="cube_blue", rgba="0.10 0.30 0.85 1")
    ET.SubElement(asset, "material", name="box_white", rgba="0.92 0.92 0.88 1")
    wb = root.find("worldbody")
    ET.SubElement(wb, "light", pos="0.3 0 1.2", dir="0 0 -1", directional="true")
    ET.SubElement(wb, "geom", name="table", type="plane", size="0.8 0.8 0.02",
                  material="table_mat", friction="1.0 0.01 0.001")
    # A3 mat (visual only) centred on the measured grid centre
    gx, gy = layout["grid_center_xy"]
    ET.SubElement(wb, "geom", name="desk_mat", type="box", size="0.148 0.21 0.0005",
                  pos=f"{gx:.3f} {gy:.3f} 0.0005", material="mat_white",
                  contype="0", conaffinity="0", group="1")
    # Foam cube: 6 cm, light, moderately high friction (foam)
    cube = layout["cube"]
    cb = ET.SubElement(wb, "body", name="cube", pos=f"{gx:.3f} {gy:.3f} {cube['half'] + 0.001:.4f}")
    ET.SubElement(cb, "freejoint", name="cube_free")
    ET.SubElement(cb, "geom", name="cube_geom", type="box",
                  size=f"{cube['half']} {cube['half']} {cube['half']}", mass=str(cube["mass"]),
                  material="cube_blue", friction=f"{cube['friction']} 0.01 0.001",
                  condim="4", solref="0.008 1", solimp="0.9 0.98 0.002")
    # Lunch box: floor + 4 walls
    box = layout["box"]
    bx, by = box["center_xy"]
    yaw = math.radians(box["long_axis_deg"])
    qw, qz = math.cos(yaw / 2), math.sin(yaw / 2)
    bb = ET.SubElement(wb, "body", name="box", pos=f"{bx:.3f} {by:.3f} 0",
                       quat=f"{qw:.5f} 0 0 {qz:.5f}")
    L, W, H, t = box["outer_l"] / 2, box["outer_w"] / 2, box["height"], box["wall"]
    ET.SubElement(bb, "geom", name="box_floor", type="box", size=f"{L} {W} {t / 2}",
                  pos=f"0 0 {t / 2}", material="box_white")
    for nm, px, py, sx, sy in (("box_wall_px", L - t / 2, 0, t / 2, W),
                               ("box_wall_nx", -L + t / 2, 0, t / 2, W),
                               ("box_wall_py", 0, W - t / 2, L, t / 2),
                               ("box_wall_ny", 0, -W + t / 2, L, t / 2)):
        ET.SubElement(bb, "geom", name=nm, type="box", size=f"{sx} {sy} {H / 2}",
                      pos=f"{px} {py} {H / 2}", material="box_white")
    ET.SubElement(bb, "site", name="box_center", pos=f"0 0 {t}", size="0.005",
                  rgba="1 0 1 0.5", group="2")
    ET.SubElement(wb, "camera", name="overview", pos="0.95 0.55 0.65",
                  xyaxes="-0.5 0.866 0 -0.42 -0.24 0.875", fovy="50")
    ET.SubElement(wb, "camera", name="front", pos="0.80 -0.05 0.45",
                  xyaxes="0.1 1 0 -0.55 0.05 0.83", fovy="55")
    # contact exclusions copied from the upstream grasp scene (+ tool vs link5)
    contact = ET.SubElement(root, "contact")
    for b1, b2 in (("world", "link1"), ("base_link", "link1"), ("link1", "link2"),
                   ("link2", "link3"), ("link2", "link5"), ("link3", "link4"),
                   ("link3", "link5"), ("link4", "link5"), ("link5", "link6"),
                   ("link5", "tool"), ("link6", "finger_left"), ("link6", "finger_right"),
                   ("tool", "finger_left"), ("tool", "finger_right"),
                   ("finger_left", "finger_right"), ("world", "base_link")):
        ET.SubElement(contact, "exclude", body1=b1, body2=b2)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limits", type=Path, default=ROOT / "configs/real_limits.json")
    ap.add_argument("--layout", type=Path, default=ROOT / "configs/t1_layout.json")
    ap.add_argument("--out", type=Path, default=ROOT / "assets/b601/b601_t1_scene.xml")
    args = ap.parse_args()
    lim = json.loads(args.limits.read_text())
    limits = {k: v["sim_range_rad"] for k, v in lim["joints"].items()}
    layout = json.loads(args.layout.read_text())
    args.out.write_text(build(limits, layout))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
