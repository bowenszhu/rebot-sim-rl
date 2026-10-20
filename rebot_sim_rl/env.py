"""Gymnasium environment: B601-RS T1 pick-and-place in MuJoCo (state observations).

Task (aligned with reBot t1_pick_place_v1): pick the ~6 cm blue die from the mat
grid and release it inside the open lunch box.

MDP
  control rate   10 Hz (25 physics steps of 4 ms)
  action (5)     [dx, dy, dz, dyaw, grip] in [-1, 1]
                 TCP position delta (2 cm / step), tool yaw delta (0.15 rad / step),
                 absolute finger-gap command (-1 closed .. +1 = 8.4 cm open).
                 A damped-least-squares IK turns the Cartesian target (tool axis
                 pointing down) into joint position targets, which are clipped to
                 the real software limits and rate-limited (1 rad/s), then tracked
                 by force-limited joint PD servos (position actuators).
  observation    38-d state vector (joint state, TCP, die pose, box offset,
                 contacts, previous action); die position optionally noisy.
  reward         dense, staged, per-step max 1 (version 2, the default):
                 reach 0.2, grasp +0.2, lift +0.1, transport +0.4 (1 - tanh(3 d)),
                 die inside box 0.9; success ends the episode with 1.0 for each
                 remaining step (equivalent to an absorbing max-reward state, so the
                 agent is never paid for delaying success). Version 1 (first pilot)
                 had no lift term and a transport term 1 - tanh(5 d) that is flat
                 ~0.4 m from the box; PPO then learned to grasp and hold (README).
                 Version 3 keeps the v2 shaping but counts success only for a careful
                 release (see CAREFUL_RELEASE_*): v2 policies let go while moving.
  success        die centre inside the box's inner footprint, below 6 cm height,
                 not touched by either finger.
  termination    success, or die knocked > 0.6 m from the workspace centre.
  truncation     150 steps (15 s).
"""
from __future__ import annotations

import math

import gymnasium as gym
import mujoco
import numpy as np

from . import SCENE_XML
from .layout import LAYOUT, cell_xy, grid_frame

R_DOWN = np.diag([1.0, -1.0, -1.0])  # tool z -> world -z, tool x -> world x at yaw 0
# The real J4 lower limit (-1.365 rad) makes a vertical tool infeasible over most of the
# workspace; operators grasp with the tool tilted ~12 deg outward (median over 40 T1
# grasps, results/model_check). The IK target therefore tilts the tool outward
# (tip away from the base) by a height-dependent angle.
TILT_DEG_AT_TABLE = 12.0
TILT_DEG_PER_M = 80.0
TILT_DEG_MAX = 28.0
# Careful placement (reward v3 success condition, also reported by the evaluation):
# at the last release the die centre is at most 8 cm high (bottom ~at the 4.5 cm rim)
# and moves horizontally at most 0.2 m/s. The scripted expert releases at a median
# 4.3 cm / 0.08 m/s; reward-v2 policies released while still moving, from up to 14 cm.
CAREFUL_RELEASE_MAX_Z = 0.08
CAREFUL_RELEASE_MAX_HSPEED = 0.2
MAX_GAP = 0.084
JOINT_VMAX = 1.0  # rad/s
LOOK_TCP = np.array([0.33, 0.12, 0.20])  # fixed "look" pose above the mat (T1 episodes start there)


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def axis_angle(axis, th):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def target_rotation(pos, yaw):
    """Tool orientation: yaw about vertical, then tilted outward in the radial plane."""
    tilt = min(TILT_DEG_MAX, TILT_DEG_AT_TABLE + TILT_DEG_PER_M * max(0.0, pos[2] - 0.03))
    r = np.array([pos[0], pos[1], 0.0])
    r /= max(np.linalg.norm(r), 1e-6)
    return axis_angle(np.cross([0.0, 0.0, 1.0], r), -math.radians(tilt)) @ rotz(yaw) @ R_DOWN


def fold90(a):
    """Fold an angle to [-pi/4, pi/4) (the die is symmetric under 90 deg)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


class B601T1Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, obs_noise_cube: float = 0.0, friction_range=None, render_mode=None,
                 max_steps: int = 150, start_margin: float = 0.015, camera: str = "overview",
                 width: int = 480, height: int = 360, reward_version: int = 2):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.kin = mujoco.MjData(self.model)  # scratch data for IK
        self.n_sub = int(round(0.1 / self.model.opt.timestep))
        self.obs_noise_cube = float(obs_noise_cube)
        self.reward_version = int(reward_version)
        self.friction_range = friction_range
        self.max_steps = max_steps
        self.start_margin = start_margin
        self.render_mode = render_mode
        self.camera = camera
        self._renderer = None
        self._wh = (width, height)

        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 7)])
        self.lo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 7)], 0].copy()
        self.hi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 7)], 1].copy()
        self.fl_qadr = m.jnt_qposadr[m.joint("finger_left_slide").id]
        self.fr_qadr = m.jnt_qposadr[m.joint("finger_right_slide").id]
        self.cube_qadr = m.jnt_qposadr[m.joint("cube_free").id]
        self.cube_dadr = m.jnt_dofadr[m.joint("cube_free").id]
        self.tcp = m.site("tcp").id
        self.cube_geom = m.geom("cube_geom").id
        self.fing = (m.geom("finger_left_geom").id, m.geom("finger_right_geom").id)
        self.box_body = m.body("box").id
        self.table_geom = m.geom("table").id
        self.cube_body = m.body("cube").id

        b = LAYOUT["box"]
        self.box_xy = np.array(b["center_xy"])
        self.box_yaw = math.radians(b["long_axis_deg"])
        self.box_inner = np.array([b["outer_l"] / 2 - b["wall"], b["outer_w"] / 2 - b["wall"]])
        self.box_h = b["height"]
        self.hover = np.array([*self.box_xy, 0.10])
        self.cube_half = LAYOUT["cube"]["half"]
        self.base_friction = float(LAYOUT["cube"]["friction"])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (38,), np.float32)
        seed = np.array([-math.atan2(LOOK_TCP[1], LOOK_TCP[0]), 2.0, 1.8, -1.15, 0.0, 0.0])
        self._look_q = self._solve_ik(seed, LOOK_TCP, 0.0, iters=300)

    # ------------------------------------------------------------------ IK
    def _solve_ik(self, q0, pos, yaw, iters=8, damping=1e-2):
        k = self.kin
        q = np.array(q0, dtype=float)
        Rt = target_rotation(pos, yaw)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            k.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.model, k)
            mujoco.mj_comPos(self.model, k)
            p = k.site_xpos[self.tcp]
            R = k.site_xmat[self.tcp].reshape(3, 3)
            ep = pos - p
            Re = Rt @ R.T
            eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            err = np.concatenate([ep, 0.5 * eo])
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, k, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.arm_dadr], 0.5 * jacr[:, self.arm_dadr]])
            dq = 0.7 * J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    # ------------------------------------------------------------ helpers
    def tcp_pos(self):
        return self.data.site_xpos[self.tcp].copy()

    def tcp_yaw(self):
        R = self.data.site_xmat[self.tcp].reshape(3, 3)
        return math.atan2(R[0, 1], -R[1, 1])  # finger axis (tool y) = (sin, -cos) of yaw

    def cube_pos(self):
        return self.data.qpos[self.cube_qadr:self.cube_qadr + 3].copy()

    def cube_yaw(self):
        w, x, y, z = self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def gap(self):
        return float(self.data.qpos[self.fl_qadr] + self.data.qpos[self.fr_qadr])

    def _contacts(self):
        left = right = table_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g = {c.geom1, c.geom2}
            if self.cube_geom in g:
                left |= self.fing[0] in g
                right |= self.fing[1] in g
        return left, right

    def careful_release(self):
        r = getattr(self, "release", None)
        return r is not None and r[0] <= CAREFUL_RELEASE_MAX_Z and r[1] <= CAREFUL_RELEASE_MAX_HSPEED

    def in_box(self, p):
        rel = rotz(-self.box_yaw)[:2, :2] @ (p[:2] - self.box_xy)
        return bool(np.all(np.abs(rel) < self.box_inner) and p[2] < 0.06)

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        q = self._look_q + self.np_random.uniform(-0.02, 0.02, 6)
        d.qpos[self.arm_qadr] = q
        d.ctrl[:6] = q
        d.ctrl[6] = 0.0
        if "cell" in options:
            xy = cell_xy(options["cell"]) + np.asarray(options.get("jitter", (0.0, 0.0)))
            yaw = math.radians(options.get("yaw_deg", 0.0))
        else:
            o, ur, uc = grid_frame()
            u, v = self.np_random.uniform(0, 1, 2)
            marg = self.start_margin * self.np_random.uniform(-1, 1, 2)
            xy = o + u * ur + v * uc + marg
            yaw = self.np_random.uniform(0, math.pi / 2)
        d.qpos[self.cube_qadr:self.cube_qadr + 3] = [xy[0], xy[1], self.cube_half + 0.0005]
        d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        if self.friction_range is not None:
            mu = self.np_random.uniform(*self.friction_range)
        else:
            mu = options.get("friction", self.base_friction)
        m.geom_friction[self.cube_geom, 0] = mu
        self.friction = mu
        mujoco.mj_forward(m, d)
        for _ in range(20):  # settle
            mujoco.mj_step(m, d)
        self.q_cmd = d.qpos[self.arm_qadr].copy()
        self.target_pos = self.tcp_pos()
        self.target_yaw = self.tcp_yaw()
        self.prev_action = np.zeros(5, np.float32)
        self.t = 0
        self.ever_grasped = False
        self.max_stage = 0
        self.holding = False
        self.release = None  # (die centre z, horizontal speed) at the most recent release
        return self._obs(), self._info(False)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        tcp = self.tcp_pos()
        tgt = self.target_pos + 0.02 * a[:3]
        tgt = tcp + np.clip(tgt - tcp, -0.03, 0.03)  # no wind-up far from the measured TCP
        tgt = np.clip(tgt, [0.15, -0.45, 0.025], [0.60, 0.40, 0.30])
        self.target_pos = tgt
        self.target_yaw = float(np.clip(wrap(self.target_yaw + 0.15 * a[3]), -1.4, 1.4))
        q_des = self._solve_ik(self.q_cmd, tgt, self.target_yaw)
        dq = np.clip(q_des - self.q_cmd, -JOINT_VMAX * 0.1, JOINT_VMAX * 0.1)
        self.q_cmd = np.clip(self.q_cmd + dq, self.lo, self.hi)
        self.data.ctrl[:6] = self.q_cmd
        self.data.ctrl[6] = (a[4] + 1) / 2 * MAX_GAP / 2
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        self.prev_action = a.astype(np.float32)

        cube = self.cube_pos()
        left, right = self._contacts()
        grasped = left and right
        self.ever_grasped |= grasped
        if self.holding and not (left or right):
            v = self.data.qvel[self.cube_dadr:self.cube_dadr + 3]
            self.release = (float(cube[2]), float(np.linalg.norm(v[:2])))
        self.holding = (self.holding or grasped) and (left or right)
        success = self.in_box(cube) and not (left or right)
        if self.reward_version >= 3 and success:
            # v3: only a careful placement counts (the T1 protocol lowers the die before release)
            success = self.careful_release()
        d_reach = np.linalg.norm(self.tcp_pos() - cube)
        r = 0.2 * (1 - math.tanh(10 * d_reach))
        stage = 0
        if grasped:
            d_hover = np.linalg.norm(cube - self.hover)
            if self.reward_version == 1:
                r += 0.2 + 0.4 * (1 - math.tanh(5 * d_hover))
            else:
                lift = min(max((cube[2] - self.cube_half) / 0.07, 0.0), 1.0)
                r += 0.2 + 0.1 * lift + 0.4 * (1 - math.tanh(3 * d_hover))
            stage = 1 if cube[2] < 0.05 else 2
        if self.in_box(cube):
            r = max(r, 0.9)
            stage = 3
        terminated = False
        if success:
            r = float(self.max_steps - self.t + 1)
            terminated = True
            stage = 4
        self.max_stage = max(self.max_stage, stage)
        lost = np.linalg.norm(cube[:2] - np.array([0.38, 0.0])) > 0.6 or cube[2] < -0.05
        terminated |= bool(lost)
        truncated = self.t >= self.max_steps and not terminated
        return self._obs(), float(r), terminated, truncated, self._info(success, grasped)

    def _obs(self):
        d = self.data
        cube = self.cube_pos()
        if self.obs_noise_cube > 0:
            cube = cube + self.np_random.normal(0, self.obs_noise_cube, 3)
        cy = fold90(self.cube_yaw())
        ty = self.tcp_yaw()
        tcp = self.tcp_pos()
        left, right = self._contacts()
        o = np.concatenate([
            d.qpos[self.arm_qadr], [self.gap()], 0.1 * d.qvel[self.arm_dadr],
            tcp, [math.sin(ty), math.cos(ty)],
            cube, [math.sin(4 * cy), math.cos(4 * cy)], [fold90(ty - cy)],
            cube - tcp, self.hover - cube,
            [float(left and right), float(left), float(right)],
            self.prev_action,
        ])
        return o.astype(np.float32)

    def _info(self, success, grasped=False):
        return {"is_success": bool(success), "grasped": bool(grasped), "stage": self.max_stage,
                "release": getattr(self, "release", None), "careful_release": self.careful_release(),
                "friction": getattr(self, "friction", None)}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._wh[1], self._wh[0])
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register("B601T1-v0", entry_point="rebot_sim_rl.env:B601T1Env")
