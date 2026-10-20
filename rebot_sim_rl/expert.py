"""Scripted IK expert for T1-sim (uses privileged true cube pose).

Phases: approach above cube -> descend -> close -> lift -> carry above box
(rotating the finger axis to the box long side, as in the real protocol) ->
lower -> open -> retreat. Outputs actions in the env's action space so the same
trajectories can be used for behaviour cloning.
"""
import math

import numpy as np

from .env import fold90, wrap

OPEN = 0.9  # ~8.0 cm gap
CLOSED = -1.0


class ScriptedExpert:
    def __init__(self, env, noise: float = 0.0, rng=None):
        self.env = env
        self.noise = noise
        self.rng = rng or np.random.default_rng(0)
        # finger axis = tool y = (sin psi, -cos psi); align with the box long axis
        self.release_yaw = wrap(env.box_yaw - math.pi / 2)
        if abs(self.release_yaw) > math.pi / 2:
            self.release_yaw = wrap(self.release_yaw + math.pi)
        self.reset()

    def reset(self):
        self.phase = 0
        self.timer = 0
        self.grasp_yaw = None

    def _move(self, target, yaw, grip, gain=1.0):
        e = self.env
        dp = (target - e.tcp_pos()) / 0.02 * gain
        n = np.linalg.norm(dp)
        if n > 1.0:
            dp = dp / n
        dyaw = np.clip(wrap(yaw - e.tcp_yaw()) / 0.15, -1, 1)
        return np.array([*dp, dyaw, grip])

    def act(self):
        e = self.env
        cube = e.cube_pos()
        tcp = e.tcp_pos()
        if self.grasp_yaw is None:
            self.grasp_yaw = e.tcp_yaw() + fold90(e.cube_yaw() - e.tcp_yaw())
        ph = self.phase
        if ph == 0:  # above cube, open, aligned
            tgt = np.array([cube[0], cube[1], 0.12])
            a = self._move(tgt, self.grasp_yaw, OPEN)
            if np.linalg.norm(tgt - tcp) < 0.012 and abs(wrap(self.grasp_yaw - e.tcp_yaw())) < 0.05 and e.gap() > 0.07:
                self.phase = 1
        elif ph == 1:  # descend
            tgt = np.array([cube[0], cube[1], 0.032])
            a = self._move(tgt, self.grasp_yaw, OPEN)
            if abs(tgt[2] - tcp[2]) < 0.008:
                self.phase, self.timer = 2, 0
        elif ph == 2:  # close
            a = self._move(np.array([cube[0], cube[1], 0.032]), self.grasp_yaw, CLOSED, gain=0.5)
            self.timer += 1
            if self.timer >= 6:
                self.phase = 3
        elif ph == 3:  # lift
            tgt = np.array([tcp[0], tcp[1], 0.15])
            a = self._move(tgt, self.grasp_yaw, CLOSED)
            if tcp[2] > 0.13:
                self.phase = 4
        elif ph == 4:  # carry above box, rotate fingers along box long side
            tgt = np.array([*e.box_xy, 0.15])
            a = self._move(tgt, self.release_yaw, CLOSED)
            if np.linalg.norm(tgt[:2] - tcp[:2]) < 0.015 and abs(wrap(self.release_yaw - e.tcp_yaw())) < 0.08:
                self.phase = 5
        elif ph == 5:  # lower into the box mouth
            tgt = np.array([*e.box_xy, 0.09])
            a = self._move(tgt, self.release_yaw, CLOSED)
            if tcp[2] < 0.10:
                self.phase, self.timer = 6, 0
        elif ph == 6:  # release
            a = self._move(np.array([*e.box_xy, 0.09]), self.release_yaw, 0.6, gain=0.5)
            self.timer += 1
            if self.timer >= 5:
                self.phase = 7
        else:  # retreat upward
            a = self._move(np.array([*e.box_xy, 0.20]), self.release_yaw, 0.6)
        if self.noise > 0:
            a = a + self.rng.normal(0, self.noise, 5)
        return np.clip(a, -1, 1).astype(np.float32)
