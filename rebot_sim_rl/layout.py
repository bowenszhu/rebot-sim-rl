"""T1 start grid in the B601 base frame (actual 2026-10-15 setup)."""
import json
import math

import numpy as np

from . import ROOT

LAYOUT = json.loads((ROOT / "configs/t1_layout.json").read_text())
ROWS = ("A", "B", "C")
COLS = ("1", "2", "3")


def cell_xy(cell: str) -> np.ndarray:
    """Bilinear interpolation of a grid cell (e.g. 'B2') from the four measured corners."""
    c = LAYOUT["grid_corners_xy"]
    u = ROWS.index(cell[0]) / 2.0  # near -> far
    v = COLS.index(cell[1]) / 2.0  # left -> right
    a1, a3, c1, c3 = (np.array(c[k]) for k in ("A1", "A3", "C1", "C3"))
    near = a1 + v * (a3 - a1)
    far = c1 + v * (c3 - c1)
    return near + u * (far - near)


def eval_starts():
    """Fixed evaluation set: 9 printed grid points x yaw {0, 45} deg (T1 protocol)."""
    return [(r + c, yaw) for r in ROWS for c in COLS for yaw in (0.0, 45.0)]


def grid_frame():
    """Origin and unit axes of the grid (row axis near->far, col axis left->right)."""
    a1, c1, a3 = cell_xy("A1"), cell_xy("C1"), cell_xy("A3")
    return a1, c1 - a1, a3 - a1


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
