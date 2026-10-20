"""Frame montage of three evaluation videos for the README (results/figures/failure_modes.png).

Rows: PPO from scratch holding the cube in the box mouth, BC never opening, and a BC→PPO "success"
that lets go while moving; four frames from the second quarter to the end of each clip.
"""
import argparse
import glob
from pathlib import Path

import imageio
import numpy as np

ROWS = ['results/videos/ppo_scratch_s0/noise0_fail_*_stage3.mp4',
        'results/videos/bc_n10_s2/noise0_fail_*yaw45*_stage2.mp4',
        'results/videos/bcppo_n10_s0/noise0_success_*.mp4']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('results/figures/failure_modes.png'))
    args = ap.parse_args()
    rows = []
    for pattern in ROWS:
        path = sorted(glob.glob(pattern))[0]
        frames = [f for f in imageio.get_reader(path)]
        idx = np.linspace(len(frames) * 0.25, len(frames) - 1, 4).astype(int)
        rows.append(np.concatenate([frames[i][::2, ::2] for i in idx], axis=1))
        print(path, len(frames), 'frames')
    imageio.imwrite(args.out, np.concatenate(rows, axis=0))


if __name__ == '__main__':
    main()
