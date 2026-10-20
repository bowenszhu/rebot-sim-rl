"""B601-RS T1 simulation environment, scripted expert and RL/BC utilities."""
import os
from pathlib import Path

# Small 6x6 linear algebra in the IK loop: multithreaded BLAS only adds contention,
# especially with many vectorised env workers.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "assets/b601/b601_t1_scene.xml"
