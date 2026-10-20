#!/usr/bin/env bash
# Fetch the B601-RS STL meshes referenced by assets/b601/rs_arm_upstream.xml and verify them
# against assets/b601/meshes.sha256.
#   tools/fetch_meshes.sh              # download from the public GitHub repository (pinned commit)
#   tools/fetch_meshes.sh ~/reBot      # or copy from a local reBot checkout (ReBot_Arm_web_RS submodule)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/assets/b601/meshes"
REPO=Yang-Ci/ReBot_Arm_DigitalTwin_RS   # formerly ReBot_Arm_web_RS
REV=d289fa7202ad7da9595a480b29397d70efacc4c1
SUB=rebotarm_ros2/src/rebotarm_mujoco_rs/models/meshes
mkdir -p "$DST"
while read -r _ f; do
  if [ $# -ge 1 ]; then
    cp "$1/ReBot_Arm_web_RS/$SUB/$f" "$DST/$f"
  elif [ ! -f "$DST/$f" ]; then
    curl -fsSL "https://raw.githubusercontent.com/$REPO/$REV/$SUB/$f" -o "$DST/$f"
  fi
done < "$ROOT/assets/b601/meshes.sha256"
(cd "$DST" && sha256sum -c --quiet "$ROOT/assets/b601/meshes.sha256") && echo "meshes OK in $DST"
