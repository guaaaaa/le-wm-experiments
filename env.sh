# LeWM environment — source this from the repo root:  source env.sh
# Sets the venv + the paths this setup uses (dataset/checkpoints on NAS, per team policy).

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/.venv/bin/activate"

# STABLEWM_HOME: dataset lives at $STABLEWM_HOME/datasets/, checkpoints at $STABLEWM_HOME/checkpoints/
export STABLEWM_HOME=/mnt/nas2/lewm
export MUJOCO_GL=egl                       # headless MuJoCo (eval envs)
export HF_HOME=/mnt/nas2/lewm/.hf-cache    # keep HF cache off the near-full home disk

echo "LeWM env ready:"
echo "  python         = $(python --version 2>&1)"
echo "  STABLEWM_HOME  = $STABLEWM_HOME"
echo "  dataset        = $STABLEWM_HOME/datasets/pusht_expert_train.h5"
echo "  released ckpt  = pusht/lewm  ->  $(readlink -f "$STABLEWM_HOME/checkpoints/pusht/lewm" 2>/dev/null)"
echo
echo "Pick a FREE GPU first:  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader"
