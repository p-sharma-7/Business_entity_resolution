#!/usr/bin/env bash
# Usage: tmux_run.sh <session_name> <command...>
# Runs command in a detached tmux session with logging to logs/<session>.log
set -euo pipefail
ROOT=${ER_ROOT:-.}
SESS="$1"; shift
LOG="$ROOT/logs/${SESS}.log"
mkdir -p "$ROOT/logs"
tmux kill-session -t "$SESS" 2>/dev/null || true
tmux new-session -d -s "$SESS" "bash -lc '
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_er
cd $ROOT
echo \"[START \$(date -Is)] $*\"
{ $* ; } 2>&1
echo \"[EXIT \$? at \$(date -Is)]\"
' 2>&1 | tee $LOG"
echo "session=$SESS log=$LOG"
