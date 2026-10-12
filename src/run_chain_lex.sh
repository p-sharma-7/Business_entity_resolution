#!/usr/bin/env bash
set -euo pipefail
export ER_TAG=_lex
export ER_DROP=107,110,111,112,113,114,115,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149
cd ${ER_ROOT:-.}/src
python run_train_gpu.py --rounds 5000
echo "=== LEX TRAINED ==="
