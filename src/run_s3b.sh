#!/usr/bin/env bash
set -euo pipefail
cd ${ER_ROOT:-.}/src
CUDA_VISIBLE_DEVICES=0 python run_stage3b.py
