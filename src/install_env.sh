#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_er
python -V
pip install --no-input --upgrade pip
pip install --no-input torch --index-url https://download.pytorch.org/whl/cu126
pip install --no-input \
  numpy pandas pyarrow polars scipy scikit-learn \
  lightgbm \
  transformers sentence-transformers \
  rapidfuzz python-Levenshtein Unidecode \
  tqdm numba joblib psutil
pip install --no-input faiss-gpu-cu12
echo "=== VERIFY ==="
python - <<'PY'
import importlib
for m in ['numpy','pandas','polars','pyarrow','scipy','sklearn','lightgbm','faiss','torch','transformers','sentence_transformers','rapidfuzz','Levenshtein','unidecode','numba']:
    try:
        mod=importlib.import_module(m); print(f'{m:22s} {getattr(mod,"__version__","ok")}')
    except Exception as e: print(f'{m:22s} FAIL {e}')
import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())
import faiss; print('faiss gpus', faiss.get_num_gpus())
PY
