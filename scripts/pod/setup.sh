#!/bin/bash
# Sets up a RunPod A100 (CUDA 13 driver) for Aquinas training and LiteRT export.
# Two venvs: /root/venv (training) and /root/venv-export (litert_torch export).
set -e
cd /root
python3 -m venv /root/bootstrap && /root/bootstrap/bin/pip install -q uv && export PATH=/root/bootstrap/bin:$PATH
uv venv -q -p 3.11 /root/venv && VIRTUAL_ENV=/root/venv uv pip install -q -r requirements-train.txt
uv venv -q -p 3.11 /root/venv-export && VIRTUAL_ENV=/root/venv-export uv pip install -q -r requirements-export.txt
/root/venv/bin/python -c "import torch, transformers, peft; print('train', torch.__version__, torch.cuda.is_available(), transformers.__version__)"
/root/venv-export/bin/python -c "import litert_torch, transformers; print('export ok', transformers.__version__)"
