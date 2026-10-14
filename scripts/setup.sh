#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.12 .venv
uv sync --locked --no-build
.venv/bin/python -c 'import torch; assert torch.__version__ == "2.16.0.dev20261013+cu130"; assert torch.version.cuda == "13.0"; assert torch.cuda.is_available(); print(torch.__version__)'
