#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# 独立环境避免修改AutoDL镜像原环境。4090D使用CUDA12.8版PyTorch。
python -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# 部分社区镜像把该库装在基础环境；PyTorch新环境的相对RPATH无法找到它。
python -m pip install --ignore-installed --no-deps nvidia-cusparselt-cu12==0.6.3
python -m pip install 'transformers==4.51.3' 'huggingface-hub>=0.34.2,<1.0' 'tokenizers>=0.21,<0.22' 'datasets==3.6.0' 'draccus==0.10.0' 'diffusers>=0.27.2,<0.36' 'accelerate>=1.7,<2' 'safetensors>=0.4.3' 'einops>=0.8' 'num2words>=0.5.14' av pyarrow numpy pillow 'gymnasium>=0.29.1,<1' 'deepdiff>=7,<9' termcolor opencv-python-headless packaging
export PYTHONPATH="$PWD/上游框架/src"
python -m pip install 'jsonlines>=4,<5' 'wandb>=0.20,<0.22' 'imageio[ffmpeg]>=2.34,<3' 'pyserial>=3.5'
python -c 'import torch; from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print(torch.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported()'
