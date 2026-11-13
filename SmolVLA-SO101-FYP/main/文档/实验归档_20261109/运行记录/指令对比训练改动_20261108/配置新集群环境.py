from pathlib import Path
import subprocess
v=Path('/root/autodl-tmp/指令对比环境')
subprocess.run(['/root/miniconda3/bin/python','-m','venv','--system-site-packages',str(v)],check=True)
packages=['transformers==4.51.3','huggingface-hub>=0.34.2,<1.0','tokenizers>=0.21,<0.22','datasets==3.6.0','draccus==0.10.0','diffusers>=0.27.2,<0.36','accelerate>=1.7,<2','safetensors>=0.4.3','einops>=0.8','num2words>=0.5.14','av','pyarrow','gymnasium>=0.29.1,<1','deepdiff>=7,<9','termcolor','opencv-python-headless','packaging','jsonlines>=4,<5','wandb>=0.20,<0.22','imageio[ffmpeg]>=2.34,<3','pyserial>=3.5']
subprocess.run([str(v/'bin/python'),'-m','pip','install',*packages],check=True)
subprocess.run([str(v/'bin/python'),'-c','import torch,transformers; print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))'],check=True)
