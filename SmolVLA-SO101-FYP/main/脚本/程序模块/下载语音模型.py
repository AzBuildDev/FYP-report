"""Download allowlisted pinned official weights; no user data is sent to an inference service."""
import argparse,json,os
from pathlib import Path
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_XET']='1'
ROOT=Path(__file__).resolve().parents[2]
MODELS={
 'Qwen2.5-1.5B-Instruct':('Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
 'whisper-tiny':('openai/whisper-tiny','169d4a4341b33bc18d8881c4b69c2e104e1cc0af'),
}
def main():
    from huggingface_hub import snapshot_download,HfApi
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);args=p.parse_args()
    repo,revision=MODELS[args.model]
    info=HfApi().model_info(repo,revision=revision,files_metadata=True)
    dest=ROOT/'模型/语音指令'/args.model
    snapshot_download(repo,revision=info.sha,local_dir=dest,
       allow_patterns=['*.json','*.safetensors','*.txt','*.model','README.md','LICENSE'],max_workers=4)
    print(json.dumps({'repo':repo,'revision':info.sha,'path':str(dest)},ensure_ascii=False))
if __name__=='__main__':main()
