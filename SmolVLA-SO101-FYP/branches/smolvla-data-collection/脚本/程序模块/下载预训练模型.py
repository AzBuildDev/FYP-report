"""Download pinned official model artifacts for this LeRobot 0.3.4 checkout."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(root / "运行环境/缓存/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from huggingface_hub import snapshot_download

artifacts = [
    ("lerobot/smolvla_base", "3326b100334ffc0a0bd1ec27e3afb1cfa2a6000c", "基础预训练模型", ["config.json", "model.safetensors", "README.md"]),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467", "视觉语言处理器", ["*.json", "merges.txt", "vocab.json"]),
]
manifest = []
for repo, revision, folder, patterns in artifacts:
    target = root / "模型" / folder
    print("Downloading", repo, revision, "to", target, flush=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=target, allow_patterns=patterns, token=False, max_workers=3)
    manifest.append({"repo_id": repo, "revision": revision, "directory": str(target), "files": [str(p.relative_to(target)) for p in target.iterdir() if p.is_file()]})
(root / "运行记录/模型下载清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("DOWNLOAD_COMPLETE", flush=True)
