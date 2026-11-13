"""校验 Release 交接包并配置本机处理器路径；不连接硬件、不上传文件。"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def configure(root, check_only=False):
    root = root.resolve()
    manifest = json.loads((root / "交接清单.json").read_text(encoding="utf-8"))
    config = root / "模型/config.json"
    original = root / "模型/config.json.发布原件"
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("清单路径越过交接包目录")
        verify = original if path == config and original.exists() else path
        if not verify.is_file() or digest(verify) != item["sha256"]:
            raise ValueError(f"文件缺失或 SHA-256 不符：{item['path']}")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    if original.exists():
        previous = json.loads(original.read_text(encoding="utf-8"))
        if {k: v for k, v in cfg.items() if k != "vlm_model_name"} != {
                k: v for k, v in previous.items() if k != "vlm_model_name"}:
            raise ValueError("模型配置中存在处理器路径以外的修改")
    processor = root / "视觉语言处理器"
    if not check_only:
        if not original.exists():
            original.write_bytes(config.read_bytes())
        cfg["vlm_model_name"] = str(processor)
        # LeRobot 0.3.4 的配置读取使用系统编码；ASCII 转义兼容 Windows GBK。
        config.write_text(json.dumps(cfg, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"交接包校验通过；模型：{config.parent}")
    print("仅校验，没有修改配置。" if check_only else f"本机处理器路径已配置：{processor}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True, help="解压后包含交接清单.json 的目录")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure(args.package_root, args.check_only)


if __name__ == "__main__":
    main()
