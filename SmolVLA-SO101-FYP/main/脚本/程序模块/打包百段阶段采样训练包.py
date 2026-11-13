"""核验准备结果并生成可上传AutoDL的ZIP与SHA256；不启动训练。"""
import argparse
import json
from pathlib import Path
import zipfile

from 目标选择标注数据 import atomic_json, sha256_file

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT = PROJECT.parent / "AutoDL_纯100段_阶段采样_20261107"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = root.with_suffix(".zip")
    if archive.exists():
        raise ValueError("压缩包已存在，不覆盖；先核对现有结果")
    for name in ("数据检查.json", "阶段采样核验.json", "CPU批次对齐核验.json", "训练验证来源核验.json", "初始化与标注来源核验.json"):
        report = json.loads((root / name).read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            raise ValueError("准备核验尚未通过：" + name)
    readme = PROJECT / "文档/项目进展/纯100段阶段采样AutoDL操作_20261107.md"
    (root / "使用说明.md").write_text(readme.read_text(encoding="utf-8"), encoding="utf-8")
    excluded = {"文件清单.json", "数据检查.json", "阶段采样核验.json", "压缩包完整性验收.json"}
    entries = []
    extra = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(("输出/", "验收输出/", "日志/")):
            continue
        if relative in excluded:
            if relative != "文件清单.json" and relative != "压缩包完整性验收.json":
                extra.append(relative)
            continue
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_json(root / "文件清单.json", entries)
    print(f"清单已生成：{len(entries)}项", flush=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for entry in entries:
            output.write(root / entry["path"], root.name + "/" + entry["path"])
        for relative in extra + ["文件清单.json"]:
            output.write(root / relative, root.name + "/" + relative)
    print("ZIP写入完成，核验CRC……", flush=True)
    with zipfile.ZipFile(archive) as saved:
        bad = saved.testzip()
        if bad is not None:
            raise ValueError("ZIP CRC失败：" + bad)
        archive_files = len(saved.infolist())
    digest = sha256_file(archive)
    archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n", encoding="utf-8")
    receipt = {"status": "passed", "archive": str(archive), "bytes": archive.stat().st_size,
               "sha256": digest, "files": archive_files, "crc_verified": True,
               "dataset_episodes": 100, "previous_200_dataset_included": False,
               "training_started": False, "robot_commands_sent": 0}
    atomic_json(root / "压缩包完整性验收.json", receipt)
    atomic_json(PROJECT / "运行记录/纯100段阶段采样训练包准备_20261107.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
