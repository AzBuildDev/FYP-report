"""从固定revision续传语音权重，按仓库附件清单核对大小和SHA256。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def segmented(url, partial, size):
    """并发下载剩余分段，保留原续传前缀；合并到另一暂存文件以便中断恢复。"""
    base = partial.stat().st_size if partial.exists() else 0
    if base > size:
        raise RuntimeError("暂存超出预期大小")
    chunks = partial.with_name(partial.name + ".分段")
    chunks.mkdir(exist_ok=True)
    chunk_size = 64 * 1024 * 1024
    spans = [(start, min(size, start + chunk_size)) for start in range(base, size, chunk_size)]

    def fetch(span):
        start, end = span
        path = chunks / f"{start}_{end}"
        for attempt in range(6):
            try:
                have = path.stat().st_size if path.exists() else 0
                if have > end - start:
                    raise RuntimeError("分段大小异常")
                if have == end - start:
                    return path
                offset = start + have
                with requests.get(url, headers={"Range": f"bytes={offset}-{end - 1}"}, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-{end - 1}/"):
                        raise RuntimeError("服务器未按请求返回分段")
                    with path.open("ab") as output:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                output.write(block)
                if path.stat().st_size != end - start:
                    raise RuntimeError("分段不完整")
                print(f"分段完成 {end}/{size}", flush=True)
                return path
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(40, 5 * (attempt + 1)))

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(fetch, spans))
    combined = partial.with_name(partial.name + ".合并")
    with combined.open("wb") as output:
        for source in ([partial] if partial.exists() else []) + paths:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
    return combined


def main():
    manifest = json.loads((ROOT / "文档/语音模型附件清单.json").read_text(encoding="utf-8"))
    sources = {s["directory"]: s for s in manifest["sources"]}
    dest = ROOT / "模型/语音指令"
    files = [item for item in manifest["files"] if item.get("source_verified")]

    def download(item):
        path = dest / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]:
            print("已校验 " + item["path"], flush=True)
            return item
        partial = path.with_name(path.name + ".part")
        source_dir, name = item["path"].split("/", 1)
        source = sources[source_dir]
        # 镜像只用于传输；信任固定revision及仓库中已有SHA256，不信任镜像内容。
        url = f"https://hf-mirror.com/{source['repository']}/resolve/{source['revision']}/{quote(name)}?download=true"
        if item["bytes"] > 1024 ** 3:
            combined = segmented(url, partial, item["bytes"])
            if combined.stat().st_size != item["bytes"] or sha256(combined) != item["sha256"]:
                raise RuntimeError("合并文件大小或SHA256不一致")
            combined.replace(path)
            print("已下载校验 " + item["path"], flush=True)
            return item
        error = None
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset != item["bytes"]:
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        append = offset > 0 and response.status_code == 206
                        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("续传位置不一致")
                        with partial.open("ab" if append else "wb") as output:
                            for block in response.iter_content(1024 * 1024):
                                if block:
                                    output.write(block)
                if partial.stat().st_size != item["bytes"] or sha256(partial) != item["sha256"]:
                    raise RuntimeError("大小或SHA256不一致，保留暂存排查")
                partial.replace(path)
                print("已下载校验 " + item["path"], flush=True)
                return item
            except Exception as exc:
                error = exc
                print(f"下载重试 {attempt + 1} {item['path']}: {type(exc).__name__}", flush=True)
        raise RuntimeError(item["path"]) from error

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(download, files))
    (dest / "桥接模型校验.json").write_text(json.dumps({"sources": manifest["sources"], "verified": verified}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("语音模型逐项校验完成", flush=True)


if __name__ == "__main__":
    main()
