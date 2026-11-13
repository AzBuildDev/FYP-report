"""语音到有限RC/BC目标；不导入机械臂，不使用虚构场景通过旧dry-run协议。"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import wave
from 语音目标解析 import normalized_text, parse_target_command

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "语音事件："


def resolve_target(text, raw):
    from 语音任务协议 import strict_json, grounding
    if re.search(r"不|勿|别|禁止", normalized_text(text)):
        raise ValueError("中文指令含否定，请重新说一个明确的正向目标")
    spec = strict_json(raw)
    grounded = grounding(normalized_text(text), "zone_A")
    if spec["status"] != "parsed" or grounded["status"] != "parsed":
        raise ValueError("指令不明确或不受支持，请说抓红方块或抓蓝方块")
    if spec["task_id"] != grounded["task_id"]:
        raise ValueError("语言模型与独立语义校验不一致，不执行")
    targets = {"red_cube_to_zone_A": "RC", "blue_cube_to_zone_A": "BC"}
    if spec["task_id"] not in targets:
        raise ValueError("当前只支持红蓝方块放入演示目标区")
    return targets[spec["task_id"]]


def is_stop(text):
    compact = re.sub(r"[\s。.!！?？,，]", "", normalized_text(text)).lower()
    return compact in {
        "停止", "停止运行", "停止机械臂", "请停止", "停止一下", "先停一下",
        "停下来", "别动了", "不要动了", "stop", "stoprobot"}


def process_request(request, asr, classifier=None, recorder=None):
    """模型可注入；真人录音仅在显式record请求时进行。"""
    text = request.get("text")
    metrics = {}
    if text is None:
        audio = Path(request["audio"]) if "audio" in request else recorder()
        text, metrics = asr.transcribe(audio, request.get("language", "zh"))
        metrics["audio_path"] = str(audio)
    if is_stop(text):
        return {"status": "stop", "transcript": text, "metrics": metrics}
    if classifier is None:
        try:
            target = parse_target_command(text)
        except ValueError as exc:
            return {"status": "rejected", "message": str(exc), "transcript": text, "metrics": metrics}
        return {"status": "target", "target": target, "transcript": text,
                "parser": "finite_target_command", "metrics": metrics}
    if getattr(classifier, "semantic_json", False) is True:
        # 明确短口令直接本地解析，避免为简单目标额外付费和等待。
        try:
            target = parse_target_command(text)
            return {"status": "target", "target": target, "transcript": text,
                    "parser": "finite_target_command", "metrics": metrics}
        except ValueError:
            pass
        try:
            semantic, semantic_metrics = classifier.classify(normalized_text(text), "zone_A")
            metrics = {**metrics, **semantic_metrics}
            if semantic["intent"] == "stop":
                return {"status": "stop", "transcript": text, "semantic": semantic, "metrics": metrics}
            if semantic["intent"] != "switch_target" or semantic["confidence"] < 0.85:
                return {"status": "rejected", "message": "语义不够明确，请说明要抓红色还是蓝色方块",
                        "transcript": text, "semantic": semantic, "metrics": metrics}
            normalized = normalized_text(text).lower()
            if re.search(r"不|勿|别|禁止|ignore|system prompt|关节|舵机|多个|都要", normalized):
                raise ValueError("指令含否定、注入或多目标语义")
            if not re.search(r"抓|拿|取|夹|换|切|移|目标|放|pick|grab|take|move|switch|change", normalized):
                raise ValueError("没有明确的抓取或目标切换动作")
            colors = set(re.findall(r"红|蓝|red|blue", normalized))
            if len(colors) != 1:
                raise ValueError("颜色不明确或包含多个颜色")
            target = semantic["target"]
            expected = "RC" if ("红" in colors or "red" in colors) else "BC"
            if target != expected:
                raise ValueError("语义结果与转写中的颜色不一致")
            return {"status": "target", "target": target, "transcript": text,
                    "parser": "dashscope_semantic_json", "semantic": semantic, "metrics": metrics}
        except (RuntimeError, ValueError) as exc:
            return {"status": "rejected", "message": str(exc), "transcript": text,
                    "metrics": metrics}
    raw, llm_metrics = classifier.classify(normalized_text(text), "zone_A")
    try:
        target = resolve_target(text, raw)
    except ValueError as exc:
        return {"status": "rejected", "message": str(exc), "transcript": text,
                "raw_model_output": raw, "metrics": {**metrics, **llm_metrics}}
    return {"status": "target", "target": target, "transcript": text,
            "raw_model_output": raw, "metrics": {**metrics, **llm_metrics}}


def main():
    # Windows 的管道默认可能是 GBK；事件和请求协议始终使用 UTF-8。
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='strict')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--backend", choices=("local", "dashscope"))
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    log_dir = ROOT / "运行记录"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"语音指令_{time.time_ns()}.jsonl"

    def emit(event):
        event = {"time": time.time(), **event}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(PREFIX + json.dumps(event, ensure_ascii=False), flush=True)

    try:
        from 云端语音识别 import CloudASR, CloudSemanticClassifier, load_config
        backend = args.backend or load_config(ROOT)['backend']
        asr = None
        classifier = None
        if args.text is None:
            if backend == 'dashscope':
                emit({"status": "loading", "message": "正在准备百炼语音 API；点击录音后才上传音频"})
                asr = CloudASR(ROOT)
            else:
                import torch
                torch.set_num_threads(2)
                from 本地语音模型 import LocalASR
                emit({"status": "loading", "message": "正在CPU加载Whisper；明确口令直接解析，不加载Qwen"})
                asr = LocalASR(ROOT / "模型/语音指令/whisper-tiny", "cpu")
        if backend == 'dashscope':
            classifier = CloudSemanticClassifier(ROOT)
        emit({"status": "ready", "log": str(log_path), "backend": backend,
              "semantic": bool(classifier), "cloud_verified": False})
    except Exception as exc:
        emit({"status": "error", "message": str(exc)})
        return 2

    def record():
        import sounddevice as sd
        emit({"status": "recording", "message": "开始录音，请在5秒内说出指令"})
        samples = sd.rec(16000 * 5, samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        path = log_dir / f"语音录音_{time.time_ns()}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(samples.tobytes())
        return path

    def handle(request):
        try:
            if not isinstance(request, dict) or request.get("kind") not in ("record", "audio", "text"):
                raise ValueError("invalid_request")
            emit({"status": "processing", "id": request.get("id"), "message": "正在处理语音，请稍候……"})
            result = process_request(request, asr, classifier=classifier, recorder=record)
            emit({**result, "id": request.get("id")})
        except Exception as exc:
            emit({"status": "rejected", "id": request.get("id"), "message": str(exc)})

    if args.serve:
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except ValueError:
                emit({"status": "rejected", "message": "invalid_json"})
                continue
            handle(request)
    else:
        request = {"kind": "audio" if args.audio else "text", "language": args.language}
        request.update({"audio": str(args.audio)} if args.audio else {"text": args.text or ""})
        handle(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
