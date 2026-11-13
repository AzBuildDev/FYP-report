"""从本机预测缓存导出延迟与执行长度的误差敏感性；不启动模型或真实设备。"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from 分析执行窗口误差 import analyze_report

ROOT = Path(__file__).resolve().parents[2]


def markdown(result):
    lines = ["# 执行窗口假设下的离线误差", "", "这是示范数据上的原始目标误差分析，不是实机动作回放或任务成功率。", "",
             f"模型 `{result['model']}`，{result['hz']} Hz，每块最多选择 {result['execute_limit']} 步。"
             "从 floor(延迟×频率) 开始，假设后续控制节拍恰好推进一步，期间没有新结果替换。", "",
             "| 情形 | 有效目标步数 | 过期窗口 | 模型误差 | 保持当前位置 | 训练动作均值 |", "|---|---:|---:|---:|---:|---:|"]
    for case in result["cases"]:
        values = [f"{row['common_scale_mae']:.4f}" if row["common_scale_mae"] is not None else "不计算" for row in case["rows"]]
        lines.append(f"| {case['name']} | {case['scored_targets']} | {case['expired_windows']} | " + " | ".join(values) + " |")
    lines.extend(["", "误差统一按训练段动作标准差缩放；统计按有效目标步数加权。延迟变化会选择不同的未来示范动作，"
                  "不能用曲线下降推断增加延迟会改善控制。", "", "![执行窗口误差](执行窗口误差.svg)", "",
                  "## 腕部旋转关节", "", "| 情形 | 模型 MAE | 保持当前位置 MAE |", "|---|---:|---:|"])
    for case in result["cases"]:
        values = [f"{row['mae_per_joint'][4]:.3f}" if row["mae_per_joint"] is not None else "不计算" for row in case["rows"][:2]]
        lines.append(f"| {case['name']} | " + " | ".join(values) + " |")
    lines.extend(["", "这里的关节 MAE 为 LeRobot 归一化位置单位，不是角度。", "", "## 适用范围", ""])
    lines.extend(f"- {note}" for note in result["notes"])
    lines.extend(["", f"离线预测来源日期：{result['source_evaluation_date']}。分析器本身只读取缓存，没有再次执行模型。", ""])
    return "\n".join(lines)


def export_result(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "执行窗口结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "执行窗口指标.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["情形", "方法", "有效目标步数", "过期窗口", "统一尺度平均误差", "腕部旋转MAE"])
        for case in result["cases"]:
            for row in case["rows"]:
                writer.writerow([case["name"], row["name"], case["scored_targets"], case["expired_windows"],
                                 row["common_scale_mae"], row["mae_per_joint"][4] if row["mae_per_joint"] else None])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "svg.fonttype": "path"})
    figure, axis = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    cases = [case for case in result["cases"] if case["scored_targets"]]
    if cases:
        for index in range(3):
            axis.plot(range(len(cases)), [case["rows"][index]["common_scale_mae"] for case in cases], marker="o", label=cases[0]["rows"][index]["name"])
        axis.set_xticks(range(len(cases)), [case["name"].replace("本次 CUDA 推理耗时假设", "本次 CUDA 耗时假设") for case in cases], rotation=12)
        axis.legend()
    else:
        axis.text(.5, .5, "全部预测块过期，不计算误差", transform=axis.transAxes, ha="center")
    axis.set_ylabel("统一训练动作标准差缩放的 MAE")
    axis.grid(axis="y", alpha=.25)
    axis.set_title("不同窗口选择下的离线误差 · 不代表实机控制或成功率")
    figure.savefig(output / "执行窗口误差.svg", metadata={"Date": None})
    svg = output / "执行窗口误差.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    figure.savefig(output / "执行窗口误差.png", dpi=160)
    plt.close(figure)
    (output / "执行窗口说明.md").write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, help="带 --save-predictions 缓存的留出报告；默认复用最近成功分析的报告")
    parser.add_argument("--delays", type=float, nargs="+", default=[0., .4, .8, 1.2, 2.5])
    parser.add_argument("--execute-limit", type=int, default=20)
    args = parser.parse_args()
    pointer = ROOT / "运行记录/执行窗口分析最新结果.json"
    if args.evaluation_report is None:
        if not pointer.exists():
            parser.error("首次需指定 --evaluation-report；原始评估先用 --save-predictions 保存本地缓存")
        previous = json.loads(pointer.read_text(encoding="utf-8"))
        if previous.get("status") != "passed" or not previous.get("evaluation_report"):
            parser.error("旧分析记录缺少原始报告路径，请指定 --evaluation-report")
        args.evaluation_report = Path(previous["evaluation_report"])
    result = analyze_report(ROOT, args.evaluation_report, args.delays, args.execute_limit)
    result["created_at"] = datetime.now().isoformat(timespec="seconds")
    output = ROOT / "运行记录/执行窗口分析" / f"分析_{datetime.now():%Y%m%d_%H%M%S_%f}"
    export_result(result, output)
    pointer.write_text(json.dumps({"status": "passed", "result_dir": str(output), "evaluation_report": str(args.evaluation_report.resolve()),
                 "scope": result["scope"], "robot_commands_sent": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        values = [row["common_scale_mae"] for row in case["rows"]]
        print(f"{case['name']}：{case['scored_targets']} 个目标，{values}")
    print(f"结果与图表：{output}。只分析本地缓存，未启动模型或真实设备。")
