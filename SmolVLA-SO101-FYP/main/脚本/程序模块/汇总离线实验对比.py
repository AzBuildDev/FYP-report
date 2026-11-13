"""复用已完成的留出动作评估，导出简单对照、表格和图；不重新训练、推理或控制硬件。"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from 管理离线实验对比 import compare_reports

ROOT = Path(__file__).resolve().parents[2]
JOINTS = ["肩部左右", "肩部升降", "肘部", "腕部俯仰", "腕部旋转", "夹爪"]


def markdown(result):
    protocol = result["protocol"]
    body = ["# 留出动作与简单对照", "", "复用已有留出推理结果，仅比较离线动作误差；不代表实机抓取成功率。", "",
            f"统一尺度取自 `{result['reference_model']}` 的训练段动作标准差。动作块长度 {protocol['chunk_size']}，"
            f"共 {len(protocol['windows'])} 个窗口；各行使用相同窗口。每块覆盖 {protocol['chunk_size'] / protocol['source']['fps']:.2f} 秒预测。", "",
            "| 方法 | 统一尺度平均误差 ↓ | 六关节原单位平均误差 ↓ | 相对保持当前位置 |", "|---|---:|---:|---:|"]
    for row in result["rows"]:
        ratio = row.get("relative_to_hold_position")
        body.append(f"| {row['name']} | {row['common_scale_mae']:.4f} | {row['raw_joint_mean_mae']:.4f} | "
                    + (f"{ratio:.2f} 倍" if ratio is not None else "—") + " |")
    body.extend(["", "| 方法 | " + " | ".join(JOINTS) + " |", "|---|" + "---:|" * 6])
    for row in result["rows"]:
        body.append(f"| {row['name']} | " + " | ".join(f"{value:.3f}" for value in row["mae_per_joint"]) + " |")
    body.extend(["", "单位为 LeRobot 归一化位置单位，身体 RANGE_M100_100、夹爪 RANGE_0_100，不能当作角度。", "",
                 "“保持当前位置”将窗口起点的六关节状态重复作为所有目标；“训练动作均值”只使用训练段统计。"
                 "两者均不读取验证窗口未来动作来构造预测，也不会发送机械臂命令。", "",
                 "![离线动作对比](动作误差对比.svg)", "", "## 限制", ""])
    body.extend(f"- {note}" for note in result["limitations"])
    if any(row.get("same_recording_batch") for row in result["rows"]):
        body.extend(["", "当前模型的验证段与训练段来自同一采集批次；仍需独立新批次与人工记录实机结果。"])
    body.extend(["", "部分窗口可能重叠；不计算统计显著性或把误差差异解释为任务成功率提升。", ""])
    return "\n".join(body)


def export_result(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "对比结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "对比指标.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["方法", "类型", "采样数", "统一尺度平均误差", "六关节原单位平均误差", *JOINTS])
        for row in result["rows"]:
            writer.writerow([row["name"], row["kind"], row["samples"], row["common_scale_mae"], row["raw_joint_mean_mae"], *row["mae_per_joint"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "svg.fonttype": "path"})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    rows = result["rows"]
    colors = ["#2166ac" if row["kind"] == "smolvla_model" else "#777777" for row in rows]
    bars = axes[0].barh([row["name"] for row in rows], [row["common_scale_mae"] for row in rows], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("统一训练动作标准差缩放的 MAE（越低越好）")
    axes[0].set_title("相同留出采样窗口")
    axes[0].bar_label(bars, fmt="%.4f", padding=4)
    axes[0].margins(x=0.25)
    for row in rows:
        axes[1].plot(JOINTS, row["mae_per_joint"], marker="o", label=row["name"])
    axes[1].set_ylabel("MAE（归一化位置单位，非角度）")
    axes[1].set_title("六关节动作误差")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("离线动作描述性对比 · 不代表实机成功率")
    figure.savefig(output / "动作误差对比.svg", metadata={"Date": None})
    svg = output / "动作误差对比.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    figure.savefig(output / "动作误差对比.png", dpi=160)
    plt.close(figure)
    (output / "对比说明.md").write_text(markdown(result), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", default=[ROOT / "运行记录/整段留出实验评估.json"])
    parser.add_argument("--reference-model", type=Path, default=ROOT / "模型/整段留出实验模型")
    args = parser.parse_args()
    result = compare_reports(ROOT, args.reports, args.reference_model)
    result["created_at"] = datetime.now().isoformat(timespec="seconds")
    output = ROOT / "运行记录/离线实验对比" / f"对比_{datetime.now():%Y%m%d_%H%M%S_%f}"
    export_result(result, output)
    # 只包含模型指纹、数据内容指纹与指标，不复制原始动作、视频或本机绝对路径。
    latest = ROOT / "运行记录/离线实验对比最新结果.json"
    latest.write_text(json.dumps({"status": "passed", "result_dir": str(output), "created_at": result["created_at"],
                                 "scope": result["scope"], "robot_commands_sent": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成对比表格和图：{output}")
    for row in result["rows"]:
        print(f"{row['name']}：统一尺度 MAE {row['common_scale_mae']:.4f}")
    print("只复用已完成推理结果并计算简单对照，没有重新训练、推理或连接硬件。")


if __name__ == "__main__":
    main()
