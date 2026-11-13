"""用已存预测和虚拟时钟复现动作块替换；不导入串口、相机或模型。"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import 异步动作控制 as scheduler


ROOT = Path(__file__).resolve().parents[2]


class VirtualClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class VirtualStop:
    def __init__(self, clock):
        self.clock, self.stopped = clock, False

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.clock.now += seconds
        return self.stopped


def replay(actions, initial, latencies, execute_limit, duration=30., hz=20.):
    """固定同一预测，只隔离调度行为；反馈理想跟随，不代表真实机器人或抓取。"""
    clock = VirtualClock()
    stop = VirtualStop(clock)
    state = np.asarray(initial, dtype=np.float32).copy()
    events, sent, accepted = [], [], []

    class RecordedWorker:
        def __init__(self, unused_predict):
            self.pending = False
            self.count = 0

        def submit(self, current, images, stamp):
            self.pending = True
            self.latency = latencies[self.count % len(latencies)]
            self.ready = clock.monotonic() + self.latency
            self.stamp = stamp
            self.count += 1

        def poll(self):
            if not self.pending or clock.monotonic() < self.ready:
                return None
            self.pending = False
            return scheduler.Prediction(self.stamp, actions.copy(), self.latency)

        def close(self):
            pass

    def frame():
        return clock.monotonic(), np.full((2, 2, 3), 80, dtype=np.uint8)

    def send(target):
        state[:] = target
        sent.append(target.copy())
        return target.copy()

    def record(event):
        events.append(event)
        if event['type'] == 'control_step':
            accepted.append(event)

    # 只替换当前模块的时钟和后台工作器；实际运行原调度循环和目标限幅函数。
    with patch.object(scheduler, 'time', clock), patch.object(scheduler, 'InferenceWorker', RecordedWorker):
        stats = scheduler.run_async_control(
            lambda *_: None, lambda: state.copy(), frame, send, stop,
            np.full(6, -100.), np.full(6, 100.), duration=duration, hz=hz,
            max_delta=.5, execute_limit=execute_limit, on_event=record)
    selected = np.asarray([event['model_target'] for event in accepted])
    sent_array = np.asarray(sent)
    step_delta = np.diff(np.vstack([initial, sent_array]), axis=0)
    if not np.isfinite(sent_array).all() or np.abs(step_delta).max() > .50001:
        raise AssertionError('虚拟发送违反原限幅约束')
    if any(event['source_age_s'] >= len(actions) / hz for event in accepted):
        raise AssertionError('虚拟发送使用了超出有效期的动作')
    indices = [event['chunk_index'] for event in accepted]
    return dict(execute_limit=execute_limit, duration_s=duration, stats=stats,
                selected_index_range=[min(indices), max(indices)],
                selected_gripper_range=[float(selected[:, 5].min()), float(selected[:, 5].max())],
                sent_gripper_range=[float(sent_array[:, 5].min()), float(sent_array[:, 5].max())],
                max_sent_step_delta=float(np.abs(step_delta).max()),
                selected_gripper_below_30=int((selected[:, 5] < 30).sum()),
                command_tick_fraction=len(sent) / (duration * hz),
                first_selected_indices=indices[:30])


def main(run, offline_report, output):
    rows = [json.loads(line) for line in run.read_text(encoding='utf-8').splitlines()]
    offline = json.loads(offline_report.read_text(encoding='utf-8'))
    latencies = [row['inference_s'] for row in rows if row['type'] == 'async_chunk']
    selected = next(case for case in offline['cases'] if case['label'] == 'saved_live')
    predictions = selected['predictions']
    cases = []
    for prediction in predictions:
        actions = np.asarray(prediction['actions'], dtype=np.float32)
        first_close = next((i for i, action in enumerate(actions) if action[5] < 30), None)
        cases.append(dict(seed=prediction['seed'], first_gripper_below_30_index=first_close,
                          variants=[replay(actions, selected['state'], latencies, limit)
                                    for limit in (10, 20, 50)]))
    real_actions = [row for row in rows if row['type'] == 'action']
    report = dict(
        hardware_connected=False, camera_opened=False, motor_commands_sent=0,
        method='运行实际异步循环，虚拟时钟/工作器；固定已存动作块与理想跟随，仅隔离调度',
        limitations='固定预测忽略新状态、新画面和接触，不能评估真实任务成功率。20/50步仅离线对照，未改实机配置。',
        source_log=str(run), source_offline_report=str(offline_report),
        scheduler_sha256=hashlib.sha256(Path(scheduler.__file__).read_bytes()).hexdigest(),
        real_end=next(row for row in rows if row['type'] == 'end'),
        real_async_end=next(row for row in rows if row['type'] == 'async_end'),
        real_executed_index_range=[min(row['chunk_index'] for row in real_actions),
                                   max(row['chunk_index'] for row in real_actions)],
        simulations=cases)
    output.mkdir(parents=True, exist_ok=True)
    (output / '动作衔接复现.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 动作块衔接离线复现', '', report['method'], '', report['limitations'], '',
             '| 种子 | 每批上限 | 实际选中索引（从0计） | 选中目标最小夹爪开度 | 开度低于30的目标数 |',
             '|---|---|---|---|---|']
    for case in cases:
        for variant in case['variants']:
            lo, hi = variant['selected_index_range']
            lines.append(f"| {case['seed']} | {variant['execute_limit']} | {lo}～{hi} | "
                         f"{variant['selected_gripper_range'][0]:.2f} | {variant['selected_gripper_below_30']} |")
    lines.extend(['', '该稳定输入在第38～40步附近才预测明显闭合。当前逻辑新结果一到就替换旧块；'
                  '即便只把每批上限调成20或50，也可能在到达闭合段之前再次替换。'
                  '这能复现调度对该固定预测的截断，不能证明实际模型在其他状态下也完全相同。', '',
                  '后续应验证预测与已执行轨迹的衔接、状态条件及场景起点；不能直接据此扩大实机限速或回放过期动作。'])
    (output / '动作衔接复现.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-log', type=Path, required=True)
    parser.add_argument('--offline-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    options = parser.parse_args()
    main(options.run_log.resolve(), options.offline_report.resolve(), options.output_dir.resolve())
