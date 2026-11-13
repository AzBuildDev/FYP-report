"""窗口操作与替代子进程测试；不连接机械臂、相机或加载权重。"""
import queue
import json
import time
import threading
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import 实机推理窗口 as ui
import 模型选择自检 as fixtures
from 推理运行设置 import task_inference_settings


class FakeOutput:
    def __init__(self):
        self.lines = queue.Queue()

    def __iter__(self):
        while True:
            line = self.lines.get(timeout=5)
            if line is None:
                return
            yield line

    def close(self):
        pass


class FakeProcess:
    def __init__(self):
        self.stdout = FakeOutput()
        self.returncode = None

    def poll(self):
        return self.returncode

    def finish(self, code=0):
        self.returncode = code
        self.stdout.lines.put(None)


class WindowTests(unittest.TestCase):
    write = staticmethod(fixtures.SelectionTests.write)

    def setUp(self):
        fixtures.SelectionTests.setUp(self)
        self.window = tk.Tk()
        self.window.withdraw()
        self.addCleanup(self.destroy)
        self.calls = []

        def factory(command, **kwargs):
            self.calls.append((command, kwargs))
            self.process = FakeProcess()
            return self.process

        self.panel = ui.InferenceWindow(self.window, self.root, model=self.model, process_factory=factory)

    def destroy(self):
        if self.panel.process is not None:
            self.process.finish()
            self.panel.reader.join(2)
            self.panel.temporary.cleanup()
        try:
            self.window.after_cancel(self.panel.poll_id)
            self.window.destroy()
        except tk.TclError:
            pass

    def settle(self):
        deadline = time.monotonic() + 2
        while self.panel.process is not None and time.monotonic() < deadline:
            self.window.update()
            time.sleep(0.01)
        self.assertIsNone(self.panel.process)

    def test_window_opens_without_start_and_launch_keeps_chinese_paths(self):
        self.assertEqual(self.calls, [])
        self.panel.confirm_button.invoke()
        self.panel.confirm()
        self.assertEqual(len(self.calls), 1)
        command, kwargs = self.calls[0]
        self.assertIn("--window-start", command)
        self.assertIn("--task-inference", command)
        self.assertIn("--preview-directory", command)
        self.assertEqual(command[command.index("--model-dir") + 1], str(self.model.resolve()))
        self.assertEqual(command[command.index("--dataset-root") + 1], str(self.dataset.resolve()))
        self.assertEqual(kwargs["cwd"], self.root)
        self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertNotIn("shell", kwargs)
        self.assertEqual(str(self.panel.confirm_button["state"]), "disabled")
        self.process.finish()
        self.settle()
        self.assertEqual(str(self.panel.confirm_button["state"]), "normal")
        self.assertTrue(self.panel.output_path.is_file())

    def voice_result(self, target='BC', status='target'):
        voice = self.panel.voice
        voice.request_id = 'voice-launch'
        voice.request_epoch = voice.epoch
        voice.request_process = self.panel.process
        voice.messages.put(dict(status=status, target=target, transcript='抓蓝方块',
                                id='voice-launch'))
        voice.poll()

    def test_voice_starts_selected_model_once_and_records_origin(self):
        self.voice_result()
        self.assertEqual(len(self.calls), 1)
        command = self.calls[0][0]
        self.assertEqual(command[command.index('--target-object')+1], 'BC')
        self.assertEqual(command[command.index('--model-dir')+1], str(self.model.resolve()))
        self.assertIn('已自动启动', self.panel.voice.status.get())
        # 同一识别结果重复到达时，不得再启动或切换。
        self.panel.voice.messages.put(dict(status='target', target='RC', transcript='抓红方块',
                                             id='voice-launch'))
        self.panel.voice.poll()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.panel.target_object.get(), 'BC')
        self.process.finish()
        self.settle()
        first_line = self.panel.output_path.read_text(encoding='utf-8').splitlines()[0]
        request = json.loads(first_line.removeprefix('启动请求：'))
        self.assertEqual(request['source'], 'voice')
        self.assertEqual(request['transcript'], '抓蓝方块')

    def test_voice_does_not_start_without_specific_checkpoint(self):
        self.panel.model.set('')
        with patch.object(ui.messagebox, 'showerror') as popup:
            self.voice_result()
        self.assertEqual(self.calls, [])
        popup.assert_not_called()
        self.assertIn('尚未选择模型', self.panel.voice.status.get())

    def test_active_voice_switches_without_starting_second_process(self):
        self.panel.confirm()
        self.voice_result()
        self.assertEqual(len(self.calls), 1)
        request = json.loads((Path(self.panel.temporary.name)/'目标请求.json').read_text(encoding='utf-8'))
        self.assertEqual(request['target'], 'BC')
        self.assertEqual(request['source'], 'voice')

    def test_rejected_voice_never_starts_inference(self):
        self.voice_result(status='rejected')
        self.assertEqual(self.calls, [])

    def test_stop_requests_cooperative_cleanup_and_close_waits(self):
        self.panel.confirm()
        stop_path = self.panel.stop_path
        self.panel.close()
        self.assertTrue(stop_path.is_file())
        self.assertTrue(self.window.winfo_exists())
        self.assertIsNotNone(self.panel.process)
        self.process.finish()
        self.settle()
        self.assertFalse(stop_path.exists())
        with self.assertRaises(tk.TclError):
            self.window.winfo_exists()

    def test_one_click_toggle_and_actual_target_acknowledgement(self):
        self.assertIn("RC", self.panel.current_target_label.get())
        self.panel.switch_button.invoke()
        self.assertEqual(self.panel.target_object.get(), "BC")
        self.assertIn("BC", self.panel.current_target_label.get())
        self.assertEqual(self.calls, [])
        self.panel.confirm()
        command, _ = self.calls[-1]
        self.assertEqual(command[command.index("--target-object") + 1], "BC")
        self.panel.switch_button.invoke()
        request = json.loads((Path(self.panel.temporary.name)/"目标请求.json").read_text(encoding="utf-8"))
        self.assertEqual(request["target"], "RC")
        self.assertIn("BC", self.panel.current_target_label.get())  # 尚未获控制端确认。
        self.assertNotIn("placed", request)
        self.process.stdout.lines.put("目标状态：" + json.dumps({"current":"RC", "request":{"id":request["id"]}, "message":"旧动作已清空"}) + "\n")
        deadline = time.monotonic() + 1
        while self.panel.active_target != "RC" and time.monotonic() < deadline:
            self.window.update()
            time.sleep(.01)
        self.assertIn("RC", self.panel.current_target_label.get())
        self.assertIn("清空", self.panel.task_status.get())
        self.panel.switch_button.invoke()
        request2 = json.loads((Path(self.panel.temporary.name)/"目标请求.json").read_text(encoding="utf-8"))
        self.assertEqual(request2["target"], "BC")
        self.assertEqual(len(self.calls), 1)  # 切换没有重启实机子进程。
        self.process.finish()
        self.settle()
        self.assertEqual(str(self.panel.switch_button["state"]), "normal")

    def test_failed_child_shows_error_and_can_retry(self):
        self.panel.confirm()
        self.process.stdout.lines.put("模型加载失败\n")
        self.process.finish(1)
        self.settle()
        self.assertIn("推理中止", self.panel.status.get())
        self.assertIn("模型加载失败", self.panel.output.get("1.0", "end"))
        self.panel.confirm()
        self.assertEqual(len(self.calls), 2)

    def test_cleanup_error_is_visible_even_with_zero_exit(self):
        self.panel.confirm()
        self.process.stdout.lines.put("资源清理异常：gripper 扭矩读回失败\n")
        self.process.finish()
        self.settle()
        self.assertIn("资源清理异常", self.panel.status.get())

    def test_overload_and_unconfirmed_release_have_priority_on_nonzero_exit(self):
        self.panel.confirm()
        self.process.stdout.lines.put("异常停止：电机 gripper 故障状态读取失败：Overload error!\n")
        self.process.stdout.lines.put("资源清理异常：gripper 停扭矩读回未确认\n")
        self.process.finish(1)
        self.settle()
        self.assertIn("过载", self.panel.status.get())
        self.assertIn("切断从臂电源", self.panel.status.get())
        self.assertIn("停扭矩未确认", self.panel.status.get())

    def test_two_camera_preview_is_rendered_and_acknowledged(self):
        from PIL import Image
        self.panel.confirm()
        directory = Path(self.panel.temporary.name)
        image = Image.new("RGB", (1280, 480), "red")
        image.paste(Image.new("RGB", (640, 480), "blue"), (640, 0))
        image.save(directory / "当前预览.jpg")
        self.panel.refresh_preview()
        self.assertIsNotNone(self.panel.preview_photo)
        self.assertGreater(self.panel.preview_photo.width(), 1)
        self.assertIsNotNone(self.panel.wrist_preview_photo)
        self.assertEqual(self.panel.preview_frames[0].size, (640, 480))
        self.assertEqual(self.panel.preview_frames[1].size, (640, 480))
        self.assertGreater(self.panel.preview_frames[0].getpixel((320, 240))[0], 240)
        self.assertGreater(self.panel.preview_frames[1].getpixel((320, 240))[2], 240)
        self.assertTrue((directory / "预览已显示").is_file())
        self.assertTrue((directory / "窗口心跳").is_file())
        self.assertIn("实时画面", self.panel.camera_status.get())
        self.assertIn("上：桌面相机", self.panel.camera_status.get())
        self.assertIn("下：夹爪相机", self.panel.camera_status.get())
        self.process.finish()
        self.settle()
        self.assertIn("最后一帧", self.panel.camera_status.get())

    def test_corrupt_preview_does_not_acknowledge_ready(self):
        self.panel.confirm()
        directory = Path(self.panel.temporary.name)
        (directory / "当前预览.jpg").write_bytes(b"invalid jpeg")
        self.panel.refresh_preview()
        self.assertFalse((directory / "预览已显示").exists())

    def test_publisher_reaches_window_and_receives_display_ack(self):
        import numpy as np
        from 实机测试监视窗口 import run_embedded_monitor, wait_for_preview
        self.panel.confirm()
        directory = Path(self.panel.temporary.name)
        camera_stop, user_stop = threading.Event(), threading.Event()
        errors = []

        def publish():
            try:
                run_embedded_monitor(self.root, directory,
                                     lambda: (time.monotonic(), np.full((48, 128, 3), 100, dtype=np.uint8)),
                                     camera_stop, user_stop, lambda: "模拟双摄")
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=publish)
        thread.start()
        try:
            deadline = time.monotonic() + 2
            while not (directory / "预览已显示").exists() and time.monotonic() < deadline:
                self.window.update()
                time.sleep(0.01)
            self.assertTrue(wait_for_preview(directory, user_stop, timeout=0.1))
            self.assertIsNotNone(self.panel.preview_photo)
            self.assertEqual(errors, [])
        finally:
            camera_stop.set()
            thread.join(2)

    def test_invalid_input_does_not_start_child(self):
        self.panel.unlimited_check.invoke()
        for duration in ("nan", "-1", "inf", "abc"):
            with self.subTest(duration=duration), patch.object(ui.messagebox, "showerror") as error:
                self.panel.seconds.set(duration)
                self.panel.confirm()
                error.assert_called_once()
        self.assertEqual(self.calls, [])

    def test_camera_page_shows_live_motor_table_and_exports_shareable_log(self):
        self.panel.confirm()
        log = self.root / "运行记录/实机模拟日志.jsonl"
        log.write_text('{"type":"action","model_target":[15,15,15,15,15,15]}\n', encoding="utf-8")
        event = dict(t=1.5, inferences=3, inference_s=.42, sent_commands=20, limited_joints=6,
                     observed_state=[0.] * 6, model_target=[15.] * 6, sent_target=[1.] * 6,
                     limited=[True] * 6, fault_status={str(i): 0 for i in range(6)})
        self.panel.messages.put(f"运行日志：{log}\n")
        self.panel.messages.put(ui.PREFIX + json.dumps(event) + "\n")
        self.window.update()
        self.window.after_cancel(self.panel.poll_id)
        self.panel.poll()
        self.assertEqual(self.panel.action_log_path, log.resolve())
        self.assertIn("20 步", self.panel.live_summary.get())
        self.assertEqual(self.panel.telemetry.item("2", "values")[1:4], ("0.00", "15.00", "1.00"))
        self.assertIn("模型预测", self.panel.output.get("1.0", "end"))
        self.panel.export_log()
        archive = log.with_name(log.stem + "_完整报告.zip")
        self.assertTrue(archive.is_file())
        import zipfile
        with zipfile.ZipFile(archive) as bundle:
            self.assertIn("运行报告.txt", bundle.namelist())
            self.assertIn("model_target", bundle.read("运行报告.txt").decode("utf-8"))
        with patch.object(self.window, "clipboard_clear"), patch.object(self.window, "clipboard_append") as copy:
            self.panel.copy_log()
            self.assertIn("model_target", copy.call_args.args[0])
        self.process.finish()
        self.settle()

    def test_spawn_failure_keeps_window_usable(self):
        self.panel.process_factory = lambda *a, **kw: (_ for _ in ()).throw(OSError("模拟启动失败"))
        with patch.object(ui.messagebox, "showerror") as error:
            self.panel.confirm()
            error.assert_called_once()
        self.assertIsNone(self.panel.process)
        self.assertIsNone(self.panel.temporary)
        self.assertEqual(str(self.panel.confirm_button["state"]), "normal")

    def test_long_duration_reaches_child_without_an_upper_limit(self):
        self.panel.unlimited_check.invoke()
        self.panel.seconds.set("90")
        self.panel.confirm()
        command = self.calls[0][0]
        self.assertEqual(command[command.index("--seconds") + 1], "90.0")

    def test_continuous_mode_disables_duration_and_still_supports_stop(self):
        self.assertTrue(self.panel.unlimited.get())
        self.panel.seconds.set("ignored while continuous")
        self.assertEqual(str(self.panel.duration_entry["state"]), "disabled")
        self.panel.confirm()
        command = self.calls[0][0]
        self.assertEqual(command[command.index("--seconds") + 1], "0")
        self.panel.stop()
        self.assertTrue(self.panel.stop_path.is_file())
        self.process.finish()
        self.settle()
        self.assertEqual(str(self.panel.duration_entry["state"]), "disabled")
        self.panel.unlimited_check.invoke()
        self.assertEqual(str(self.panel.duration_entry["state"]), "normal")

    def test_task_duration_accepts_long_runs_and_explicit_continuous_mode(self):
        self.assertEqual(task_inference_settings().seconds, 30)
        self.assertIsNone(task_inference_settings(0).seconds)
        for value in (90, 600, 86400):
            self.assertEqual(task_inference_settings(value).seconds, value)
        for value in (float("inf"), float("nan"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                task_inference_settings(value)


if __name__ == "__main__":
    unittest.main()
