"""用独立时间戳、断帧和模拟采集进程验证逐路诊断；不访问相机或机械臂。"""
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from 检查控制输入 import validate_frame
from 相机运行诊断 import CameraDiagnostics
from 相机测试窗口 import named_capture


class CameraLogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.clock = 10.
        self.log = io.StringIO()
        specs = {key: dict(label=label, device=device, index=index, fourcc='YUY2', fps=20,
                           width=4, height=4)
                 for key, label, device, index in [('front', '桌面相机', 'front_device', 1),
                                                   ('wrist', '夹爪相机', 'wrist_device', 2)]}
        self.diag = CameraDiagnostics(specs, self.directory, self.log, clock=lambda: self.clock)
        self.addCleanup(self.diag.close)
        self.image = np.full((4, 4, 3), 80, dtype=np.uint8)

    def snapshot(self, front=9.9, wrist=9.8, front_image=True, wrist_image=True):
        return self.diag.read({key: dict(time=stamp, image=self.image if has_image else None)
                               for key, stamp, has_image in [('front', front, front_image),
                                                             ('wrist', wrist, wrist_image)]})

    def test_stale_wrist_identified_while_front_is_healthy(self):
        self.diag.update('wrist', 'read_started', {'monotonic': 9.3})
        images = self.snapshot(wrist=9.4)
        with self.assertRaisesRegex(RuntimeError, '夹爪相机.*0.600.*0.500') as caught:
            validate_frame(9.4, images, 10.)
        details = caught.exception.camera_details
        self.assertEqual(set(details['failed_cameras']), {'wrist'})
        self.assertAlmostEqual(details['cameras']['wrist']['read_in_progress_s'], .7)
        self.diag.fault(caught.exception)
        rows = [json.loads(line) for line in self.log.getvalue().splitlines()]
        self.assertEqual(rows[-1]['type'], 'camera_fault')
        self.assertEqual(rows[-1]['cameras']['front']['device'], 'front_device')
        self.assertIn('camera_fault', (self.directory/'夹爪相机.jsonl').read_text(encoding='utf-8'))
        self.assertNotIn('camera_fault', (self.directory/'桌面相机.jsonl').read_text(encoding='utf-8'))

    def test_both_failures_reported_with_their_own_reasons(self):
        images = self.snapshot(front=9., wrist_image=False)
        with self.assertRaises(RuntimeError) as caught:
            validate_frame(9., images, 10.)
        self.assertEqual(set(caught.exception.camera_details['failed_cameras']), {'front', 'wrist'})
        self.assertIn('桌面相机', str(caught.exception))
        self.assertIn('未收到画面', str(caught.exception))

    def test_capture_error_and_process_exit_are_not_dropped(self):
        self.diag.update('wrist', 'error', '模拟 USB 读取失败')
        self.diag.update('wrist', 'process_exit', {'returncode': 7})
        snapshot = self.snapshot()
        self.assertEqual(snapshot.diagnostics['wrist']['errors'], ['模拟 USB 读取失败'])
        self.assertEqual(snapshot.diagnostics['wrist']['process']['returncode'], 7)
        self.assertEqual(snapshot.diagnostics['front']['errors'], [])
        self.assertIs(validate_frame(9.8, snapshot, 10.), snapshot)

    def test_cadence_counters_and_reporting_are_bounded(self):
        for stamp in (9.8, 9.9):
            self.diag.update('front', 'frame', (stamp, self.image))
        self.snapshot()
        self.snapshot()
        self.assertEqual(len(self.log.getvalue().splitlines()), 1)
        self.clock += .6
        images = self.snapshot()
        self.assertEqual(images.diagnostics['front']['frames'], 2)
        self.assertAlmostEqual(images.diagnostics['front']['max_gap_s'], .1)
        self.assertEqual(len(self.log.getvalue().splitlines()), 2)

    def test_named_black_frame_and_invalid_timestamp(self):
        images = self.snapshot(front=10.1)
        images['wrist'] = np.zeros_like(self.image)
        with self.assertRaises(RuntimeError) as caught:
            validate_frame(9.8, images, 10.)
        self.assertIn('时间戳在检查时间之后', str(caught.exception))
        self.assertIn('夹爪相机', str(caught.exception))
        self.assertIn('黑帧', str(caught.exception))

    def test_driver_file_and_partial_frame_eof(self):
        events = []
        class Process:
            pid = 123
            stdout = io.BytesIO(b'partial')
            def poll(self): return None
            def terminate(self): pass
            def wait(self, timeout): return 0
        driver_path = self.directory/'夹爪相机_驱动.log'
        with patch('imageio_ffmpeg.get_ffmpeg_exe', return_value='模拟ffmpeg'), \
             patch('subprocess.Popen', return_value=Process()) as popen:
            named_capture(2, 'YUY2', 20, lambda kind, value: events.append((kind, value)),
                          threading.Event(), 'wrist_device', None, width=4, height=4,
                          log_path=driver_path)
        self.assertEqual(Path(popen.call_args.kwargs['stderr'].name), driver_path)
        eof = next(value for kind, value in events if kind == 'reader_end')
        self.assertEqual(eof['received_bytes'], 7)
        self.assertEqual(eof['expected_bytes'], 48)
        self.assertTrue(any(kind == 'error' for kind, value in events))
        self.assertFalse(any(kind == 'frame' for kind, value in events))


if __name__ == '__main__':
    unittest.main()
