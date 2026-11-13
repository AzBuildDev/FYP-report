"""记录每路主机到帧与采集进程状态；时间戳不代表相机曝光时间。"""
import json
import threading
import time


class CameraImages(dict):
    """保持模型图像字典接口，同时保留同一次快照的逐路时间戳。"""

    def __init__(self, images, timestamps, diagnostics):
        super().__init__(images)
        self.timestamps = timestamps
        self.diagnostics = diagnostics


class CameraDiagnostics:
    def __init__(self, specs, directory, log, *, clock=time.monotonic):
        self.specs, self.log, self.clock = specs, log, clock
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.started = clock()
        self.last_report = float('-inf')
        self.closed = False
        self.files = {key: (directory / f"{spec['label']}.jsonl").open('w', encoding='utf-8', buffering=1)
                      for key, spec in specs.items()}
        self.states = {key: dict(frames=0, last_frame_monotonic=None, last_gap_s=None,
                                 max_gap_s=0., image_shape=None, read_started_monotonic=None,
                                 last_read_s=None, errors=[], process=None, reader_ended=False)
                       for key in specs}

    def _write(self, key, event):
        self.files[key].write(json.dumps(event, ensure_ascii=False) + '\n')

    def update(self, key, kind, value):
        now = self.clock()
        with self.lock:
            if self.closed:
                return
            state = self.states[key]
            if kind == 'frame':
                stamp, image = value
                previous = state['last_frame_monotonic']
                if previous is not None:
                    state['last_gap_s'] = stamp - previous
                    state['max_gap_s'] = max(state['max_gap_s'], stamp - previous)
                state.update(frames=state['frames'] + 1, last_frame_monotonic=stamp,
                             image_shape=list(image.shape))
                return
            if kind == 'read_started':
                state['read_started_monotonic'] = value['monotonic']
                return
            if kind == 'read_complete':
                state.update(last_read_s=value['duration_s'], read_started_monotonic=None)
                return
            if kind == 'error':
                state['errors'].append(str(value))
                state['errors'] = state['errors'][-10:]
            elif kind in ('opened', 'process_exit', 'closed'):
                state['process'] = value
            elif kind == 'reader_end':
                state['reader_ended'] = True
            self._write(key, dict(type='capture_event', camera=key, label=self.specs[key]['label'],
                                  t=now - self.started, event=kind, detail=value))

    def read(self, frames):
        """由控制主线程调用；采集线程只更新内存，不争用总日志。"""
        now = self.clock()
        with self.lock:
            snapshot = {}
            for key, frame in frames.items():
                state, spec = self.states[key], self.specs[key]
                stamp = frame['time']
                snapshot[key] = {**state, 'errors': list(state['errors']), 'label': spec['label'],
                                 'device': spec['device'], 'index': spec['index'],
                                 'fourcc': spec['fourcc'], 'requested_fps': spec['fps'],
                                 'expected_shape': [spec['height'], spec['width'], 3],
                                 'driver_log': str(self.directory / f"{spec['label']}_驱动.log"),
                                 'snapshot_monotonic': now, 'frame_monotonic': stamp,
                                 'has_image': frame['image'] is not None,
                                 'host_frame_age_s': now - stamp if frame['image'] is not None else None,
                                 'read_in_progress_s': (now - state['read_started_monotonic']
                                                        if state['read_started_monotonic'] is not None else None)}
            if not self.closed and now - self.last_report >= .5:
                for key, state in snapshot.items():
                    self._write(key, dict(type='camera_status', camera=key, t=now-self.started, **state))
                self.log.write(json.dumps(dict(type='camera_status', t=now-self.started,
                                               timestamp_basis='host complete-frame arrival, not exposure',
                                               cameras=snapshot), ensure_ascii=False) + '\n')
                self.last_report = now
        return CameraImages({key: frame['image'] for key, frame in frames.items()},
                            {key: frame['time'] for key, frame in frames.items()}, snapshot)

    def fault(self, exception):
        details = getattr(exception, 'camera_details', None)
        if details is None:
            return
        event = dict(type='camera_fault', t=self.clock()-self.started, error=str(exception), **details)
        with self.lock:
            if self.closed:
                return
            for key in details['failed_cameras']:
                if key in self.files:
                    self._write(key, event)
            self.log.write(json.dumps(event, ensure_ascii=False) + '\n')

    def close(self):
        with self.lock:
            if self.closed:
                return
            for key, stream in self.files.items():
                self._write(key, dict(type='camera_end', state=self.states[key]))
                stream.close()
            self.closed = True
