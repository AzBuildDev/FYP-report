"""持续续训、内存记录和保存后重启；用户PAUSE时保持停止。"""
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'输出/新200段_batch32_持续训练'
PAUSE, RESTART = OUTPUT/'PAUSE', OUTPUT/'MEMORY_RESTART.json'
STATE, JOURNAL = ROOT/'服务器部署状态.json', ROOT/'日志/内存恢复调度.jsonl'
METRICS = OUTPUT/'metrics.jsonl'

def record(state, **values):
    data = dict(state=state, time=datetime.now().isoformat(), batch_size=32,
                continuous=True, save_every=5000, **values)
    STATE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    with JOURNAL.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False)+'\n')

def owned_rss_mib(pid):
    lines = subprocess.check_output(['ps','-eo','pid=,ppid=,rss='], text=True).splitlines()
    rows = [tuple(map(int, l.split())) for l in lines if l.strip()]
    owned = {pid}
    for _ in range(8):
        extended = owned | {p for p,parent,rss in rows if parent in owned}
        if extended == owned:
            break
        owned = extended
    return sum(rss for p,parent,rss in rows if p in owned) / 1024

def terminate_group(child):
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()

attempt = 0
while True:
    if PAUSE.exists() and not RESTART.exists():
        record('paused_by_user')
        break
    if RESTART.exists():
        record('resuming_after_memory_maintenance', reason=json.loads(RESTART.read_text()))
        RESTART.unlink()
        PAUSE.unlink(missing_ok=True)
    with (ROOT/'日志/加速持续训练.log').open('a', encoding='utf-8', buffering=1) as log:
        child = subprocess.Popen([sys.executable,'-u',str(ROOT/'启动加速训练.py'),'--resume'],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stamp = METRICS.stat().st_mtime_ns
        progress = time.monotonic()
        forced = None
        record('running', launcher_pid=child.pid, recovery_attempt=attempt)
        while child.poll() is None:
            time.sleep(10)
            latest = METRICS.stat().st_mtime_ns
            if latest != stamp:
                stamp, progress = latest, time.monotonic()
            rss = owned_rss_mib(child.pid)
            if rss > 49152:
                forced = 'system_memory_exceeded_48GiB'
            elif time.monotonic()-progress > 900:
                forced = 'no_progress_for_15_minutes'
            if forced:
                record('recovering_stalled_process', reason=forced, rss_mib=rss)
                terminate_group(child)
                break
    if child.returncode == 0:
        status = json.loads((OUTPUT/'status.json').read_text())
        record(status['state'], step=status['step'])
        if RESTART.exists():
            attempt = 0
            continue
        break
    if forced or child.returncode == -9:
        attempt += 1
        if attempt <= 3:
            record('retrying_from_last_complete_checkpoint', reason=forced or 'killed', attempt=attempt)
            continue
    record('failed', returncode=child.returncode, reason=forced)
    break
