"""独立后台训练；保存后可恢复内存维护，用户PAUSE/完成时保持停止。"""
from pathlib import Path
import subprocess,sys,json,os,signal,time
from datetime import datetime
ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/'训练参数.json').read_text(encoding='utf-8'))
OUTPUT=ROOT/config['output_relative'];LOG=ROOT/'日志'
LOG.mkdir(exist_ok=True)
def record(state,**kw):
    v={'time':datetime.now().isoformat(),'state':state,**kw}
    (ROOT/'服务器训练状态.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    with (LOG/'后台调度.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')

if __name__=='__main__':
    # 单实例锁防止重复启动训练，退出时由内核释放。
    import fcntl
    lock=(ROOT/'后台训练.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempts=0
    while True:
        pause,restart=OUTPUT/'PAUSE',OUTPUT/'MEMORY_RESTART.json'
        if pause.exists() and not restart.exists():record('paused_by_user');break
        if restart.exists():
            if not (OUTPUT/'last/training_state.pt').exists():raise RuntimeError('维护恢复缺少完整检查点')
            restart.unlink();pause.unlink(missing_ok=True)
        resume=(OUTPUT/'last/training_state.pt').exists()
        status=OUTPUT/'status.json'
        if resume and status.exists() and json.loads(status.read_text()).get('state')=='completed':
            record('completed');break
        cmd=[sys.executable,'-u',str(ROOT/'启动训练.py')]+(['--resume'] if resume else [])
        with (LOG/'阶段采样训练.log').open('a',encoding='utf-8',buffering=1) as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record('running',pid=child.pid,resume=resume,batch_size=config['batch_size'])
            child.wait()
        if child.returncode==0:
            state=json.loads(status.read_text())
            record(state['state'],step=state['step'])
            if restart.exists():attempts=0;continue
            break
        attempts+=1
        # 未知异常保留现场，避免OOM/数据错误反复消耗；诊断后才能恢复。
        record('failed',returncode=child.returncode,recovery_attempt=attempts)
        break
