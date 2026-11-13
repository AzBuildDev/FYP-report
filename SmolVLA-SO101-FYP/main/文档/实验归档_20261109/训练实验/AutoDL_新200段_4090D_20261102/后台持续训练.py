"""在screen中运行持续训练，退出状态以真实检查点报告为准。"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

root=Path(__file__).resolve().parent
output=root/'输出/新200段_batch32_持续训练'
state=root/'服务器部署状态.json'
def write(status, **extra):
    state.write_text(json.dumps({'state':status,'batch_size':32,'continuous':True,
                    'save_every':5000,'updated_at':datetime.now().isoformat(),**extra},ensure_ascii=False,indent=2)+'\n')
write('starting_continuous_training')
with (root/'日志/加速持续训练.log').open('a',encoding='utf-8',buffering=1) as log:
    result=subprocess.run([sys.executable,'-u',str(root/'启动加速训练.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    write('failed',returncode=result.returncode)
else:
    actual=json.loads((output/'status.json').read_text())
    write(actual['state'],step=actual['step'])
