"""上传清单内训练文件到新GPU实例，保留来源并校验。"""
import json,tarfile,time
from pathlib import Path
from 指令对比集群操作 import connect
r=Path(r'D:\FYP\AutoDL_纯80段_指令对比_20261108')
archive=r.parent/'指令对比集群包_20261108.tar.gz'
manifest=json.loads((r/'文件清单.json').read_text(encoding='utf-8'))
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
    for rel in [e['path'] for e in manifest]+['文件清单.json','数据检查.json']:
        t.add(r/rel,arcname=rel,recursive=False)
print('包大小',archive.stat().st_size,flush=True)
c=connect()
try:
    s=c.open_sftp();last=[0]
    def progress(done,total):
        if time.monotonic()-last[0]>20 or done==total:
            print('上传',round(done/total*100,1),'%',flush=True);last[0]=time.monotonic()
    s.put(str(archive),'/root/autodl-tmp/指令对比集群包_20261108.tar.gz',callback=progress);s.close()
    ch=c.get_transport().open_session();ch.set_combine_stderr(True)
    ch.exec_command('/root/miniconda3/bin/python -u -')
    code="""from pathlib import Path
import tarfile
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108');r.mkdir(exist_ok=False)
with tarfile.open('/root/autodl-tmp/指令对比集群包_20261108.tar.gz') as t:t.extractall(r,filter='data')
print('解包完成')
"""
    ch.sendall(code.encode());ch.shutdown_write()
    print(ch.makefile('rb').read().decode(),flush=True)
    if ch.recv_exit_status():raise RuntimeError('远程解包失败')
finally:c.close()
