"""并发续传训练压缩包，整包SHA256核对后解包。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,time
from 指令对比集群操作 import connect
a=Path(r'D:\FYP\指令对比集群包_20261108.tar.gz')
dest='/root/autodl-tmp/指令对比集群包_20261108.tar.gz'
c=connect();s=c.open_sftp();prefix=s.stat(dest).st_size
if prefix>=a.stat().st_size:raise RuntimeError('请先检查远端完整包')
s.rename(dest,dest+'.prefix');s.close();c.close()
size=a.stat().st_size;step=(size-prefix+5)//6
def upload(i):
    start=prefix+i*step;end=min(size,start+step)
    if start>=end:return None
    c=connect();s=c.open_sftp();path=dest+f'.part{i}'
    try:
        with a.open('rb') as f,s.file(path,'wb') as g:
            g.set_pipelined(True);f.seek(start);left=end-start
            while left:
                data=f.read(min(1024*1024,left));g.write(data);left-=len(data)
        print('分片完成',i,end-start,flush=True)
        return path
    finally:s.close();c.close()
print('已保留前缀',prefix,'并发续传',size-prefix,flush=True)
with ThreadPoolExecutor(max_workers=6) as pool:
    results=list(pool.map(upload,range(6)))
digest=hashlib.sha256(a.read_bytes()).hexdigest()
c=connect()
try:
    ch=c.get_transport().open_session();ch.set_combine_stderr(True);ch.exec_command('/root/miniconda3/bin/python -u -')
    code='''from pathlib import Path
import shutil,hashlib,tarfile
d=Path(DEST)
with d.open('wb') as out:
 for p in PARTS:
  with open(p,'rb') as f:shutil.copyfileobj(f,out,1024*1024)
assert hashlib.sha256(d.read_bytes()).hexdigest()==DIGEST
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108');r.mkdir(exist_ok=False)
with tarfile.open(d) as t:t.extractall(r,filter='data')
for p in PARTS:Path(p).unlink()
print('整包SHA256通过，解包完成')
'''.replace('DEST',repr(dest)).replace('PARTS',repr([dest+'.prefix']+[p for p in results if p])).replace('DIGEST',repr(digest))
    ch.sendall(code.encode());ch.shutdown_write();print(ch.makefile('rb').read().decode(),flush=True)
    if ch.recv_exit_status():raise RuntimeError('校验解包失败')
finally:c.close()
