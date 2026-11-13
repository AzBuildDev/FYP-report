"""为GitHub答辩资料归档全部非权重运行记录与独立训练代码，生成哈希索引。"""
from pathlib import Path
import gzip,hashlib,json,shutil,re
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'文档/实验归档_20261109'
OUT.mkdir(exist_ok=True)
deny={'.pt','.safetensors','.lock','.pyc','.pyo'}
rows=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def copy(p,rel):
 dest=OUT/rel
 compressed=p.suffix in {'.jsonl','.log'} and p.stat().st_size>1024*1024
 if compressed:dest=dest.with_name(dest.name+'.gz')
 dest.parent.mkdir(parents=True,exist_ok=True)
 if compressed:
  with p.open('rb') as f,dest.open('wb') as out,gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as g:shutil.copyfileobj(f,g)
 else:shutil.copy2(p,dest)
 rows.append({'source':str(p),'path':dest.relative_to(OUT).as_posix(),'source_size':p.stat().st_size,'source_sha256':digest(p),'stored_size':dest.stat().st_size,'stored_sha256':digest(dest),'gzip':compressed})
for p in sorted((ROOT/'运行记录').rglob('*')):
 if p.is_file() and p.suffix.lower() not in deny and '__pycache__' not in p.parts:
  copy(p,Path('运行记录')/p.relative_to(ROOT/'运行记录'))
for package in sorted(ROOT.parent.glob('AutoDL_*')):
 if not package.is_dir():continue
 for p in sorted(package.rglob('*')):
  rel=p.relative_to(package)
  if not p.is_file() or any(x in rel.parts for x in ['模型','训练数据','上游框架','.venv','__pycache__','输出','显存实测']):continue
  if p.suffix.lower() in {'.py','.json','.jsonl','.log','.md','.sh','.txt','.csv'}:
   copy(p,Path('训练实验')/package.name/rel)
(OUT/'文件索引.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'files':len(rows),'stored_bytes':sum(r['stored_size'] for r in rows),'max_file':max(r['stored_size'] for r in rows)},ensure_ascii=False))
