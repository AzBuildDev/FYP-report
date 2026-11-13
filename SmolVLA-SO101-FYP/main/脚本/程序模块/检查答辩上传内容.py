"""上传前检查文件大小、疑似凭据与Python语法；仅报告位置不回显内容。"""
from pathlib import Path
import re,json,gzip
r=Path(__file__).resolve().parents[2]
patterns=[re.compile(x.encode()) for x in [r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{30,}',r'hf_[A-Za-z0-9]{25,}',r'sk-[A-Za-z0-9]{25,}',r'(?i)(?:password|api_key|access_token)\s*[=:]\s*[\x22\x27][^\x22\x27\r\n]{8,}[\x22\x27]']]
findings=[];large=[];syntax=[];count=0
for base in [r/'脚本',r/'文档',r/'README.md']:
 for p in ([base] if base.is_file() else base.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts:continue
  if p.stat().st_size>=95*1024**2:large.append(str(p.relative_to(r)))
  if p.suffix not in {'.py','.md','.json','.jsonl','.log','.txt','.sh','.bat','.gz'}:continue
  b=gzip.open(p,'rb').read() if p.suffix=='.gz' else p.read_bytes()
  for pat in patterns:
   if pat.search(b):findings.append(str(p.relative_to(r)));break
  if p.suffix=='.py':
   try:compile(b,str(p),'exec')
   except SyntaxError as e:syntax.append({'file':str(p.relative_to(r)),'line':e.lineno})
  count+=1
print(json.dumps({'files_scanned':count,'potential_secrets':findings,'large':large,'syntax':syntax},ensure_ascii=False))
if findings or large or syntax:raise SystemExit(1)
