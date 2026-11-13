"""保存集群实际配置、基准与状态；不下载模型或修改训练。"""
from pathlib import Path
import json,datetime
from 指令对比集群操作 import connect
base='/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108'
out=Path(r'D:\FYP\SmolVLA-SO101-FYP\运行记录\指令对比训练改动_20261108\新集群27362')
out.mkdir(exist_ok=True)
c=connect()
try:
    s=c.open_sftp()
    for rel in ['训练参数_contrast.json','文件清单.json','吞吐选择.json','后台指令对比训练.py','输出/contrast_纯80段/metrics.jsonl','输出/contrast_纯80段/run_config.json','日志/正式指令对比训练.log']:
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
        try:s.get(base+'/'+rel,str(target))
        except FileNotFoundError:print('尚未生成',rel)
    s.close()
finally:c.close()
(out/'连接与部署.json').write_text(json.dumps({'host':'connect.westd.seetacloud.com','port':27362,'root':base,'screen':'smolvla80contrast','time':datetime.datetime.now().isoformat(),'gpu':'NVIDIA RTX 6000D','notes':'正式训练已启动；旧50948端口不可用；实际进度见metrics，非永久监控。'},ensure_ascii=False,indent=2),encoding='utf-8')
print(out)
