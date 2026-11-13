from pathlib import Path
import json,datetime
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108/输出/contrast_纯80段')
if (r/'MEMORY_RESTART.json').exists():raise RuntimeError('存在内存重启标记，需先检查调度，未修改')
(r/'PAUSE').write_text(json.dumps({'reason':'用户准备暂停训练关机，保存最新模型到本机','requested_at':datetime.datetime.now().isoformat()},ensure_ascii=False))
print('PAUSE requested; trainer will validate/save and exit normally')
