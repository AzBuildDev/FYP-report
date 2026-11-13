"""独立指令对比实验。默认仅打印；--run正式训练，--benchmark显存实测。"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('baseline','anchor','contrast'),default='contrast')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true');mode.add_argument('--benchmark',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--batch-size',type=int,help='只允许显存实测覆盖；正式参数需写入配置与清单')
    a=p.parse_args()
    if a.resume and not a.run:p.error('--resume只用于正式--run')
    if a.stop_after is not None and not a.run:p.error('--stop-after只用于正式--run')
    if a.batch_size is not None and not a.benchmark:p.error('batch覆盖只用于显存测试')
    c=json.loads((ROOT/('训练参数_'+a.arm+'.json')).read_text(encoding='utf-8'))
    batch=a.batch_size or c['batch_size']
    out=ROOT/c['output_relative'] if not a.benchmark else ROOT/'显存实测'/f'{a.arm}_batch{batch}'
    cmd=[sys.executable,'-u',str(ROOT/'脚本/程序模块/百段阶段训练.py'),
         '--dataset-root',str(ROOT/'训练数据'),'--data-check',str(ROOT/'数据检查.json'),
         '--annotations',str(ROOT/'标注/阶段标注_80段.json'),'--output-dir',str(out)]
    for key in ('steps','warmup','save_every','validate_every','validation_samples','seed','sampling_mode',
                'instruction_weight','instruction_margin','instruction_pairs','instruction_microbatch','instruction_min_motion','instruction_ranking_weight'):
        cmd+=['--'+key.replace('_','-'),str(c[key])]
    cmd+=['--batch-size',str(batch)]
    if a.resume:cmd+=['--resume',str(out/'last')]
    if a.benchmark:cmd+=['--benchmark-only']
    if a.stop_after is not None:cmd+=['--stop-after',str(a.stop_after)]
    print(json.dumps({'arm':a.arm,'config':c,'command':cmd,'execution_requested':a.run or a.benchmark},ensure_ascii=False),flush=True)
    if not(a.run or a.benchmark):return
    subprocess.run([sys.executable,str(ROOT/'准备训练.py')],cwd=ROOT,check=True)
    if a.run and not a.resume:
        import torch
        def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        expected_objective={'version':'instruction_margin_v1','weight':c['instruction_weight'],
            'margin':c['instruction_margin'],'pairs':c['instruction_pairs'],'microbatch':c['instruction_microbatch'],
            'min_motion':c['instruction_min_motion'],'ranking_weight':c['instruction_ranking_weight'],
            'warmup_steps':c['warmup'],'positive_anchor':1.0,'seed_mode':'fork_rng_step','arm_dimensions':5,
            'marker_inclusive':True,'selection_success_evaluated':False}
        candidates=list((ROOT/'显存实测').glob(a.arm+'_batch*/显存吞吐实测.json'))
        valid=[]
        for path in candidates:
            b=json.loads(path.read_text())
            if (b.get('status')=='passed' and b.get('batch_size')==batch
                and b.get('instruction_objective')==expected_objective
                and b.get('trainer_sha256')==digest(ROOT/'脚本/程序模块/百段阶段训练.py')
                and b.get('objective_source_sha256')==digest(ROOT/'脚本/程序模块/目标指令对比.py')
                and b.get('gpu_name')==torch.cuda.get_device_name(0)
                and b.get('gpu_total_mib',0)-b.get('peak_gpu_reserved_mib',0)>=2048):valid.append(b)
        if not valid:raise RuntimeError('先对当前对比损失与batch做显存实测并保留至少2GiB余量')
    env=os.environ.copy()
    env.update(PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               PYTHONUTF8='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
