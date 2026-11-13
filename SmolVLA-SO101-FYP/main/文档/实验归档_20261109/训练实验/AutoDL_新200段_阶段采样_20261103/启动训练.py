"""新200段阶段采样实验；正式配置由训练参数.json绑定，断点恢复不允许混用旧实验。"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def command(config,output,resume=False,stop_after=None,benchmark=False):
    values=[sys.executable,'-u',str(ROOT/'脚本/程序模块/新200段训练.py'),
            '--dataset-root',str(ROOT/'训练数据'),'--data-check',str(ROOT/'数据检查.json'),
            '--annotations',str(ROOT/'标注/阶段标注_200段.json'),'--output-dir',str(output)]
    for key,flag in [('steps','steps'),('batch_size','batch-size'),('warmup','warmup'),
                     ('save_every','save-every'),('validate_every','validate-every'),
                     ('validation_samples','validation-samples'),('seed','seed')]:
        values += ['--'+flag,str(config[key])]
    if config.get('continuous'): values += ['--continuous']
    if resume:values += ['--resume',str(output/'last')]
    if stop_after is not None:values += ['--stop-after',str(stop_after)]
    if benchmark:values += ['--benchmark-only']
    return values

def environment():
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
               PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false',MALLOC_ARENA_MAX='2',
               MALLOC_MMAP_THRESHOLD_='131072',MALLOC_TRIM_THRESHOLD_='131072')
    return env

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--benchmark-batch',type=int)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    config=json.loads((ROOT/'训练参数.json').read_text(encoding='utf-8'))
    output=ROOT/config['output_relative']
    if a.benchmark_batch:
        config['batch_size']=a.benchmark_batch
        output=ROOT/f'验收输出/benchmark_batch{a.benchmark_batch}'
    if a.smoke:
        config['validation_samples']=1
        output=ROOT/'验收输出/保存恢复实测'
    if a.benchmark_batch and (a.resume or a.smoke):
        p.error('显存实测使用独立输出，不接受resume/smoke')
    subprocess.run(command(config,output,a.resume,a.stop_after,bool(a.benchmark_batch)),
                   cwd=ROOT,env=environment(),check=True)
