"""显式--run才训练；默认仅显示命令。A/B互用同一数据/预算但输出隔离。"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def command(config,arm,resume=False,stop_after=None):
    output=ROOT/config['output_relative']
    values=[sys.executable,'-u',str(ROOT/'脚本/程序模块/百段阶段训练.py'),
            '--dataset-root',str(ROOT/'训练数据'),'--data-check',str(ROOT/'数据检查.json'),
            '--annotations',str(ROOT/'标注/阶段标注_100段.json'),'--output-dir',str(output),
            '--sampling-mode',config['sampling_mode']]
    for key in ('steps','batch_size','warmup','save_every','validate_every','validation_samples','seed'):
        values+=['--'+key.replace('_','-'),str(config[key])]
    if resume:values+=['--resume',str(output/'last')]
    if stop_after is not None:values+=['--stop-after',str(stop_after)]
    return values

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('A','B'),default='B')
    p.add_argument('--run',action='store_true',help='实际启动训练；未指定则只打印计划')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int,help='保存后停止，供另行授权的GPU保存恢复验证')
    a=p.parse_args()
    config=json.loads((ROOT/f'训练参数_{a.arm}.json').read_text(encoding='utf-8'))
    values=command(config,a.arm,a.resume,a.stop_after)
    print(json.dumps({'arm':a.arm,'config':config,'command':values,'training_started':bool(a.run)},ensure_ascii=False),flush=True)
    if not a.run:return
    if not (ROOT/'数据检查.json').is_file():p.error('先执行准备训练.py完成数据与采样核验')
    subprocess.run([sys.executable,str(ROOT/'准备训练.py')],cwd=ROOT,check=True)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
        PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072')
    subprocess.run(values,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
