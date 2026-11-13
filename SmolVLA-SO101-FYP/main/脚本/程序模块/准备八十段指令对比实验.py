"""从已核验80段包生成独立实验，原数据/模型/正在运行训练不修改；不启动训练。"""
from pathlib import Path
import argparse,hashlib,json,shutil

HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=Path(r'D:\FYP\AutoDL_纯80段双物体_阶段采样_20261108')
DEFAULT_DEST=Path(r'D:\FYP\AutoDL_纯80段_指令对比_20261108')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('父版本代码发生变化，需要重新审查替换位置：'+old[:80])
    return text.replace(old,new,1)

def trainer_source(source):
    text=source
    text=replace_once(text,'from 阶段采样 import StageSampler, load_annotations, SAMPLING_VERSION',
        'from 阶段采样 import StageSampler, load_annotations, SAMPLING_VERSION\n'
        'from 目标指令对比 import ApproachPairs, VERSION, backward_instruction_term, evaluate_instruction_pairs, load_training_checkpoint')
    text=replace_once(text,'        policy = SmolVLAPolicy.from_pretrained(load_checkpoint, config=resume_cfg, local_files_only=True)',
        '        policy = load_training_checkpoint(SmolVLAPolicy,resume_cfg,load_checkpoint)')
    text=replace_once(text,"    args = parser.parse_args()",'''    parser.add_argument('--instruction-weight',type=float,default=0.1)
    parser.add_argument('--instruction-margin',type=float,default=0.02)
    parser.add_argument('--instruction-pairs',type=int,default=8)
    parser.add_argument('--instruction-microbatch',type=int,default=2)
    parser.add_argument('--instruction-min-motion',type=float,default=0.1)
    parser.add_argument('--instruction-ranking-weight',type=float,default=1.0)
    args = parser.parse_args()
    if not 0 <= args.instruction_weight <= 1 or not 0 < args.instruction_margin < 1:
        parser.error('对比权重需在0至1，间隔需在0至1之间')
    if not 0 <= args.instruction_ranking_weight <= 1:
        parser.error('对比排序权重必须在0至1')
    if args.instruction_pairs < 2 or args.instruction_pairs % 2 or args.instruction_microbatch < 1 or args.instruction_min_motion < 0:
        parser.error('对比数量必须为正偶数，microbatch为正，运动阈值非负')''')
    text=replace_once(text,"              'training_data': train_source, 'validation_data': val_source}",
        "              'instruction_objective': {'version':VERSION,'weight':args.instruction_weight,"
        "'margin':args.instruction_margin,'pairs':args.instruction_pairs,'microbatch':args.instruction_microbatch,"
        "'min_motion':args.instruction_min_motion,'ranking_weight':args.instruction_ranking_weight,'warmup_steps':args.warmup,"
        "'positive_anchor':1.0,'seed_mode':'fork_rng_step','arm_dimensions':5,"
        "'marker_inclusive':True,'selection_success_evaluated':False},\n"
        "              'training_data': train_source, 'validation_data': val_source}")
    text=replace_once(text,"    records_for_step = sampler.records",'''    records_for_step = sampler.records
    train_pairs = ApproachPairs(train_tables,annotations,stats['action']['std'],cfg.chunk_size,min_motion=args.instruction_min_motion)
    val_pairs = ApproachPairs(val_tables,annotations,stats['action']['std'],cfg.chunk_size,min_motion=args.instruction_min_motion)
    if info['features']['action']['names'] != ['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']:
        raise ValueError('对比关节顺序需要重新核对')
    emit({'event':'instruction_pools','train':train_pairs.report(),'validation':val_pairs.report()})
    def instruction_update(step):
        if args.instruction_weight == 0:
            return {'effective_weight':0.0,'pair_count':0}
        # 独立且可恢复的随机序列，开关对比项不会扰动主模仿项的噪声序列。
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(args.seed + step*100003 + 137)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return backward_instruction_term(policy,train_pairs,step,args.instruction_pairs,args.seed,
                    lambda records:make_batch(records,videos,keys,tasks,cfg.chunk_size),
                    args.instruction_weight,args.instruction_margin,args.warmup,args.instruction_microbatch,args.instruction_ranking_weight)''')
    # 先释放主模仿图，再进行小批对比的反传；正式训练和显存测试走同一逻辑。
    text=replace_once(text,"            loss.backward()\n            torch.nn.utils.clip_grad_norm_",
        "            loss.backward()\n            benchmark_imitation=float(loss.detach())\n            del batch,loss\n            instruction_metrics=instruction_update(update)\n            torch.nn.utils.clip_grad_norm_")
    text=replace_once(text,"            del batch,loss\n        result=", "        result=")
    text=replace_once(text,"'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2,",
        "'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2,\n"
        "                'instruction_objective':config['instruction_objective'],'last_instruction_metrics':instruction_metrics,\n"
        "                'trainer_sha256':config['trainer_sha256'],'objective_source_sha256':config['source_dependencies']['目标指令对比.py'],\n"
        "                'gpu_name':torch.cuda.get_device_name(0),")
    text=replace_once(text,"        loss.backward()\n        grad_norm =",
        "        loss.backward()\n        imitation_value=float(loss.detach())\n        del batch,loss\n        instruction_metrics=instruction_update(step)\n        grad_norm =")
    text=replace_once(text,"        recent.append(float(loss.detach()))\n        del batch, loss", "        recent.append(imitation_value)")
    text=replace_once(text,"'sampling': sampler.statistics(step), 'batch_size': args.batch_size,",
        "'sampling': sampler.statistics(step), 'instruction':instruction_metrics,'batch_size': args.batch_size,")
    text=replace_once(text,"            improved = validation['task_balanced_mae'] < best",'''            with torch.autocast('cuda',dtype=torch.bfloat16):
                validation['instruction_compatibility'] = evaluate_instruction_pairs(policy,val_pairs,
                    lambda records:make_batch(records,videos,keys,tasks,cfg.chunk_size),args.instruction_margin)
            improved = validation['task_balanced_mae'] < best''')
    compile(text,'百段阶段训练.py','exec')
    return text

LAUNCHER='''"""独立指令对比实验。默认仅打印；--run正式训练，--benchmark显存实测。"""
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
'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--destination',type=Path,default=DEFAULT_DEST)
    a=p.parse_args();src=a.source.resolve();dst=a.destination.resolve()
    if dst.exists():raise FileExistsError('独立实验目录已存在，不覆盖：'+str(dst))
    if src==dst or src in dst.parents or dst in src.parents:raise ValueError('父实验与新实验必须隔离')
    original=json.loads((src/'文件清单.json').read_text(encoding='utf-8'))
    checked=[]
    for entry in original:
        f=src/entry['path']
        if f.stat().st_size!=entry['size'] or sha(f)!=entry['sha256']:raise ValueError('父包校验失败：'+entry['path'])
        checked.append(entry)
    dst.mkdir()
    for name in ('训练数据','模型','上游框架','脚本','标注'):
        shutil.copytree(src/name,dst/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        print('COPIED',name,flush=True)
    for name in ('划分.json','数据检查.json','准备训练.py','安装依赖.sh'):
        shutil.copy2(src/name,dst/name)
    prep=dst/'准备训练.py'
    prep.write_text(prep.read_text(encoding='utf-8').replace('先在AutoDL查看启动训练.py打印的计划。','先在AutoDL查看启动指令对比训练.py打印的计划。'),encoding='utf-8')
    module=dst/'脚本/程序模块'
    for name in ('目标指令对比.py','测试目标指令对比.py'):
        shutil.copy2(HERE/name,module/name)
    trainer=module/'百段阶段训练.py'
    trainer.write_text(trainer_source(trainer.read_text(encoding='utf-8-sig')),encoding='utf-8')
    (dst/'启动指令对比训练.py').write_text(LAUNCHER,encoding='utf-8')
    config=json.loads((src/'训练参数_B.json').read_text(encoding='utf-8'))
    config.update(instruction_margin=0.02,instruction_pairs=8,instruction_microbatch=2,instruction_min_motion=0.1)
    for arm,weight,ranking in (('baseline',0.0,0.0),('anchor',0.1,0.0),('contrast',0.1,1.0)):
        write(dst/('训练参数_'+arm+'.json'),{**config,'instruction_weight':weight,'instruction_ranking_weight':ranking,'output_relative':'输出/'+arm+'_纯80段'})
    exp={'experiment':'纯80段_指令对比_20261108','status':'prepared_not_started','parent':str(src),
         'method':'experimental_counterfactual_flow_matching_margin_with_positive_anchor',
         'changed_variable':'auxiliary_objective_enabled; both arms share base sampler/batch/schedule',
         'note':'contrast对比anchor时只改变排序项；anchor保持相同额外正确窗口、错误指令前向和随机序列但不反传错误分支；baseline关闭所有辅助项。',
         'normal_sample_windows':config['batch_size']*config['steps'],
         'additional_counterfactual_pairs':config['instruction_pairs']*config['steps'],
         'batch_status':'继承452作为候选；新目标必须重新实测显存，不声称最快或最优',
         'marker_usage':'包含标记的接近窗口，只对比标记及之前的前5个关节；完整动作模仿仍保留',
         'claim':'并非官方SmolVLA训练配方；尚无实机改善证据，不保证解决干扰物依赖',
         'checkpoint_selection':'best_val仍按动作MAE；指令相容性另报，不能当选色成功率'}
    exp['resume_precision']='所有对照配置均在加载前转换训练参数到FP32，避免恢复时先截成BF16再转回FP32。'
    write(dst/'实验说明.json',exp)
    report=json.loads((dst/'数据检查.json').read_text(encoding='utf-8'))
    report['dataset_root']=str(dst/'训练数据');write(dst/'数据检查.json',report)
    manifest=[]
    for f in sorted(dst.rglob('*')):
        # 跨机器准备会重写报告中的绝对路径；数据本体仍逐文件进入清单。
        if f.is_file() and f.name!='数据检查.json':manifest.append({'path':f.relative_to(dst).as_posix(),'size':f.stat().st_size,'sha256':sha(f)})
    write(dst/'文件清单.json',manifest)
    print(json.dumps({'status':'prepared','destination':str(dst),'files':len(manifest),'training_started':False},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
