"""用录制数据做临时GPU反传及保存/加载验收，不连接机械臂，不修改原权重。"""
from pathlib import Path
import argparse,gc,hashlib,json,os,subprocess,sys,time

PROJECT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path(r'D:\FYP\AutoDL_纯80段_指令对比_20261108'))
    p.add_argument('--checkpoint',type=Path,default=PROJECT/'模型/纯80段双物体_阶段采样_20261108/step_001065')
    a=p.parse_args();exp=a.experiment.resolve()
    # 有实机worker时不给它争抢GPU。窗口本身不执行推理，可保持打开。
    command="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"
    listing=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20).stdout
    if '实机抓取测试.py' in listing:raise RuntimeError('检测到实机推理worker；请结束该轮后再运行GPU验收')
    os.environ.update(PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    sys.path[:0]=[str(exp/'脚本/程序模块'),str(exp/'上游框架/src')]
    import torch,numpy as np
    import 百段阶段训练 as training
    from 目标指令对比 import ApproachPairs,paired_instruction_loss,backward_instruction_term,swap_tasks,load_training_checkpoint
    from 阶段采样 import load_annotations
    from 微调预训练模型 import PreTrainedConfig,SmolVLAPolicy
    if not torch.cuda.is_available():raise RuntimeError('需要空闲CUDA显卡验收')
    report_dir=PROJECT/'运行记录/指令对比训练改动_20261108'
    report_dir.mkdir(parents=True,exist_ok=True)
    save_dir=report_dir/('GPU临时保存恢复_'+time.strftime('%H%M%S'))
    original_sha=training.file_sha256(a.checkpoint/'model.safetensors')
    info,tables,tasklist=training.read_data(exp/'训练数据',exp/'数据检查.json')
    split=json.loads((exp/'划分.json').read_text(encoding='utf-8'))
    anns,_=load_annotations(exp/'训练数据',exp/'标注/阶段标注_80段.json',split)
    train=training.select_tables(tables,split['train'])
    stats=training.training_stats(train)
    source=training.dataset_source(exp/'训练数据',info,train,tasklist)
    pool=ApproachPairs(train,anns,stats['action']['std'])
    cfg=PreTrainedConfig.from_pretrained(a.checkpoint,local_files_only=True)
    cfg.vlm_model_name=str(exp/'模型/视觉语言处理器');cfg.load_vlm_weights=False
    policy=load_training_checkpoint(SmolVLAPolicy,cfg,a.checkpoint).to('cuda').train()
    params=[x for x in policy.parameters() if x.requires_grad]
    for x in params:x.data=x.data.float()
    optimizer=torch.optim.AdamW(params,lr=1e-5,betas=cfg.optimizer_betas,eps=cfg.optimizer_eps,weight_decay=cfg.optimizer_weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:1.)
    rng=np.random.default_rng(42)
    keys=list(policy.config.image_features);tasks={t['task_index']:t['task'] for t in tasklist}
    cache=training.LockedVideoFrameCache(max_bytes=256*1024**2)
    videos={k:training.preload_frames(exp/'训练数据',info,tables,k,cache) for k in keys}
    factory=lambda records:training.make_batch(records,videos,keys,tasks,cfg.chunk_size)
    plan=pool.plan(1,2,42)
    batch=factory(pool.records(plan[:1]))
    wrong={**batch,'task':swap_tasks(batch['task'])}
    good_tokens=policy.prepare_language(batch)[0];bad_tokens=policy.prepare_language(wrong)[0]
    assert not torch.equal(good_tokens,bad_tokens),'红蓝文本编码意外相同'
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print('GPU_CHECK_START',torch.cuda.get_device_name(0),str(dtype),flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(123)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=dtype):
        imitation,_=policy(dict(batch))
    assert torch.isfinite(imitation)
    imitation.backward();imitation_value=float(imitation.detach());del imitation
    before=params[-1].detach().clone()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(999)
        with torch.autocast('cuda',dtype=dtype):
            aux=backward_instruction_term(policy,pool,1,2,42,factory,.1,.02,1,microbatch=1)
    assert all(x.grad is None or torch.isfinite(x.grad).all() for x in params)
    grad_norm=float(torch.nn.utils.clip_grad_norm_(params,cfg.optimizer_grad_clip_norm,error_if_nonfinite=True))
    optimizer.step();scheduler.step()
    changed=float((params[-1].detach()-before).abs().max());assert changed>0
    del before
    torch.manual_seed(456)
    noise=policy.model.sample_noise((1,cfg.chunk_size,cfg.max_action_dim),'cuda')
    diffusion_time=policy.model.sample_time(1,'cuda')
    def fingerprints(model):
        result={}
        for name,tensor in model.state_dict().items():
            data=tensor.detach().cpu().contiguous()
            result[name]={'dtype':str(data.dtype),'shape':list(data.shape),
                          'sha256':hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()}
        return result
    original_fingerprints=fingerprints(policy)
    # 往返一致性用FP32评估，避免BF16后端重建/舍入被误判成权重变化。
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        _,before_metrics=paired_instruction_loss(policy,batch,[plan[0]['cutoff']],noise=noise,time=diffusion_time)
    run_config={'annotation_sha256':training.file_sha256(exp/'标注/阶段标注_80段.json'),
                'batch_size':1,'seed':42,'purpose':'temporary_implementation_acceptance_not_formal_training',
                'instruction_objective':{'weight':.1,'margin':.02,'pairs':2,'microbatch':1}}
    training.save_checkpoint(save_dir,policy,optimizer,scheduler,rng,1,1.,run_config,source,stats,{'acceptance_only':True})
    print('TEMP_CHECKPOINT_SAVED',str(save_dir),flush=True)
    peak=torch.cuda.max_memory_allocated()/1024**2
    del policy,params,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
    cfg=PreTrainedConfig.from_pretrained(save_dir,local_files_only=True)
    cfg.vlm_model_name=str(exp/'模型/视觉语言处理器');cfg.load_vlm_weights=False
    restored=load_training_checkpoint(SmolVLAPolicy,cfg,save_dir).to('cuda')
    restored_params=[x for x in restored.parameters() if x.requires_grad]
    for x in restored_params:x.data=x.data.float()
    restored_fingerprints=fingerprints(restored)
    state_differences=[name for name in original_fingerprints if original_fingerprints[name]!=restored_fingerprints.get(name)]
    (report_dir/'保存恢复权重比对.json').write_text(json.dumps({'mismatches':state_differences,'tensors':len(original_fingerprints)},ensure_ascii=False,indent=2),encoding='utf-8')
    assert not state_differences,state_differences[:10]
    restored_optimizer=torch.optim.AdamW(restored_params,lr=1e-5,betas=cfg.optimizer_betas,eps=cfg.optimizer_eps,weight_decay=cfg.optimizer_weight_decay)
    saved=torch.load(save_dir/'training_state.pt',map_location='cpu',weights_only=False)
    assert saved['step']==1 and saved['run_config']==run_config
    restored_optimizer.load_state_dict(saved['optimizer'])
    assert restored_optimizer.state
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        _,after_metrics=paired_instruction_loss(restored,batch,[plan[0]['cutoff']],noise=noise,time=diffusion_time)
    diffs={k:abs(after_metrics[k]-before_metrics[k]) for k in before_metrics if not isinstance(before_metrics[k],bool)}
    assert max(diffs.values())<1e-5,diffs
    # 实际恢复后再做一次对比反传/更新，验证optimizer不仅能反序列化。
    restored_optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=dtype):
        resumed,_=paired_instruction_loss(restored,batch,[plan[0]['cutoff']],noise=noise,time=diffusion_time)
    resumed.backward()
    assert all(x.grad is None or torch.isfinite(x.grad).all() for x in restored_params)
    torch.nn.utils.clip_grad_norm_(restored_params,cfg.optimizer_grad_clip_norm,error_if_nonfinite=True)
    restored_optimizer.step()
    assert training.file_sha256(a.checkpoint/'model.safetensors')==original_sha
    result={'status':'passed','gpu':torch.cuda.get_device_name(0),'dtype':str(dtype),
            'imitation_loss':imitation_value,'instruction_metrics':aux,'gradient_norm':grad_norm,
            'parameter_changed_max':changed,'peak_allocated_mib':peak,'red_blue_tokens_different':True,
            'checkpoint_roundtrip_max_metric_difference':max(diffs.values()),'resumed_update_passed':True,
            'all_saved_tensors_identical':True,'tensors_checked':len(original_fingerprints),
            'original_weights_unchanged':True,'temporary_checkpoint':str(save_dir),
            'formal_training_started':False,'robot_commands':0,'grasp_success_tested':False,
            'limit':'本地小批验收；集群BF16大批显存和正式入口16步保存恢复仍需另验'}
    (report_dir/'GPU反传保存恢复验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    training.DECODE_POOL.shutdown(wait=True)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
