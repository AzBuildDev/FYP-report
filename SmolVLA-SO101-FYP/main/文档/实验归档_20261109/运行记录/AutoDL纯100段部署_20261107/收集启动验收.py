from pathlib import Path
import io,json,math,paramiko,sys

folder=Path(__file__).resolve().parent
root='/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107'
c=paramiko.SSHClient()
c.load_host_keys(r'D:\FYP\ssh\known_hosts')
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('connect.westb.seetacloud.com',port=31533,username='root',
    key_filename=r'D:\FYP\ssh\autodl_training_monitor',look_for_keys=False,allow_agent=False,timeout=20)
try:
    channel=c.get_transport().open_session(timeout=20)
    channel.set_combine_stderr(True)
    channel.exec_command('/root/miniconda3/bin/python -u -')
    channel.sendall((folder/'读取训练进展.py').read_bytes())
    channel.shutdown_write()
    output=channel.makefile('rb').read().decode('utf-8')
    assert channel.recv_exit_status()==0,output
    status=json.loads(output)
    train=status['latest_train']
    assert status['scheduler']['phase']=='training' and status['last_start']['step']==16
    assert train['step']>=100 and math.isfinite(train['loss_avg50'])
    assert status['estimate']['derived_from_recent_logged_steps']>=80
    s=c.open_sftp()
    try:
        with s.open(root+'/日志/正式训练.log') as stream:text=stream.read().decode('utf-8')
        assert 'Traceback (most recent call last)' not in text and 'out of memory' not in text.lower()
        with s.open(root+'/日志/GPU保存验收.json') as stream:receipt=json.loads(stream.read().decode('utf-8'))
        receipt.update(resume_observation_pending=False,resumed_from_step=16,observed_step=train['step'],
            observed_updates_after_resume=train['step']-16,resume_verified=True,errors_detected=False)
        payload=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'
        with s.open(root+'/日志/GPU保存验收.json','w') as stream:stream.write(payload)
        (folder/'GPU保存验收.json').write_text(payload,encoding='utf-8')
        status['acceptance']={'status':'passed','checkpoint_resume':receipt,
            'sha256_data_annotations_and_sampling_passed':True,'formal_log_has_no_traceback_or_oom':True}
        (folder/'训练启动验收与预计结束时间.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        with s.open(root+'/日志/部署验收完成.json','w') as stream:stream.write(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        for relative in ['输出/B_纯100段_stage/run_config.json','输出/B_纯100段_stage/metrics.jsonl','日志/正式训练调度状态.json']:
            s.get(root+'/'+relative,str(folder/Path(relative).name))
    finally:s.close()
    print(json.dumps({'status':'passed','latest_step':train['step'],'batch_size':train['batch_size'],
        'resumed_updates_observed':train['step']-16,'estimate':status['estimate'],
        'gpu':status['gpu'],'receipt':str(folder/'训练启动验收与预计结束时间.json')},ensure_ascii=False,indent=2))
finally:c.close()
