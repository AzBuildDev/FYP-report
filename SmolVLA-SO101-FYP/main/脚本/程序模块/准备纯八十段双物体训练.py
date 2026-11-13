"""从已验收100段中复制前80段双物体，独立保留标注、划分和训练来源。"""
from pathlib import Path
from datetime import datetime
import collections
import json
import shutil
import subprocess
import sys
import tarfile
from 目标选择标注数据 import DatasetCatalog, atomic_json, sha256_file

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT.parent/'AutoDL_纯100段_阶段采样_20261107'
DEST=PROJECT.parent/'AutoDL_纯80段双物体_阶段采样_20261108'
CHECKPOINT=PROJECT/'模型/纯100段_阶段采样_20261107/step_003435'

def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def main():
    if DEST.exists():raise ValueError('独立目录已经存在，拒绝覆盖：'+str(DEST))
    if shutil.disk_usage(DEST.parent).free<10*1024**3:raise ValueError('本机可用空间不足10GiB')
    manifest={x['path']:x for x in json.loads((SOURCE/'文件清单.json').read_text())}
    source_annotation=SOURCE/'标注/阶段标注_100段.json'
    original=json.loads(source_annotation.read_text())
    checkpoint=json.loads((CHECKPOINT/'checkpoint_summary.json').read_text())
    if sha256_file(source_annotation)!=checkpoint['run_config']['annotation_sha256']:
        raise ValueError('100段标注与已经部署的检查点来源不同')
    selected=set(range(80))
    episodes=[x for x in jsonl(SOURCE/'训练数据/meta/episodes.jsonl') if x['episode_index'] in selected]
    if {x['episode_index'] for x in episodes}!=selected:raise ValueError('前80段不完整')
    info=json.loads((SOURCE/'训练数据/meta/info.json').read_text())
    oldcheck=json.loads((SOURCE/'数据检查.json').read_text())
    if oldcheck['status']!='passed':raise ValueError('原100段未通过完整解码检查')
    DEST.mkdir()
    for folder in ('脚本/程序模块','训练数据/meta','标注','日志'):(DEST/folder).mkdir(parents=True)
    # 原训练器支持任意完整段数；复用相同实现，单独改变数据/标注和运行参数文件。
    shutil.copytree(SOURCE/'脚本/程序模块',DEST/'脚本/程序模块',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    test=DEST/'脚本/程序模块/测试阶段采样.py'
    text=test.read_text().replace('阶段标注_100段.json','阶段标注_80段.json')
    text=text.replace('for batch in (16, 32, 64):','for batch in (16, 32, 64, 140):')
    text=text.replace('tested_batches=[16, 32, 64]','tested_batches=[16, 32, 64, 140]')
    test.write_text(text,encoding='utf-8')
    print('复制独立训练数据和固定基础模型……',flush=True)
    for name in ('模型','上游框架'):
        shutil.copytree(SOURCE/name,DEST/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    copied=[]
    for ep in sorted(selected):
        paths=[info['data_path'].format(episode_chunk=ep//info['chunks_size'],episode_index=ep)]
        paths += [info['video_path'].format(episode_chunk=ep//info['chunks_size'],episode_index=ep,video_key=k)
                  for k,v in info['features'].items() if v['dtype']=='video']
        for relative in paths:
            key='训练数据/'+relative
            src=SOURCE/key;dst=DEST/key;expected=manifest[key]
            if src.stat().st_size!=expected['size'] or sha256_file(src)!=expected['sha256']:
                raise ValueError('原数据与完整解码时的文件清单不符：'+key)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            if sha256_file(dst)!=expected['sha256']:raise ValueError('复制校验失败：'+key)
            copied.append(key)
    info.update(total_episodes=80,total_frames=sum(x['length'] for x in episodes),total_videos=160,splits={'train':'0:80'})
    atomic_json(DEST/'训练数据/meta/info.json',info)
    for name,rows in [('episodes.jsonl',episodes),('episodes_stats.jsonl',[x for x in jsonl(SOURCE/'训练数据/meta/episodes_stats.jsonl') if x['episode_index'] in selected])]:
        (DEST/'训练数据/meta'/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    shutil.copy2(SOURCE/'训练数据/meta/tasks.jsonl',DEST/'训练数据/meta/tasks.jsonl')
    catalog=DatasetCatalog(DEST/'训练数据')
    annotations={**original,'created_at':datetime.now().astimezone().isoformat(),
        'dataset_root':str(catalog.root),'dataset_fingerprint':catalog.fingerprint,'count':80,
        'episodes':[x for x in original['episodes'] if x['episode_index'] in selected],
        'subset_source':{'annotation_sha256':sha256_file(source_annotation),'source_dataset_fingerprint':original['dataset_fingerprint'],
                         'source_episode_indices':list(range(80)),'marker_frames_unchanged':True}}
    atomic_json(DEST/'标注/阶段标注_80段.json',annotations)
    oldsplit=json.loads((SOURCE/'划分.json').read_text())
    split={'seed':oldsplit['seed'],'train':[x for x in oldsplit['train'] if x in selected],
           'validation':[x for x in oldsplit['validation'] if x in selected],
           'method':'保留100段实验前80段的原有划分；64训练16验证；不重新随机分配；布局配对未核对，不声称布局独立验证'}
    if len(split['train'])!=64 or len(split['validation'])!=16:raise ValueError('划分数量错误')
    atomic_json(DEST/'划分.json',split)
    run=checkpoint['run_config']
    config={k:run[k] for k in ('batch_size','steps','warmup','save_every','validate_every','seed')}
    config.update(validation_samples=run['validation_samples_per_episode'],sampling_mode='stage',
                  output_relative='输出/B_纯80段双物体_stage',initialization='official_base_fresh_optimizer',continuous=False)
    atomic_json(DEST/'训练参数_B.json',config)
    launch=(SOURCE/'启动训练.py').read_text().replace('阶段标注_100段.json','阶段标注_80段.json')
    (DEST/'启动训练.py').write_text(launch,encoding='utf-8')
    shutil.copy2(SOURCE/'准备训练.py',DEST/'准备训练.py')
    shutil.copy2(SOURCE/'安装依赖.sh',DEST/'安装依赖.sh')
    experiment={'experiment':'纯80段双物体阶段采样_20261108','selected_original_video_indices':list(range(80)),
        'excluded_original_video_indices':list(range(80,100)),'scene_source':'按用户已确认的前80段双物体范围筛选；不以算法自动猜测场景',
        'control_experiment':'与100段实验保持batch140、6858更新、采样/优化器/初始化/学习率；改变数据子集并按训练子集重算归一化',
        'markers':'保留原80段标记帧号，不重新解释接触/合爪定义',
        'train_episodes':64,'validation_episodes':16,'train_target_counts':dict(collections.Counter(catalog.episodes[x]['target'] for x in split['train'])),
        'validation_target_counts':dict(collections.Counter(catalog.episodes[x]['target'] for x in split['validation'])),
        'sampled_windows':config['batch_size']*config['steps'],'initialization':'official_base_fresh_optimizer',
        'validation':'动作误差；不是选色成功率；数据量及验证集改变，跨数据集MAE不能直接认定性能优劣',
        'parent_annotation_sha256':sha256_file(source_annotation),'training_started':False}
    atomic_json(DEST/'实验说明.json',experiment)
    report={**oldcheck,'dataset_root':str(catalog.root),'total_episodes':80,'total_frames':info['total_frames'],
            'episodes':[x for x in oldcheck['episodes'] if x['episode'] in selected],
            'warnings':[x for x in oldcheck['warnings'] if any(x.startswith(f'episode {ep}:') for ep in selected)],
            'verification_method':'原100段完整逐帧解码结果继承；本次80段全部Parquet和双摄视频逐文件SHA256与原清单及复制目标匹配',
            'parent_full_decode_report_sha256':sha256_file(SOURCE/'数据检查.json'),'checked_at':datetime.now().astimezone().isoformat()}
    atomic_json(DEST/'数据检查.json',report)
    atomic_json(DEST/'子集来源核验.json',{'status':'passed','files_verified':len(copied),'original_indices_retained':True,
        'single_object_files_included':False,'marker_frames_unchanged':True,'parent_annotation_sha256':sha256_file(source_annotation)})
    (DEST/'使用说明.md').write_text('# 纯80段双物体独立训练\n\n仅视频0～79，保留标记，64训练／16验证。\n\n检查：`python 准备训练.py`\n\n训练：`python 启动训练.py --arm B --run`\n\n恢复：追加 `--resume`。独立输出，保持100段实验。需要GPU可用后才能启动。\n',encoding='utf-8')
    excluded={'文件清单.json','数据检查.json','阶段采样核验.json'}
    entries=[]
    for path in sorted(DEST.rglob('*')):
        rel=path.relative_to(DEST).as_posix()
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc' and rel not in excluded and not rel.startswith('日志/'):
            entries.append({'path':rel,'size':path.stat().st_size,'sha256':sha256_file(path)})
    atomic_json(DEST/'文件清单.json',entries)
    subprocess.run([sys.executable,str(DEST/'准备训练.py')],cwd=DEST,check=True)
    # 集群已有已校验的大文件；只上传小文件，部署时逐文件核验并独立复制其余源文件。
    transfer=DEST.with_name(DEST.name+'_部署小文件.tar.gz')
    with tarfile.open(transfer,'x:gz') as tar:
        for path in sorted(DEST.rglob('*')):
            rel=path.relative_to(DEST).as_posix()
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc' and not rel.startswith(('模型/','上游框架/','训练数据/data/','训练数据/videos/','日志/')):
                tar.add(path,arcname=rel)
    print(json.dumps({'status':'prepared','root':str(DEST),'transfer':str(transfer),'train':len(split['train']),
                      'validation':len(split['validation']),'config':config,'training_started':False},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
