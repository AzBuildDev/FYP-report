"""标记前/完整轨迹各半、各自RC/BC均衡；不截断未来动作，不生成反指令标签。"""
from pathlib import Path
import json
import numpy as np
from 目标选择标注数据 import DatasetCatalog, derive_intervals, PROTOCOL

SAMPLING_VERSION = 'stage_balanced_v1'

def load_annotations(dataset_root, annotation_path, split, chunk=50):
    catalog = DatasetCatalog(dataset_root)
    data = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    if (data.get('schema_version') != 2 or data.get('protocol') != PROTOCOL
        or data.get('kind') != 'reviewed_first_contact_stages'
        or data.get('episode_index_base') != 0 or data.get('frame_index_base') != 0
        or data.get('interval_endpoints') != 'inclusive' or data.get('chunk_size') != chunk
        or data.get('fps') != catalog.fps or data.get('dataset_fingerprint') != catalog.fingerprint):
        raise ValueError('标注协议、段号基准、动作长度或数据指纹不匹配')
    rows = data.get('episodes', [])
    if data.get('count') != 200 or len(rows) != 200 or len(catalog.ids) != 200:
        raise ValueError('本实验要求完整的200段确认标注')
    annotations = {}
    for row in rows:
        ep = row['episode_index']
        if type(ep) is not int or ep in annotations or ep not in catalog.episodes:
            raise ValueError('标注段号重复或不存在（必须从0开始）')
        meta = catalog.episodes[ep]
        if (row.get('status') != 'reviewed' or row.get('scene_confirmed') is not True
            or any(row.get(k) != meta[k] for k in ('task','target','length'))):
            raise ValueError(f'段{ep}未确认或颜色/长度与Parquet不一致')
        intervals = derive_intervals(row['markers'], meta['length'], chunk)
        if intervals != row.get('intervals') or not intervals['pre_contact_window_starts_inclusive']:
            raise ValueError(f'段{ep}标记无效或没有足够的标记前动作窗口')
        if row.get('video_sha256') != catalog.video_hashes(ep):
            raise ValueError(f'段{ep}的视频已变化，拒绝使用旧标注')
        annotations[ep] = row
    if set(annotations) != set(catalog.ids):
        raise ValueError('标注缺段')
    train, val = split['train'], split['validation']
    if (len(train) != 160 or len(set(train)) != 160 or len(val) != 40 or len(set(val)) != 40
        or set(train) & set(val) or set(train) | set(val) != set(catalog.ids)):
        raise ValueError('划分不是互斥完整的160/40段')
    for ids, count in ((train,80),(val,20)):
        if any(sum(annotations[ep]['target']==target for ep in ids)!=count for target in ('RC','BC')):
            raise ValueError('划分中的RC/BC不均衡')
    return annotations, {'sampling_version':SAMPLING_VERSION,'dataset_fingerprint':catalog.fingerprint,
                         'episodes':200,'train_episodes':160,'validation_episodes':40,
                         'train_target_counts':{'RC':80,'BC':80},'validation_target_counts':{'RC':20,'BC':20},
                         'stage_fraction':{'pre_marker':0.5,'full_trajectory':0.5},
                         'chunk_size':chunk,'action_window_truncated':False,
                         'negative_instruction_supervision':False,'rgb_color_preserved':True}

class StageSampler:
    def __init__(self, tables, annotations, batch_size, seed=42, chunk=50):
        if type(batch_size) is not int or batch_size < 4 or batch_size % 4:
            raise ValueError('batch必须能被4整除，以严格保证四个采样分层均衡')
        self.batch_size,self.seed,self.chunk=batch_size,seed,chunk
        self.tables={int(t['episode_index'][0]):t for t in tables}
        if len(self.tables)!=len(tables):
            raise ValueError('训练段号重复')
        self.annotations=annotations
        self.groups={task:[] for task in (0,1)}
        for ep,table in self.tables.items():
            row=annotations[ep];task=int(table['task_index'][0])
            if set(table['task_index'])!={task} or row['target'] != ('RC' if task==0 else 'BC'):
                raise ValueError('训练标签不是整段一致的RC/BC')
            if len(table['action'])!=row['length'] or set(table['episode_index'])!={ep}:
                raise ValueError('动作长度/段号与标注不符')
            self.groups[task].append(ep)
        if not all(self.groups.values()):
            raise ValueError('缺少一种颜色的训练数据')

    def plan(self,step):
        if type(step) is not int or step < 1:
            raise ValueError('更新步必须从1开始')
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step]))
        records=[]
        for stage in ('pre_marker','full_trajectory'):
            for task in (0,1):
                group=self.groups[task]
                for _ in range(self.batch_size//4):
                    ep=group[int(rng.integers(len(group)))];row=self.annotations[ep]
                    maximum=row['length']-self.chunk
                    if stage=='pre_marker':
                        maximum=min(maximum,row['markers']['first_contact']-1)
                    if maximum<0:
                        raise ValueError('采样范围为空，禁止静默降级')
                    frame=int(rng.integers(maximum+1))
                    records.append({'episode':ep,'frame':frame,'task_index':task,'stage':stage})
        return [records[int(i)] for i in rng.permutation(self.batch_size)]

    def records(self,step):
        return [(self.tables[r['episode']],r['frame']) for r in self.plan(step)]
