"""从已原子保存的last保留每2000步完整检查点；不修改训练器、采样、优化器或学习率。"""
from pathlib import Path
import argparse,json,hashlib,os,time,fcntl,shutil,sys
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'输出/新200段_阶段采样'
FILES=('config.json','model.safetensors','dataset_stats.json','training_provenance.json','checkpoint_summary.json','training_state.pt')
MANIFESTS=OUTPUT/'下载清单'

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def snapshot(source,target,expected_step):
    if target.exists():
        summary=json.loads((target/'checkpoint_summary.json').read_text())
        assert summary['step']==expected_step
    else:
        staging=target.with_name(target.name+'.archiving')
        if staging.exists():shutil.rmtree(staging)
        staging.mkdir(parents=True)
        fd=os.open(source,os.O_RDONLY|os.O_DIRECTORY)
        try:
            for name in FILES:os.link(name,str(staging/name),src_dir_fd=fd)
            summary=json.loads((staging/'checkpoint_summary.json').read_text())
            assert summary['step']==expected_step,'last changed before capture'
            provenance=json.loads((staging/'training_provenance.json').read_text())
            assert provenance['steps']==expected_step
            assert sha256(staging/'model.safetensors')==provenance['checkpoint_sha256']
            staging.rename(target)
        finally:
            os.close(fd)
            if staging.exists():shutil.rmtree(staging)
    entries=[{'name':n,'bytes':(target/n).stat().st_size,'sha256':sha256(target/n)} for n in FILES]
    result={'step':expected_step,'directory':str(target),'files':entries,'completed_at':time.time(),
            'includes_optimizer_scheduler_rng':True}
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        source=OUTPUT/'last';summary=json.loads((source/'checkpoint_summary.json').read_text())
        target=ROOT/'验收输出/归档复制自检'
        result=snapshot(source,target,summary['step'])
        assert len(result['files'])==6 and all(sha256(target/e['name'])==e['sha256'] for e in result['files'])
        shutil.rmtree(target)
        (ROOT/'整2K归档自检.json').write_text(json.dumps({'status':'passed','step':summary['step'],'files':6}),encoding='utf-8')
        print('ARCHIVE_SELF_TEST_PASSED',summary['step'],flush=True)
        return
    MANIFESTS.mkdir(exist_ok=True)
    sys.stdout=(ROOT/'日志/整2K归档.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=sys.stdout
    lock=(ROOT/'日志/整2K归档.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    planned=json.loads((ROOT/'训练参数.json').read_text())['steps']
    while True:
        source=OUTPUT/'last'
        try:
            summary=json.loads((source/'checkpoint_summary.json').read_text())
            step=summary['step']
            if step==planned:
                status=json.loads((OUTPUT/'status.json').read_text())
                if status.get('state')!='completed':
                    time.sleep(10)
                    continue
            manifest=MANIFESTS/f'step_{step:06d}.json'
            if (step%2000==0 or step==planned) and not manifest.exists():
                result=snapshot(source,OUTPUT/f'step_{step:06d}',step)
                atomic(manifest,result)
                print('ARCHIVED',step,flush=True)
            atomic(ROOT/'整2K归档状态.json',{'state':'watching','last_complete_step':step,'interval':2000,'checked_at':time.time()})
            if step==planned:
                best=OUTPUT/'best_val'
                best_step=json.loads((best/'checkpoint_summary.json').read_text())['step']
                best_target=OUTPUT/f'best_val_final_{best_step:06d}'
                result=snapshot(best,best_target,best_step)
                result['kind']='best_val_final'
                atomic(MANIFESTS/f'best_val_final_{best_step:06d}.json',result)
                print('FINAL_CHECKPOINT_ARCHIVED',step,flush=True)
                break
        except (FileNotFoundError,AssertionError,OSError) as exc:
            print('ARCHIVE_RETRY',type(exc).__name__,str(exc),flush=True)
        time.sleep(10)
if __name__=='__main__':main()
