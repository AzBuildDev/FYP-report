"""English-first local ASR -> finite LLM task -> validated preview -> explicit dry run."""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from 语音任务协议 import compile_request, fixture_scene, DryRunExecutor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--text')
    source.add_argument('--audio',type=Path,help='Existing mono 16kHz PCM16 WAV, <=20 seconds; never records')
    p.add_argument('--language',choices=['en','zh'],default='en')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--model',choices=['Qwen2.5-0.5B-Instruct','Qwen2.5-1.5B-Instruct','Qwen3-1.7B'],default='Qwen2.5-1.5B-Instruct')
    p.add_argument('--scene',type=Path,help='Read-only scene JSON. Without scene only a rejected preview is produced.')
    p.add_argument('--fixture-scene',action='store_true',help='Explicit synthetic fixture, never a camera observation')
    p.add_argument('--default-target',choices=['zone_A','zone_B'])
    p.add_argument('--execute-dry-run',action='store_true',help='Explicitly call DryRunExecutor after validation')
    p.add_argument('--timeout',type=float,default=90,help='Hard subprocess deadline including model loading/ASR')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--include-scene',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--policy-profile',choices=['legacy_red','red_blue_30k'],default='legacy_red',
                   help='Explicit semantic mapping; red_blue_30k remains dry-run, pending hardware validation')
    args=p.parse_args()
    if not 0 < args.timeout <= 300:p.error('--timeout must be in (0,300]')
    if args.scene and args.fixture_scene:p.error('choose scene OR fixture-scene')
    if not args.worker:
        # The child belongs to this invocation. Timeout never touches other user processes.
        worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
        try:
            stdout,stderr=worker.communicate(timeout=args.timeout)
            if stdout:print(stdout,end='')
            if worker.returncode and not stdout:
                print(json.dumps({'status':'rejected','reason':'worker_failed','dry_run':True,'error':stderr[-2000:]},ensure_ascii=False))
            return worker.returncode
        except subprocess.TimeoutExpired:
            if os.name=='nt':
                # Windows venv redirectors spawn a base interpreter: terminate the OWNED tree.
                subprocess.run(['taskkill','/PID',str(worker.pid),'/T','/F'],capture_output=True)
            else:worker.kill()
            worker.communicate(timeout=10)
            print(json.dumps({'status':'rejected','reason':'pipeline_timeout','dry_run':True,'executable':False}))
            return 2
    try:
        from 本地语音模型 import ROOT, LocalASR, LocalClassifier
        scene=fixture_scene() if args.fixture_scene else json.loads(args.scene.read_text(encoding='utf-8')) if args.scene else None
        metrics={}
        text=args.text
        if args.audio:
            asr=LocalASR(ROOT/'模型/语音指令/whisper-tiny',args.device)
            text,metrics=asr.transcribe(args.audio,args.language)
            del asr
        model_path=ROOT/'模型/语音指令'/args.model
        llm=LocalClassifier(model_path,args.device)
        raw,llm_metrics=llm.classify(text,args.default_target)
        # Synthetic fixture timestamps are refreshed after model loading, clearly labelled.
        if args.fixture_scene:scene=fixture_scene()
        result=compile_request(text,raw,scene,args.default_target,policy_profile=args.policy_profile)
        result.update(raw_model_output=raw,metrics={**metrics,**llm_metrics},model=llm.metadata,
                      scene_source=scene.get('source','external_json_unverified') if scene else None)
        if args.include_scene:result['scene_snapshot']=scene
        if args.execute_dry_run and result['status']=='ready':
            result['execution']=DryRunExecutor(args.policy_profile).execute(result,scene)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['status']=='ready' else 2
    except Exception as e:
        print(json.dumps({'status':'rejected','reason':type(e).__name__,'detail':str(e),'dry_run':True,'executable':False},ensure_ascii=False))
        return 2

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
