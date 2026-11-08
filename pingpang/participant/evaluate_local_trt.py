"""Sequential official-runner measurements and local public-set scoring."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'pingpang')]
from participant.evaluate_tcn import compare,evaluate,read_rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['baseline','local'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--input',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    entry={'baseline':'solution_public_trt_fast','local':'solution_public_trt_local'}[a.variant]
    dest=a.output/a.variant
    if dest.exists():raise FileExistsError(dest)
    env=dict(os.environ,TTNET_TRT_DIR=str(ROOT/'out/public-trt-001/engines'),TTNET_LOCAL_TRT_DIR=str((a.output/'engines').resolve()),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    cmd=[sys.executable,str(ROOT/'pingpang/run.py'),'--input',str(a.input),'--output',str(dest),'--solution',f'participant.{entry}:Solution','--decoder','gpu']
    with (a.output/(a.variant+'-run.log')).open('w') as log:
        subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
    status=json.loads((dest/'run_status.json').read_text(encoding='utf-8'))
    assert status['totals']['ok']==15 and status['totals']['frames']==2109,status['totals']
    assert status['validation']['n_errors']==status['validation']['n_warnings']==0
    metrics=evaluate(a.input/'manifest.json',a.input/'gt_public.jsonl',dest/'predictions.jsonl')
    reference=a.output/'baseline/predictions.jsonl' if a.variant!='baseline' else ROOT/'out/public-trt-001/fast/predictions.jsonl'
    parity=compare(reference,dest/'predictions.jsonl')
    old={(x['video_id'],x['frame_id']):(x['x'],x['y']) for x in read_rows(reference)}
    new={(x['video_id'],x['frame_id']):(x['x'],x['y']) for x in read_rows(dest/'predictions.jsonl')}
    changes=[dict(video_id=k[0],frame_id=k[1],prior=old[k],current=new[k]) for k in sorted(old.keys()&new.keys()) if old[k]!=new[k]]
    summary=dict(variant=a.variant,fps=status['performance']['infer_fps'],seconds=status['timing']['infer_sec'],
                 prepare_seconds=status['timing']['prepare_sec'],metrics=metrics,parity=parity,coordinate_changes=changes,command=cmd)
    (a.output/(a.variant+'-summary.json')).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(variant=a.variant,fps=summary['fps'],macro_f1=metrics['macro_f1'],total=metrics['total'],parity=parity,coordinate_changes=len(changes)),indent=2),flush=True)


if __name__=='__main__':main()
