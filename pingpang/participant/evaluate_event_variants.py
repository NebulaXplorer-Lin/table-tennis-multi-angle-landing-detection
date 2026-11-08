"""Run one named entry through official full-video runner, then validate and score."""
from pathlib import Path
import sys,json,subprocess,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'pingpang')]
from participant.evaluate_tcn import compare,evaluate
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--entry',required=True);a.add_argument('--name',required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();data=Path('/workspace/pingpong/pingpang/public_data');dest=args.output/args.name
    assert not dest.exists(),dest
    cmd=[sys.executable,str(ROOT/'pingpang/run.py'),'--input',str(data),'--output',str(dest),'--solution',args.entry,'--decoder','gpu']
    with (args.output/(args.name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
    status=json.loads((dest/'run_status.json').read_text())
    assert status['totals']['ok']==15 and status['totals']['frames']==2109
    assert status['validation']['n_errors']==status['validation']['n_warnings']==0
    cache=status['solution_info']['global_cache_video_stats']
    assert len(cache)==15 and all(v['normalize_calls']==v['frames'] and v['resize_calls']==v['frames'] and v['max_entries']<=3 for v in cache)
    parity=compare(ROOT/'out/public-local-trt-001/local/predictions.jsonl',dest/'predictions.jsonl')
    metric=evaluate(data/'manifest.json',data/'gt_public.jsonl',dest/'predictions.jsonl')
    summary=dict(name=args.name,entry=args.entry,fps=status['performance']['infer_fps'],seconds=status['timing']['infer_sec'],metrics=metric,parity=parity)
    shadow=status['solution_info'].get('event_shadow')
    if shadow:
        summary['shadow']=dict(windows=len(shadow),max_logit_delta=max(r['logits_max_abs'] for r in shadow),max_probability_delta=max(r['probability_max_abs'] for r in shadow),threshold_flips=sum(r['threshold_flip'] for r in shadow))
    (args.output/(args.name+'-summary.json')).write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2),flush=True)
    assert parity['identical'],parity
    if shadow:assert len(shadow)==2109 and not summary['shadow']['threshold_flips']
