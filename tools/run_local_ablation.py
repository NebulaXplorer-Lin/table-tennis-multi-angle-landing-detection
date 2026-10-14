"""Run independent gap experiments through the unmodified official runner."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from evaluate_landings import evaluate

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_path(path: Path) -> Path:
    """Return an absolute path with at most one Windows long-path prefix."""
    value = str(path.resolve())
    if os.name != 'nt' or value.startswith('\\\\?\\'):
        return Path(value)
    if value.startswith('\\\\'):
        return Path('\\\\?\\UNC\\'+value[2:])
    return Path('\\\\?\\'+value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gaps', type=int, nargs='+', default=[10,8,6])
    a = ap.parse_args()
    if any(g < 1 for g in a.gaps) or len(set(a.gaps)) != len(a.gaps):
        ap.error('gaps must be distinct positive integers')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    group = stamp+'_ball-onnx_min-gap'
    data = runtime_path(ROOT/'pingpang/public_data')
    model = runtime_path(ROOT/'pingpang/participant/weights/ball.onnx')
    if not model.is_file() or not (data/'manifest.json').is_file():
        raise FileNotFoundError('Missing ball.onnx or public_data/manifest.json; get these from the official package.')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    missing = [v['video_id'] for v in manifest['videos'] if not (data/v['path']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing videos: {missing}')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable,str(ROOT/'pingpang/integrity.py'),'--verify'],env=env,check=True)
    metadata: dict[str, str | bool] = dict(model_sha256=sha(model),manifest_sha256=sha(data/'manifest.json'),
        gt_sha256=sha(data/'gt_public.jsonl'),python=sys.version,platform=platform.platform(),
        pyproject_sha256=sha(ROOT/'pyproject.toml'),lock_sha256=sha(ROOT/'uv.lock'),
        solution_sha256=sha(ROOT/'pingpang/participant/solution.py'),
        local_adapter_sha256=sha(ROOT/'pingpang/participant/solution_local.py'))
    commit = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    dirty = subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True)
    metadata.update(commit=commit.stdout.strip(),working_tree_dirty=bool(dirty.stdout.strip()))
    comparison = []
    for gap in a.gaps:
        exp_id = group+f'-gap{gap}'
        record = runtime_path(ROOT/'experiments/records'/exp_id)
        output = runtime_path(ROOT/'out/experiments'/exp_id/'eval/public')
        record.mkdir(parents=True,exist_ok=False)
        output.mkdir(parents=True,exist_ok=False)
        config = dict(min_gap=gap,backend='onnxruntime_cuda_fp32',decoder='cpu',input_shape=[1,9,288,512],
                      data=str(data),output=str(output),public_validation_used_for_parameter_selection=True)
        (record/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        command=[sys.executable,str(ROOT/'pingpang/run.py'),'--input',str(data),
            '--output',str(output),'--solution','participant.solution_local:Solution','--decoder','cpu']
        run: dict[str, object] = dict(experiment_id=exp_id,state='running',started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 command=command,**metadata)
        run_file=record/'run.json'
        run_file.write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
        try:
            print(f'\n=== MIN_GAP={gap} ===',flush=True)
            with (output/'console.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,LOCAL_MIN_GAP=str(gap)),
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                assert proc.stdout is not None, 'Runner stdout pipe was not created'
                for line in proc.stdout:
                    print(line,end='',flush=True)
                    log.write(line)
                    log.flush()
                rc=proc.wait()
            run['exit_code']=rc
            if rc != 0:
                raise RuntimeError(f'Runner exited with code {rc}; see {output / "console.log"}')
            status=json.loads((output/'run_status.json').read_text(encoding='utf-8'))
            run['runtime']=status.get('solution_info',{})
            assert rc==0 and status['status']=='ok' and status['totals']['ok']==len(manifest['videos']), 'Incomplete run'
            score=evaluate(output/'predictions.jsonl',data)
            score['performance']=status.get('performance',{})
            score['prediction_sha256']=sha(output/'predictions.jsonl')
            (record/'evaluation.json').write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding='utf-8')
            row=dict(experiment_id=exp_id,min_gap=gap,state='completed',**score['overall'],macro_f1=score['macro_f1'],
                     f1_0=score['by_angle']['0度']['f1'],f1_45=score['by_angle']['45度']['f1'],f1_90=score['by_angle']['90度']['f1'],
                     local_fps=status['performance']['infer_fps'])
            comparison.append(row)
            run['state']='completed'
            (record/'summary.md').write_text(f"# MIN_GAP={gap}\n\nSame local model/backend/decoder; only MIN_GAP varies.\n\nLocal estimated macro F1: {score['macro_f1']:.4f}.\n\nNot an official competition score. Public validation is used for parameter selection.\n",encoding='utf-8')
        except Exception as exc:
            run.update(state='failed',error=str(exc))
            raise
        finally:
            run['finished_at']=datetime.now(timezone(timedelta(hours=8))).isoformat()
            run_file.write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    compare_dir=ROOT/'experiments'
    csv_path=compare_dir/(group+'_comparison.csv')
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    lines=['# Local MIN_GAP comparison','','Only compare rows from this local group. Scores are approximate; FPS is not competition FPS.','',
           '| MIN_GAP | Precision | Recall | F1 overall | F1 0 | F1 45 | F1 90 | F1 mean |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in comparison:
        lines.append('| '+str(x['min_gap'])+' | '+' | '.join(f"{x[k]:.4f}" for k in ['precision','recall','f1','f1_0','f1_45','f1_90','macro_f1'])+' |')
    report=compare_dir/(group+'_comparison.md')
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\nCOMPLETED:',report)


if __name__=='__main__':
    main()
