"""One explicit entry point for composed research experiments and data tools."""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import subprocess
import sys

from ttnet.common import PROJECT,RUNS,DATA,EVENT_DATA,identity,now,save,sha
from ttnet.configuration import load_config,project_path


def execution_plan(args):
    run=project_path(args.run) if args.run else None
    if args.resume and run is None:raise ValueError('--resume requires --run')
    if args.command!='train' and (args.config or args.smoke or args.stage):
        raise ValueError('evaluate/export/benchmark use saved run configuration; --config, --smoke and --stage are training options')
    if args.resume and args.command=='train' and args.config:
        raise ValueError('resume uses the original run/config.yaml; omit --config')
    config_path=args.config
    if args.resume and not config_path:config_path=run/'config.yaml'
    if args.command!='train' and run and not config_path:config_path=run/'config.yaml'
    config_path=config_path or 'configs/experiments/ball_fixed3.json'
    config,provenance=load_config(config_path,args.set)
    task=provenance.get('task') or ('event' if config.get('stage')=='event' or 'target_fps' in config or config.get('mode') in ('frozen','joint') else 'ball')
    if task=='event' and not args.resume:config.setdefault('mode','frozen')
    if args.command!='train' and args.set:raise ValueError('--set is supported for new training runs; evaluation uses its saved checkpoint configuration')
    if args.resume and args.set:raise ValueError('resume requires the original config without --set')
    if args.command!='train' and run is None:raise ValueError(f'{args.command} requires --run')
    name=provenance.get('name') or config.get('scheme') or ('event_tcn_'+config.get('mode','frozen') if task=='event' else 'ball_'+config.get('temporal_mode','single'))
    if run is None:
        stamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')
        run=RUNS/f'{stamp}_{name}'
    dataset=project_path(args.dataset) if args.dataset else Path(config.get('data' if task=='ball' else 'dataset',DATA if task=='ball' else EVENT_DATA))
    record=PROJECT/'experiments/records'/run.name
    effective=run/'config.yaml' if args.resume else record/'config.yaml'
    prefix=[sys.executable,'-u','-m'];commands=[]
    def add(module,*values):commands.append(prefix+[module,*map(str,values)])
    if args.command=='train':
        if run.exists() and not args.resume:raise FileExistsError(f'{run}; choose a new run or --resume')
        if task=='event':
            if args.stage not in (None,'all'):raise ValueError('--stage applies to ball localization only')
            add('ttnet.events.train','--mode',config['mode'],'--run',run,'--dataset',dataset,'--config',effective,'--device',args.device)
        else:
            stage=args.stage or (config.get('stage') if args.resume else 'all')
            if args.resume and stage!=config.get('stage'):raise ValueError('resume stage differs from original run')
            if stage=='all':
                if args.resume or args.smoke:raise ValueError('use an explicit global/local stage for --resume or --smoke')
                add('ttnet.pipeline','--dataset',dataset,'--config',effective,'--task',run,'--device',args.device)
            else:
                add('ttnet.training.train','--stage',stage,'--run',run,'--dataset',dataset,'--config',effective,'--device',args.device)
                checkpoint=args.global_checkpoint or config.get('global_checkpoint')
                if stage=='local':
                    if not checkpoint:raise ValueError('local stage requires --global-checkpoint')
                    commands[-1] += ['--global-checkpoint',str(checkpoint if args.resume else project_path(checkpoint))]
        if args.resume:commands[-1].append('--resume')
        if args.smoke:commands[-1].append('--smoke')
    elif task=='event':
        if args.dataset or args.global_checkpoint or args.local_checkpoint:
            raise ValueError('event evaluation uses the dataset and full system weights saved in the run')
        if args.command=='evaluate' and config.get('best_metric')=='pp_landing_macro_f1':
            if args.resume:raise ValueError('use a new evaluation output directory for continuous comparison')
            add('ttnet.events.comparison','evaluate','--checkpoint',run/'checkpoints/best.pt',
                '--annotations',config['pp_annotations'],'--output',run/'evaluations/continuous',
                '--scheme',config['scheme'].rsplit('_',1)[-1],'--device',args.device,'--use-checkpoint-threshold')
        elif args.command=='evaluate':
            add('ttnet.events.evaluate','--run',run,'--device',args.device)
            if args.resume:commands[-1].append('--resume')
        elif args.command=='benchmark':
            add('ttnet.cli','_event_benchmark','--run',run,'--device',args.device)
        elif args.command=='export':raise ValueError('EventHead ONNX export has not been implemented; locator export uses a ball run')
    else:
        if args.resume:raise ValueError('--resume evaluation is available for events; ball evaluations use --run directly')
        global_checkpoint=args.global_checkpoint or config.get('global_checkpoint') or run/'checkpoints/best.pt'
        local_checkpoint=args.local_checkpoint or (run/'checkpoints/best.pt' if config.get('stage')=='local' else None)
        add('ttnet.evaluation.'+args.command,'--run',run,'--dataset',dataset,'--global-checkpoint',project_path(global_checkpoint))
        if local_checkpoint:commands[-1] += ['--local-checkpoint',str(project_path(local_checkpoint))]
        if args.command!='export':
            commands[-1] += ['--device',args.device]
            if config.get('amp'):commands[-1].append('--amp')
    return dict(task=task,scheme=name,run=str(run),dataset=str(dataset),record=str(record),
                config=config,provenance=provenance,commands=commands)


def run_experiment(args):
    plan=execution_plan(args)
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if not Path(plan['dataset']).is_dir():raise FileNotFoundError(f'dataset not prepared: {plan["dataset"]}')
    record=Path(plan['record'])
    if args.command=='train' and not args.resume:
        cfg=dict(plan['config'],scheme=plan['scheme'])
        save(record/'config.yaml',cfg)
        provenance=dict(plan['provenance'],requested_config_hash=plan['provenance']['config_hash'],
                        effective_config_hash=identity(cfg),source_hashes={p:sha(p) for p in plan['provenance']['sources']})
        save(record/'provenance.json',provenance)
        save(record/'run.json',dict(status='launching',source=plan['run'],started=now(),commands=plan['commands']))
    try:
        launch_root=PROJECT/'out/launches'/Path(plan['run']).name
        attempt=launch_root/f'{len(list(launch_root.glob("attempt_*")))+1:03d}'
        attempt=attempt.with_name('attempt_'+attempt.name)
        attempt.mkdir(parents=True)
        with (attempt/'console.log').open('w') as stream:
            for command in plan['commands']:
                with subprocess.Popen(command,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
                    for line in child.stdout:
                        stream.write(line);stream.flush();sys.stdout.write(line);sys.stdout.flush()
                    code=child.wait()
                save(attempt/'exit.json',dict(command=command,exit_code=code,finished=now()))
                if code:raise subprocess.CalledProcessError(code,command)
    except BaseException as error:
        if args.command=='train':
            failure=dict(status='launcher_failed',source=plan['run'],error=repr(error),finished=now())
            save(record/'launch_failure.json',failure)
            if not (Path(plan['run'])/'run.json').exists():save(record/'run.json',failure)
        raise
    finally:
        if args.command=='train':
            run=Path(plan['run'])
            if (run/'run.json').exists():save(record/'run.json',json.loads((run/'run.json').read_text()))
            if (run/'config.yaml').exists() and not args.resume:
                actual=json.loads((run/'config.yaml').read_text())
                save(record/'config.yaml',actual)
                provenance_path=record/'provenance.json'
                if provenance_path.exists():
                    provenance=json.loads(provenance_path.read_text())
                    provenance['effective_config_hash']=identity(actual)
                    save(provenance_path,provenance)
        from ttnet.workspace import refresh_index
        refresh_index()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='list runnable experiment presets')
    for command in ('train','evaluate','benchmark','export'):
        p=sub.add_parser(command)
        p.add_argument('--config',type=Path);p.add_argument('--set',action='append',default=[])
        p.add_argument('--run',type=Path);p.add_argument('--dataset',type=Path)
        p.add_argument('--global-checkpoint',type=Path);p.add_argument('--local-checkpoint',type=Path)
        p.add_argument('--stage',choices=('global','local','all'));p.add_argument('--device',default='cuda')
        p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('data');p.add_argument('action',choices=('plan','migrate','verify','preview'))
    p.add_argument('--workers',type=int,default=2);p.add_argument('--dataset',type=Path)
    p.add_argument('--video');p.add_argument('--frame',type=int);p.add_argument('--output',type=Path)
    p.add_argument('--full',action='store_true')
    p=sub.add_parser('results');p.add_argument('--run',type=Path)
    p=sub.add_parser('_event_benchmark');p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda')
    args=parser.parse_args()
    try:
        if args.command=='list':
            for path in sorted((PROJECT/'configs/experiments').glob('*.json')):
                config,meta=load_config(path)
                print(f'{meta["name"]}: task={meta["task"]}, dataset={meta["dataset"]["name"]}, config={path.relative_to(PROJECT)}')
        elif args.command=='data':
            from ttnet.data_tools.workspace_migrate import plan,migrate
            if args.action in ('plan','migrate'):
                print(json.dumps(plan() if args.action=='plan' else migrate(args.workers),indent=2))
            elif args.action=='verify':
                from ttnet.data_tools.storage_check import verify_store
                if args.dataset is None:raise ValueError('--dataset is required')
                print(json.dumps(verify_store(project_path(args.dataset),full=args.full),indent=2))
            else:
                import cv2
                from ttnet.data_tools.frame_store import FrameStore
                if args.dataset is None or args.video is None or args.frame is None or args.output is None:
                    raise ValueError('preview requires --dataset --video --frame --output')
                rgb=FrameStore(project_path(args.dataset)).get_rgb(args.video,args.frame)
                output=project_path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(output),rgb[:,:,::-1]):raise OSError(output)
                print(output)
        elif args.command=='results':
            if args.run:
                run=project_path(args.run);config=json.loads((run/'config.yaml').read_text())
                if config.get('stage')=='event':
                    from ttnet.events.report import publish
                    publish(run)
                else:
                    from ttnet.publication import publish_run
                    publish_run(run)
            from ttnet.workspace import refresh_index
            print(refresh_index())
        elif args.command=='_event_benchmark':
            import torch
            from ttnet.events.engine import EventSystem
            from ttnet.events.evaluate import head_benchmark
            state=torch.load(args.run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            system=EventSystem(state['config'],torch.device(args.device),initialize=False)
            system.load_state_dict(state['system']);system.configure(state['config']['mode'],False)
            dest=args.run/'evaluations/benchmark';dest.mkdir(parents=True,exist_ok=True)
            print(json.dumps(head_benchmark(system,dest),indent=2))
        else:run_experiment(args)
    except (ValueError,FileNotFoundError,FileExistsError) as error:parser.error(str(error))


if __name__=='__main__':main()
