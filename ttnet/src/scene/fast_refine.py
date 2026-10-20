"""Bounded single-seed refinement: sampling, window EMA, then a gated AR trial."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from ttnet.common import PROJECT, now, save, sha

BASE = PROJECT/'out/experiments/20261016_table_ar'
BASELINE = BASE/'tcn_frames_lr0.0003_d0.1'
RULE_CACHE = PROJECT/'data/cache/features/813bf2bd95403b7c9e69'


def read(path):
    return json.loads(Path(path).read_text())


def build_features(config, output):
    from ttnet.events.data import EventCorpus
    from ttnet.events.continuous import attach_pp
    from ttnet.events.engine import EventSystem, build_cache, feature_cache_path
    torch.set_num_threads(4)
    cfg=read(config);corpus=EventCorpus(cfg['dataset']);attach_pp(corpus,cfg['pp_annotations'])
    system=EventSystem(cfg,torch.device('cuda'));system.configure('frozen',False)
    dest=feature_cache_path(system,corpus)
    build_cache(system,corpus,dest)
    save(output,dict(cache=str(dest)))


class FastRefinement:
    def __init__(self,root,started_at=None):
        self.root=root.resolve();self.root.mkdir(parents=True,exist_ok=True)
        settings=self.root/'study.json'
        if settings.exists():self.settings=read(settings)
        else:
            self.settings=dict(started_epoch=started_at or time.time(),budget_seconds=10800,
                 new_training_cutoff_seconds=9000,ar_cache_budget_seconds=2100,
                 baseline=str(BASELINE),rule_cache=str(RULE_CACHE),seed=42,
                 plan='20/20/60 PP; window EMA tau=.15; AR weight5/dropout5%/span3/beam1')
            save(settings,self.settings)
        self.rows=[];self.notes=[];self.partial=False
        self.base=read(BASELINE/'config.yaml')
        for key in ('dataset_id','split_id','train_samples_sha256','val_samples_sha256','head_initial_sha256','smoke'):
            self.base.pop(key,None)

    def elapsed(self):return time.time()-self.settings['started_epoch']

    def command(self,name,args,*,timeout=None,new_training=False):
        marker=self.root/'steps'/f'{name}.json'
        if marker.exists() and read(marker).get('exit_code')==0:return True
        if new_training and self.elapsed()>=self.settings['new_training_cutoff_seconds']:
            self.notes.append(f'{name}: skipped after 150-minute training cutoff');self.partial=True;return False
        remaining=self.settings['budget_seconds']-self.elapsed()
        if remaining<=0:
            self.notes.append(f'{name}: skipped at total budget limit');self.partial=True;return False
        timeout=min(timeout or remaining,remaining)
        log=self.root/'logs'/f'{name}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
        command=[sys.executable,'-u','-m',*map(str,args)]
        save(self.root/'status.json',dict(status='running',step=name,log=str(log),updated=now(),elapsed_seconds=self.elapsed()))
        print(f'START {name}',flush=True);started=time.time()
        timed_out=False
        with log.open('w') as handle:
            try:code=subprocess.run(command,cwd=PROJECT,stdout=handle,stderr=subprocess.STDOUT,timeout=timeout).returncode
            except subprocess.TimeoutExpired:code=124;timed_out=True
        save(marker,dict(command=command,log=str(log),exit_code=code,timed_out=timed_out,
                         elapsed_seconds=time.time()-started,finished=now()))
        print(f'DONE {name} exit={code}',flush=True)
        if timed_out:
            self.notes.append(f'{name}: timed out; partial artifacts preserved for resume');self.partial=True;return False
        if code:raise RuntimeError(f'{name}: exit {code}; see {log}')
        return True

    def add_row(self,name,run,metrics_path):
        result=read(metrics_path);m=result['metrics']
        row=dict(name=name,run=str(run),eval_id='continuous',metrics=str(metrics_path),
                 checkpoint=str(run/'checkpoints/best.pt'),checkpoint_sha256=result['checkpoint_sha256'],
                 seed=42,epoch=result['epoch'],macro_f1=m['macro_f1'],
                 tp=m['landing']['tp'],fp=m['landing']['fp'],fn=m['landing']['fn'],
                 threshold=m['threshold'],e2e_fps=result['e2e_fps'],by_view=m['by_view'],
                 protocol_id=result['protocol_id'],status='evaluated')
        self.rows.append(row);return row

    def winner(self,tracker=None):
        rows=[r for r in self.rows if r['e2e_fps']>=50 and
              (tracker is None or read(Path(r['run'])/'config.yaml').get('tracker_type')==tracker)]
        return max(rows,key=lambda r:(r['macro_f1'],-r['fp']))

    def event(self,name,config,cache):
        run=self.root/name;cfg=dict(config,scheme=name);path=self.root/'configs'/f'{name}.json'
        if path.exists() and read(path)!=cfg:raise ValueError(f'configuration changed: {name}')
        save(path,cfg)
        args=['ttnet.events.train','--config',path,'--dataset',cfg['dataset'],'--run',run,
              '--cache',cache,'--mode','frozen','--device','cuda']
        if run.exists():args+=['--resume','--restart-incomplete']
        if not self.command(name+'_train',args,new_training=True):return None
        dest=run/'evaluations/continuous'
        if dest.exists() and not (dest/'metrics.json').exists():
            dest.rename(dest.with_name('continuous_incomplete_'+str(time.time_ns())))
        if not self.command(name+'_eval',['ttnet.events.comparison','evaluate','--checkpoint',run/'checkpoints/best.pt',
             '--annotations',cfg['pp_annotations'],'--output',dest,'--device','cuda','--scheme',name,'--use-checkpoint-threshold']):return None
        result=self.add_row(name,run,dest/'metrics.json')
        state=read(run/'run.json');save(run/'run.json',dict(state,status='evaluated',evaluation=str(dest/'metrics.json')))
        return result

    def execute(self):
        snapshot=self.root/'source_snapshot'
        for area in ('events','tracking','scene','tests'):
            for source in (PROJECT/'ttnet/src'/area).glob('*.py'):
                dest=snapshot/area/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copyfile(source,dest)
        self.add_row('baseline_tcn',BASELINE,BASELINE/'evaluations/continuous/metrics.json')
        manifest=self.root/'hard_negatives.json'
        if not self.command('mine_hard_negatives',['ttnet.events.refinement','--checkpoint',BASELINE/'checkpoints/best.pt',
             '--cache',RULE_CACHE,'--output',manifest]):return
        sampled=dict(self.base,pp_positive_fraction=.2,pp_hard_negative_fraction=.2,
                     pp_hard_negative_manifest=str(manifest))
        self.event('tcn_sampled',sampled,RULE_CACHE)
        rule=self.winner('association');rule_cfg=read(Path(rule['run'])/'config.yaml')
        smooth=dict(rule_cfg,table_smoothing='window_ema',table_smoothing_tau_sec=.15)
        self.event('tcn_smoothed',smooth,RULE_CACHE)
        ar=self.root/'ar_null_fix'
        args=['ttnet.tracking.train_ar','--cache',BASE/'ar_training_cache','--run',ar,
              '--null-loss-weight',5,'--candidate-dropout-prob',.05,'--candidate-dropout-span',3,
              '--selection-metric','null_f1','--reference-checkpoint',BASE/'ar_table/best.pt']
        if ar.exists():args+=['--resume']
        if not self.command('ar_null_train',args,new_training=True):return
        gate=read(ar/'run.json');save(self.root/'ar_gate.json',gate)
        if not gate['end_to_end_gate_passed']:
            self.notes.append('AR stopped at fixed-cache gate; no trajectory cache or event-head training.')
            return
        if not self.command('ar_diagnosis',['ttnet.scene.diagnose_ar','--cache',BASE/'ar_training_cache',
             '--ar',ar/'best.pt','--output',ar/'diagnosis.json']):return
        # The AR tracker consumes raw predicted boxes. Event-window smoothing is independent.
        rule=self.winner('association');cfg=read(Path(rule['run'])/'config.yaml')
        cfg=dict(cfg,tracker_type='autoregressive',ar_checkpoint=str(ar/'best.pt'),ar_beam_size=1,ar_use_table=True)
        path=self.root/'configs/ar_event.json';save(path,cfg)
        descriptor=self.root/'ar_cache.json'
        if not self.command('ar_cache',['ttnet.scene.fast_refine','--build-cache-config',path,'--cache-output',descriptor],
                            timeout=self.settings['ar_cache_budget_seconds'],new_training=True):return
        self.event('tcn_ar_fixed',cfg,Path(read(descriptor)['cache']))

    def finalize(self,external=True):
        if not self.rows:return
        winner=self.winner();lock_path=self.root/'locked_selection.json'
        if lock_path.exists():
            previous=read(lock_path)
            if external and previous['winner']['checkpoint_sha256']!=winner['checkpoint_sha256']:
                raise ValueError('external selection lock changed')
            winner=previous['winner']
        selected=self.root/'selected';selected.mkdir(exist_ok=True)
        target=selected/'checkpoints/best.pt';target.parent.mkdir(exist_ok=True)
        source=Path(winner['checkpoint'])
        if target.exists() and sha(target)!=winner['checkpoint_sha256'] and not lock_path.exists():
            target.rename(target.with_name('previous_'+sha(target)[:16]+'.pt'))
        if not target.exists():shutil.copyfile(source,target)
        if sha(target)!=winner['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
        state=torch.load(target,map_location='cpu',weights_only=False)
        save(selected/'config.yaml',state['config'])
        selection=Path(winner['run'])/'best_selection.json'
        if selection.exists():shutil.copyfile(selection,selected/'best_selection.json')
        lock=dict(winner=winner,thresholds=state['thresholds'],locked=now(),
                  criterion='validation macro F1, fewer FP on tie, E2E FPS >=50',
                  note='Locked before external evaluation; no external-driven tuning.')
        if external and not lock_path.exists():save(lock_path,lock)
        if external:
            self.command('selected_external',['ttnet.scene.external','--run',selected])
        save(self.root/'results.json',self.rows)
        fields=['name','eval_id','seed','epoch','macro_f1','tp','fp','fn','threshold','e2e_fps',
                'checkpoint_sha256','checkpoint','metrics','status']
        with (self.root/'comparison.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(self.rows)
        lines=['# 轻量优化结果','',
               '固定 PP 114/40 视频、标签和 seed42，冻结定位器；单种子验证选型。', '',
               '| 方案 | 宏 F1 | TP/FP/FN | 完整视频 FPS |',
               '|---|---:|---:|---:|']
        for r in self.rows:lines.append(f"| {r['name']} | {r['macro_f1']:.2%} | {r['tp']}/{r['fp']}/{r['fn']} | {r['e2e_fps']:.2f} |")
        lines+=['',f"推荐：**{winner['name']}**；权重：[best.pt]({target})。",
                f"最佳验证错误叠加：[index.html]({Path(winner['metrics']).parent/'index.html'})。",
                '', 'FPS 包含原视频 PyAV 解码、定位、关联、事件头和 JSON 序列化，不含初始化及磁盘写入。',
                '13 帧事件窗需要未来6帧，定位器另需1帧；该吞吐不代表零前视在线延迟。']
        ext=selected/'evaluations/external/metrics.json'
        if ext.exists():
            m=read(ext)['metrics'];lines+=['',f"锁定阈值外测：TTGames anchor 分类宏 F1 {m['ttgames_test']['macro_f1']:.2%}；PP public 落点宏 F1 {m['pp_public']['macro_f1']:.2%}。",
             'TTGames 是稀疏 anchor 分类；PP public 是历史已查看的迁移集。']
        else:self.partial=True;lines+=['','外部评估尚未完成。']
        lines+=['','## 分支记录','']+[f'- {n}' for n in self.notes]
        ar=self.root/'ar_gate.json'
        if ar.exists():
            gate=read(ar);m=gate.get('selected_validation')
            if m:lines+=[f"- AR 最佳合格权重：正候选命中率 {m['positive_accuracy']:.2%}，null P/R/F1 {m['null_precision']:.2%}/{m['null_recall']:.2%}/{m['null_f1']:.2%}；门槛通过：{gate['end_to_end_gate_passed']}。"]
        text='\n'.join(lines)+'\n';(self.root/'summary.md').write_text(text)
        save(self.root/'status.json',dict(status='partial' if self.partial else 'completed',finished=now(),
             elapsed_seconds=self.elapsed(),winner=winner['name'],notes=self.notes,summary=str(self.root/'summary.md')))
        archive=PROJECT/'result'/self.root.name;archive.mkdir(parents=True,exist_ok=True)
        for name in ('comparison.csv','results.json','locked_selection.json','study.json','status.json'):
            if (self.root/name).exists():shutil.copyfile(self.root/name,archive/name)
        shutil.copyfile(selected/'config.yaml',archive/'config.yaml')
        if ext.exists():shutil.copyfile(ext,archive/'external_metrics.json')
        (archive/'README.md').write_text(text)
        print(text,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--started-at',type=float)
    p.add_argument('--build-cache-config',type=Path)
    p.add_argument('--cache-output',type=Path)
    args=p.parse_args()
    if args.build_cache_config:
        if not args.cache_output:p.error('--cache-output is required')
        build_features(args.build_cache_config,args.cache_output);return
    if not args.root:p.error('--root is required')
    study=FastRefinement(args.root,args.started_at)
    try:
        study.execute()
        study.finalize()
    except BaseException as exc:
        study.partial=True;study.notes.append(f'Execution interrupted: {exc}')
        study.finalize(external=False)
        raise


if __name__=='__main__':main()
