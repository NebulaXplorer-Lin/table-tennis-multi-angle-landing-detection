"""Resumable, evidence-preserving queue for table, AR and controlled event trials."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from ttnet.common import PROJECT,save,sha,now


class Study:
    def __init__(self,root):
        self.root=root.resolve();self.root.mkdir(exist_ok=True,parents=True)
        self.base=json.loads((PROJECT/'out/experiments/20261016-153213_event_comparison_b8_seed42_B/config.yaml').read_text())
        for key in ('dataset_id','split_id','train_samples_sha256','val_samples_sha256','head_initial_sha256','smoke'):self.base.pop(key,None)
        self.table=str(self.root/'table/best.pt');self.rows=[]

    def command(self,name,args):
        marker=self.root/'steps'/f'{name}.json'
        if marker.exists() and json.loads(marker.read_text()).get('exit_code')==0:return
        log=self.root/'logs'/f'{name}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
        save(self.root/'status.json',dict(status='running',step=name,log=str(log),updated=now(),pid=os.getpid()))
        print('START',name,flush=True)
        with log.open('w') as f:
            code=subprocess.call([sys.executable,'-u','-m',*map(str,args)],cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT)
        save(marker,dict(exit_code=code,command=args,log=str(log),finished=now()))
        if code:raise RuntimeError(f'{name}: exit {code}, see {log}')
        print('DONE',name,flush=True)

    def config(self,**kw):
        cfg=copy.deepcopy(self.base)
        cfg.update(context_head=True,scene_mode='relative',table_checkpoint=self.table,table_source='predicted',head_dropout=.1,time_encoding=False,event_window_mode='frames')
        cfg.update(kw);return cfg

    def event(self,name,cfg,evaluate=True):
        run=self.root/name;path=self.root/'configs'/f'{name}.json';cfg=dict(cfg,scheme=name)
        if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError(f'changed config for {name}')
        save(path,cfg)
        state=json.loads((run/'run.json').read_text()) if (run/'run.json').exists() else {}
        if state.get('status') not in ('trained_pending_eval','completed','evaluated'):
            args=['ttnet.events.train','--config',str(path),'--dataset',cfg['dataset'],'--run',str(run),'--mode','frozen','--device','cuda']
            if run.exists():args+=['--resume','--restart-incomplete']
            self.command(name+'_train',args)
        if evaluate:
            out=run/'evaluations/continuous'
            if out.exists() and not (out/'metrics.json').exists():out.rename(out.with_name('continuous_incomplete_'+str(time.time_ns())))
            self.command(name+'_eval',['ttnet.events.comparison','evaluate','--checkpoint',str(run/'checkpoints/best.pt'),'--annotations',cfg['pp_annotations'],'--output',str(out),'--device','cuda','--scheme',name,'--use-checkpoint-threshold'])
        selection=json.loads((run/'best_selection.json').read_text())
        result=dict(name=name,run=str(run),config=cfg,cached_f1=selection['continuous']['macro_f1'],epoch=selection['epoch'])
        if (run/'evaluations/continuous/metrics.json').exists():
            r=json.loads((run/'evaluations/continuous/metrics.json').read_text());result.update(f1=r['metrics']['macro_f1'],fps=r['e2e_fps'],fp=r['metrics']['landing']['fp'])
        self.rows=[r for r in self.rows if r['name']!=name]+[result];save(self.root/'results.json',self.rows)
        return result

    def prepare_oracle(self):
        # No extra detector computation: oracle remains an explicitly labeled diagnostic.
        cfg=self.config(table_source='oracle');path=self.root/'configs/oracle.json';save(path,cfg)
        self.command('oracle_cache',['ttnet.scene.cache','--config',str(path),'--base',str(PROJECT/'data/cache/features/35af6f8dbee933512d8c'),'--table-features',str(self.root/'table/features'),'--output',str(self.root/'oracle_cache.json')])

    def execute(self):
        if not (self.root/'table_cache.json').exists():raise ValueError('predicted table cache must be prepared first')
        self.prepare_oracle()
        stage=[]
        for mode in ('none','box','relative'):
            stage.append(self.event('scene_'+mode,self.config(scene_mode=mode)))
        self.event('scene_oracle',self.config(table_source='oracle'))
        winner=max(stage,key=lambda r:(r['f1'],-r['fp']))
        use_table=winner['config']['scene_mode']!='none'
        save(self.root/'scene_selection.json',dict(winner=winner['name'],use_table=use_table,criterion='full_video_val_macro_f1_then_fewer_fp'))
        cache=str(self.root/'ar_training_cache')
        # Both variants are trained, so negative table results remain interpretable.
        ar_candidates=[]
        for table in (True,False):
            name='ar_table' if table else 'ar_no_table';run=self.root/name
            args=['ttnet.tracking.train_ar','--cache',cache,'--run',str(run)]
            if not table:args+=['--no-table']
            if run.exists():args+=['--resume']
            if not (run/'run.json').exists() or json.loads((run/'run.json').read_text()).get('status')!='trained_pending_end_to_end_eval':
                self.command(name+'_train',args)
        # Prefer the predeclared scene winner for the full search-width comparison.
        arname='ar_table' if use_table else 'ar_no_table'
        for beam in (1,4,8):
            cfg=self.config(scene_mode=winner['config']['scene_mode'],tracker_type='autoregressive',ar_checkpoint=str(self.root/arname/'best.pt'),ar_beam_size=beam,ar_use_table=use_table)
            diag=['ttnet.scene.diagnostic','--checkpoint',str(Path(winner['run'])/'checkpoints/best.pt'),'--ar',cfg['ar_checkpoint'],'--beam',str(beam),'--output',str(self.root/f'diagnostic_ar{beam}')]
            if not use_table:diag+=['--no-table']
            self.command(f'ar_beam{beam}_fixed_head',diag)
            ar_candidates.append(self.event(f'ar_beam{beam}',cfg))
        # Rule association stays eligible if learned association does not help.
        track_winner=max([winner,*ar_candidates],key=lambda r:(r['f1'],-r['fp']))
        save(self.root/'tracker_selection.json',dict(winner=track_winner['name'],criterion='full_video_val_macro_f1_then_fewer_fp'))
        # Both architectures have the same 2 windows x 2 learning rates x 2 dropout budget.
        finalists=[]
        for kind in ('tcn','transformer'):
            trials=[]
            for window in ('frames','seconds'):
                for lr in (1e-4,3e-4):
                    for dropout in (.1,.2):
                        name=f'{kind}_{window}_lr{lr}_d{dropout}'
                        cfg=dict(track_winner['config'],event_head_type=kind,event_window_mode=window,lr=lr,head_dropout=dropout,time_encoding=True)
                        trials.append(self.event(name,cfg,evaluate=False))
            best=max(trials,key=lambda r:r['cached_f1'])
            best=self.event(best['name'],best['config'],evaluate=True)
            finalists.append(best)
        save(self.root/'locked_selection.json',dict(finalists=finalists,locked=now(),note='Locked before external evaluations; no external-driven tuning.'))
        multiseed=[]
        for final in finalists:
            multiseed.append(final)
            for seed in (43,44):multiseed.append(self.event(final['name']+f'_seed{seed}',dict(final['config'],seed=seed)))
        # Preserve original B architecture and training recipe for all three seeds.
        path=self.root/'configs/baseline_cache.json';save(path,self.base)
        self.command('baseline_cache',['ttnet.scene.cache','--config',str(path),'--base',str(PROJECT/'data/cache/features/35af6f8dbee933512d8c'),'--table-features',str(self.root/'table/features'),'--output',str(self.root/'baseline_cache.json')])
        for seed in (42,43,44):multiseed.append(self.event(f'baseline_B_seed{seed}',dict(self.base,seed=seed)))
        for final in finalists:
            self.command(final['name']+'_latency',['ttnet.scene.latency','--run',final['run']])
            self.command(final['name']+'_external',['ttnet.scene.external','--run',final['run']])
        save(self.root/'multiseed.json',multiseed)
        self.command('report',['ttnet.scene.study_report','--root',str(self.root)])
        save(self.root/'status.json',dict(status='completed',finished=now(),results=str(self.root/'results.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();study=Study(args.root)
    try:study.execute()
    except BaseException as error:
        state=json.loads((args.root/'status.json').read_text()) if (args.root/'status.json').exists() else {}
        save(args.root/'status.json',dict(state,status='failed',error=repr(error),updated=now()));raise

if __name__=='__main__':main()
