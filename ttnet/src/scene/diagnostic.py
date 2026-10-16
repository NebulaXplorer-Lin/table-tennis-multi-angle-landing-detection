"""Change only the frozen tracker, keeping a trained event head unchanged."""
import argparse,json
from pathlib import Path
import torch
from ttnet.common import save
from ttnet.events.engine import EventSystem
from ttnet.events.comparison import evaluate_pp


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--ar',type=Path,required=True);p.add_argument('--beam',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--no-table',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=dict(state['config'],tracker_type='autoregressive',ar_checkpoint=str(a.ar),ar_beam_size=a.beam,ar_use_table=not a.no_table)
    ar=torch.load(a.ar,map_location='cpu',weights_only=False)
    weights=dict(state['system']);weights.update({'ar_model.'+k:v for k,v in ar['model'].items()})
    modified=dict(state,config=cfg,system=weights);a.output.mkdir(exist_ok=True,parents=True);torch.save(modified,a.output/'diagnostic.pt')
    out=a.output/'continuous'
    if not (out/'metrics.json').exists():
        if out.exists():out.rename(out.with_name('continuous_incomplete_'+str(__import__('time').time_ns())))
        evaluate_pp(a.output/'diagnostic.pt',cfg['pp_annotations'],out,'cuda','frozen_head_ar'+str(a.beam),state['thresholds'])

if __name__=='__main__':main()
