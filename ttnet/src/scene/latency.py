"""Synchronized per-frame processing latency, separate from unsynchronized E2E FPS."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from ttnet.common import DATA,save
from ttnet.events.engine import EventSystem
from ttnet.events.data import assemble_window
from ttnet.evaluation.ball_corpus import BallCorpus


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    state=torch.load(a.run/'checkpoints/best.pt',map_location='cpu',weights_only=False);cfg=state['config'];system=EventSystem(cfg,torch.device('cuda'),initialize=False);system.load_state_dict(state['system']);system.configure('frozen',False)
    corpus=BallCorpus(DATA,'val');vid='22_001';n=min(250,len(corpus.videos[vid]['frames']));system.reset_tracking(vid);features={};records={};samples=[]
    with torch.inference_mode(),torch.autocast('cuda',enabled=cfg['amp']):
        for fid in range(n):
            torch.cuda.synchronize();start=time.perf_counter();d,r,_=system.forward_frames(corpus,vid,[fid]);features.update(d);records.update(r)
            # Delayed event output needs up to 15 future native frames at 60 FPS.
            center=fid-15
            if center>=15:
                system.head(*assemble_window(corpus,vid,center,features,records,system.device,cfg))
            torch.cuda.synchronize()
            if fid>=50:samples.append((time.perf_counter()-start)*1000)
    save(a.run/'evaluations/latency.json',dict(samples_ms=samples,p50_ms=float(np.percentile(samples,50)),p95_ms=float(np.percentile(samples,95)),warmup_frames=50,video=vid,protocol='synchronized locator/table/association + one delayed event window; NPY frame reads included; separate from raw-video E2E FPS; no claim of zero lookahead'))

if __name__=='__main__':main()
