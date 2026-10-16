"""Compare augmented cached features with actual chronological native-frame inference."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ttnet.common import DATA,save,jsonl
from ttnet.events.engine import EventSystem
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.scene.table import TableHead,decode_table
from ttnet.data_tools.geometry import letterbox_transform


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    cfg=json.loads((a.root/'configs/table_relative.json').read_text());system=EventSystem(cfg,torch.device('cuda'));system.configure('frozen',False)
    corpus=BallCorpus(DATA,'val');evidence=[]
    for vid in ('02_003','03_001','22_001'):
        if vid not in corpus.videos:continue
        system.reset_tracking(vid)
        n=min(20,len(corpus.videos[vid]['frames']))
        with torch.inference_mode(),torch.autocast('cuda'):
            d,r,_=system.forward_frames(corpus,vid,range(n))
        base={x['frame_id']:x for x in jsonl(Path('data/cache/features/35af6f8dbee933512d8c')/f'pp_{vid}.jsonl')}
        new={x['frame_id']:x for x in jsonl(a.root/f'ar_training_cache/{vid}.jsonl')}
        coordinate=[];box=[]
        for fid in range(n):
            p,q=r[fid]['prediction'],base[fid]['prediction']
            if bool(p)!=bool(q) or (p and p['valid']!=q['valid']):raise AssertionError('observation validity changed')
            if p:coordinate.append(float(np.linalg.norm(np.array([p['x'],p['y']])-[q['x'],q['y']])))
            w,h=corpus.videos[vid]['meta']['width'],corpus.videos[vid]['meta']['height']
            box.append(float(np.max(abs(np.array(r[fid]['table']['bbox_xyxy'])-new[fid]['table']['bbox_xyxy'])/[w,h,w,h])))
        evidence.append(dict(video=vid,frames=n,max_ball_delta_px=max(coordinate,default=0.),max_table_delta_normalized=max(box)))
    if not evidence:raise AssertionError('no parity coverage')
    if max(r['max_ball_delta_px'] for r in evidence)>1.:raise AssertionError(evidence)
    if max(r['max_table_delta_normalized'] for r in evidence)>.005:raise AssertionError(evidence)
    save(a.root/'cache_parity.json',dict(status='passed',evidence=evidence,table_tolerance_normalized=.005,ball_tolerance_px=1.,note='AMP batch size variation; frozen rule predictions checked against pre-change cache'))
    print(evidence)

if __name__=='__main__':main()
