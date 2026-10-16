"""Same-device four-way full-video runtime check before event training."""
import argparse
import json
from pathlib import Path

import cv2
import torch
from ttnet.common import DATA, PROJECT, jsonl, save, sha
from ttnet.configuration import load_config
from ttnet.events.engine import EventSystem
from ttnet.events.inference import predict_video
from ttnet.events.evaluate import head_benchmark


def benchmark(output, video_ids, device):
    dest=Path(output);dest.mkdir(parents=True,exist_ok=False)
    sources={r['video_id']:r for r in jsonl(DATA/'splits/val.jsonl')}
    if set(video_ids)-set(sources):raise ValueError('use frozen PP validation videos')
    results={}
    for scheme in 'ABCD':
        cfg,_=load_config(f'configs/experiments/event_comparison_{scheme}.json')
        torch.set_num_threads(cfg['torch_threads']);cv2.setNumThreads(1);torch.manual_seed(cfg['seed'])
        system=EventSystem(cfg,torch.device(device));system.configure('frozen',False)
        path=dest/scheme;path.mkdir()
        head=head_benchmark(system,path)
        measurements=[]
        for vid in video_ids:
            meta=json.loads((DATA/f'metadata/videos/{vid}.json').read_text())
            _,_,speed=predict_video(system,PROJECT/sources[vid]['video'],vid,target_fps=None)
            measurements.append(dict(speed,video_id=vid,resolution=meta['resolution']))
        fps=sum(r['frames'] for r in measurements)/sum(r['elapsed_sec'] for r in measurements)
        results[scheme]=dict(config=cfg,head=head,videos=measurements,e2e_fps=fps,meets_50_fps=fps>=50,
            global_sha256=sha(cfg['global_checkpoint']),local_sha256=sha(cfg['local_checkpoint']))
        save(dest/'progress.json',results)
        del system
        if device.startswith('cuda'):torch.cuda.empty_cache()
    result=dict(status='runtime_check_only',hardware=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU',
        note='Frozen trained locators, initialized event heads; no event accuracy or model selection claim. Includes complete video decode through serialization, excludes initialization and disk writes.',
        schemes=results)
    save(dest/'metrics.json',result)
    return {k:dict(fps=v['e2e_fps'],meets_50_fps=v['meets_50_fps']) for k,v in results.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--video',action='append',required=True)
    a=p.parse_args()
    try:print(json.dumps(benchmark(a.output,a.video,a.device),indent=2))
    except Exception as error:
        if a.output.exists():save(a.output/'failure.json',dict(status='failed',error=repr(error)))
        raise


if __name__=='__main__':main()
