"""Final locked-threshold public and TTGames test evaluation (no retuning)."""
import argparse,json
from pathlib import Path
import torch
import cv2
from ttnet.common import PROJECT,save,sha,now
from ttnet.events.data import EventCorpus
from ttnet.events.engine import EventSystem
from ttnet.events.evaluate import sparse_evaluation,public_evaluation,head_benchmark


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    state=torch.load(a.run/'checkpoints/best.pt',map_location='cpu',weights_only=False);cfg=state['config']
    system=EventSystem(cfg,torch.device('cuda'),initialize=False);system.load_state_dict(state['system']);system.configure('frozen',False)
    dest=a.run/'evaluations/external';dest.mkdir(exist_ok=True,parents=True)
    ident=dict(checkpoint_sha256=sha(a.run/'checkpoints/best.pt'),thresholds=state['thresholds'],locked=True)
    if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=ident:raise ValueError('external identity changed')
    save(dest/'identity.json',ident);metrics={}
    corpus=EventCorpus(cfg['dataset'])
    for name,fn in [('ttgames_test',lambda path:sparse_evaluation(system,corpus,'test',path,state['thresholds'])),('pp_public',lambda path:public_evaluation(system,PROJECT/'data/pp_public_data',path,state['thresholds']))]:
        completed=dest/f'{name}_result.json';path=dest/name
        if completed.exists():metrics[name]=json.loads(completed.read_text());continue
        if path.exists():path.rename(path.with_name(name+'_incomplete_'+str(__import__('time').time_ns())))
        result=fn(path);save(completed,result);metrics[name]=result
    metrics['head_benchmark']=head_benchmark(system,dest)
    save(dest/'metrics.json',dict(metrics=metrics,identity=ident,finished=now()))

if __name__=='__main__':main()
