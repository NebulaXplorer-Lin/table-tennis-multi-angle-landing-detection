"""Capture real gated event windows; original head and predictions unchanged."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'pingpang')]
import torch
from participant.solution_public_trt_local import Solution
from participant.evaluate_tcn import compare
from core.dataset import load_manifest
from core.logging_utils import RunLogger
from core.runner import Runner

class Capture(Solution):
    def prepare(self,ctx):
        super().prepare(ctx);self.samples=[];self.counts={}
        old=self.system.head.forward
        def forward(features,motion,valid,table):
            args=(features,motion,valid,*(table[k] for k in ('roi','context','roi_valid','context_valid')))
            result=old(features,motion,valid,table)
            p=float(result.float().sigmoid()[0,0]);fid=self.next_center
            if fid<7 or fid>=self.meta.n_frames-7 or fid%25==0 or .1<=p<=.6:
                self.samples.append(dict(video_id=self.vid,frame_id=fid,score=p,inputs=tuple(x.detach().cpu().clone() for x in args)))
                self.counts[self.vid]=self.counts.get(self.vid,0)+1
            return result
        self.system.head.forward=forward

if __name__=='__main__':
    dest=Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=False)
    data=Path('/workspace/pingpong/pingpang/public_data');solution=Capture();logger=RunLogger(dest/'capture')
    try:
        status=Runner(solution,input_dir=data,output_dir=dest/'capture',logger=logger,decoder='gpu').run(load_manifest(data/'manifest.json'))
        logger.dump_status(dest/'capture');assert status['totals']['ok']==15
    finally:logger.close()
    parity=compare(ROOT/'out/public-local-trt-001/local/predictions.jsonl',dest/'capture/predictions.jsonl');assert parity['identical'],parity
    torch.save(solution.samples,dest/'event_inputs.pt')
    summary=dict(samples=len(solution.samples),per_video=solution.counts,parity=parity)
    (dest/'capture_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
