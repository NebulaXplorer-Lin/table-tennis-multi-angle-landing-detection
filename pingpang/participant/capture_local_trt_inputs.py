"""Capture actual tracker ROI inputs from the current fast entry; not a speed baseline."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'pingpang')]
import torch
from participant.solution_public_trt_fast import Solution
from core.dataset import load_manifest
from core.logging_utils import RunLogger
from core.runner import Runner
from participant.evaluate_tcn import compare


class Capture(Solution):
    def prepare(self,context):
        super().prepare(context)
        self.samples=[];self.calls={}
        old=self.system.local_model.forward
        def forward(*args,**kwargs):
            index=self.calls.get(self.vid,0);self.calls[self.vid]=index+1
            fid=self.localized+1
            if index in (0,1,10,20) or fid==self.meta.n_frames-1:
                self.samples.append(dict(video_id=self.vid,frame_id=fid,call_index=index,
                                         inputs=tuple(x.detach().cpu().clone() for x in args)))
            return old(*args,**kwargs)
        self.system.local_model.forward=forward


def main():
    dest=Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=False)
    source=Path('/workspace/pingpong/pingpang/public_data')
    logger=RunLogger(dest/'capture');solution=Capture()
    try:
        status=Runner(solution,input_dir=source,output_dir=dest/'capture',logger=logger,decoder='gpu').run(load_manifest(source/'manifest.json'))
        logger.dump_status(dest/'capture')
        assert status['totals']['ok']==15
    finally:logger.close()
    parity=compare(ROOT/'out/public-trt-001/fast/predictions.jsonl',dest/'capture/predictions.jsonl')
    assert parity['identical'],parity
    torch.save(solution.samples,dest/'local_inputs.pt')
    summary=dict(samples=len(solution.samples),calls=solution.calls,parity=parity,
                 mask_patterns=sorted({tuple(s['inputs'][1][0].tolist()) for s in solution.samples}))
    (dest/'capture_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
