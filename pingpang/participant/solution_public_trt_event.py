"""Four-network TensorRT entry; unchanged streaming and event decision logic."""
from pathlib import Path
import os,json,torch,tensorrt as trt
from participant.solution_public_trt_local import Solution as LocalSolution
from participant.public_trt_backend import TensorRTEngine

class Solution(LocalSolution):
    def prepare(self,ctx):
        super().prepare(ctx)
        folder=Path(os.environ.get('TTNET_EVENT_TRT_DIR',Path(__file__).resolve().parents[2]/'out/public-event-trt-001/engines'))
        manifest=json.loads((folder/'manifest.json').read_text())
        if manifest['checkpoint_sha256']!=ctx.info['checkpoint_sha256'] or manifest['tensorrt']!=trt.__version__ or manifest['gpu']!=torch.cuda.get_device_name():
            raise ValueError('Event engine checkpoint/runtime/GPU mismatch')
        spec=manifest['event'];engine=TensorRTEngine(folder/spec['engine'],spec['inputs'],spec['outputs']);dtype=getattr(torch,spec['output_dtype'])
        self.trt_engines['event']=engine
        self.original_event_forward=self.system.head.forward
        def forward(features,motion,frame_valid,table):
            return engine(features,motion,frame_valid,*(table[k] for k in ('roi','context','roi_valid','context_valid')))[0].to(dtype)
        self.system.head.forward=forward
        ctx.info.update(trt_stages=['global','table','local','event'],event_trt_directory=str(folder),event_trt_precision=spec['precision'])
