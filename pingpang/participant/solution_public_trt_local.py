"""Current fast solution with only the remaining Local model replaced by TensorRT."""
import json,os
from pathlib import Path
import torch,tensorrt as trt
from participant.solution_public_trt_fast import Solution as FastSolution
from participant.public_trt_backend import TensorRTEngine


class Solution(FastSolution):
    def prepare(self,context):
        super().prepare(context)
        directory=Path(os.environ.get('TTNET_LOCAL_TRT_DIR',Path(__file__).resolve().parents[2]/'out/public-local-trt-001/engines'))
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if manifest['checkpoint_sha256']!=context.info['checkpoint_sha256'] or manifest['tensorrt']!=trt.__version__ or manifest['gpu']!=torch.cuda.get_device_name():
            raise ValueError('Local engine checkpoint/runtime/GPU mismatch')
        spec=manifest['local'];runtime=TensorRTEngine(directory/spec['engine'],spec['inputs'],spec['outputs'])
        self.trt_engines['local']=runtime
        dtypes=[getattr(torch,n) for n in spec['output_dtypes']]
        def forward(*args):return tuple(x.to(dtype) for x,dtype in zip(runtime(*args),dtypes))
        self.system.local_model.forward=forward
        context.info.update(trt_stages=['global','table','local'],local_trt_directory=str(directory),
                            local_trt_precision='FP16 convolutions with FP32 normalization/reduction')
