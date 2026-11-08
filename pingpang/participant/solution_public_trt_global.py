"""Cached public solution with only the Global forward replaced by TensorRT."""
import os
from pathlib import Path
from participant.solution_public_cached import Solution as Base
from participant.public_trt_backend import attach


class Solution(Base):
    trt_stages=('global',)

    def prepare(self,context):
        super().prepare(context)
        self.checkpoint_identity=context.info['checkpoint_sha256']
        directory=Path(os.environ.get('TTNET_TRT_DIR',Path(__file__).resolve().parents[2]/'out/public-trt-001/engines'))
        attach(self,directory,self.trt_stages)
        context.info.update(backend='PyTorch + TensorRT',trt_stages=list(self.trt_stages),trt_directory=str(directory),
                            trt_precision='FP16 convolutions; protected FP32 reductions/normalization')

