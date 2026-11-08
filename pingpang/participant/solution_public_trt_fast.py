"""Global/table TensorRT plus exact lookup normalization for uncached images."""
from participant.solution_public_trt_both import Solution as BothSolution
from participant.public_fast_normalize import normalize


class Solution(BothSolution):
    def prepare(self,context):
        super().prepare(context)
        self.global_cache.original_normalize=normalize
        context.info['normalization']='Exact uint8 per-channel lookup, float32 CHW; original Global cache retained'

