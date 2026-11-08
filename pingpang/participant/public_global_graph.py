"""Replay the fixed Global TensorRT model; tracker and other networks stay outside."""
import torch

class GlobalGraph:
    def __init__(self,backend):
        self.backend=backend
        self.inputs=[torch.zeros(tuple(backend.engine.get_tensor_shape(n)),device='cuda',dtype=backend.dtypes[backend.engine.get_tensor_dtype(n)]) for n in backend.inputs]
        if len(self.inputs)>1:self.inputs[1].fill_(True)
        self.graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();current=torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):backend(*self.inputs)
        current.wait_stream(stream)
        with torch.cuda.graph(self.graph,stream=stream):
            self.outputs=backend(*self.inputs)
        current.wait_stream(stream)

    def __call__(self,*args):
        if len(args)!=len(self.inputs):raise ValueError('Global graph input count mismatch')
        for source,target in zip(args,self.inputs):
            if source.shape!=target.shape or source.device!=target.device:raise ValueError('Global graph requires original fixed CUDA inputs')
            target.copy_(source)
        self.graph.replay()
        return self.outputs
