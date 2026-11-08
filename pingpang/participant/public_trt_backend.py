"""Static-shape TensorRT execution on the caller's PyTorch CUDA stream."""
from pathlib import Path
import json
import torch
import tensorrt as trt


class TensorRTEngine:
    def __init__(self, path, input_names=None, output_names=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot deserialize {path}')
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        if input_names is not None:
            if set(input_names)!=set(self.inputs):raise ValueError('TensorRT input names do not match')
            self.inputs=list(input_names)
        if output_names is not None:
            if set(output_names)!=set(self.outputs):raise ValueError('TensorRT output names do not match')
            self.outputs=list(output_names)
        self.dtypes = {trt.float32:torch.float32,trt.float16:torch.float16,trt.int32:torch.int32,trt.int8:torch.int8,trt.bool:torch.bool}

    def __call__(self, *args):
        if len(args)!=len(self.inputs):
            raise ValueError(f'Expected inputs {self.inputs}')
        tensors=[]
        for name,x in zip(self.inputs,args):
            shape=tuple(self.engine.get_tensor_shape(name))
            if tuple(x.shape)!=shape or not x.is_cuda:
                raise ValueError(f'{name}: expected CUDA {shape}, got {x.device} {tuple(x.shape)}')
            x=x.to(self.dtypes[self.engine.get_tensor_dtype(name)]).contiguous()
            tensors.append(x)
            self.context.set_tensor_address(name,x.data_ptr())
        outputs=[]
        for name in self.outputs:
            x=torch.empty(tuple(self.engine.get_tensor_shape(name)),device=args[0].device,
                          dtype=self.dtypes[self.engine.get_tensor_dtype(name)])
            outputs.append(x)
            self.context.set_tensor_address(name,x.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream(args[0].device).cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')
        return tuple(outputs)


def attach(solution, directory, stages):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['checkpoint_sha256']!=solution.checkpoint_identity:
        raise ValueError('TensorRT engine checkpoint does not match loaded checkpoint')
    if manifest['tensorrt']!=trt.__version__ or manifest['gpu']!=torch.cuda.get_device_name():
        raise ValueError('TensorRT engine must be built for this runtime and GPU')
    solution.trt_engines={}
    for stage in stages:
        spec=manifest[stage]
        backend=TensorRTEngine(directory/spec['engine'],spec['inputs'],spec['outputs'])
        if backend.inputs!=spec['inputs'] or backend.outputs!=spec['outputs']:
            raise ValueError('Unexpected TensorRT tensor order')
        solution.trt_engines[stage]=backend
        dtype=[getattr(torch,n) for n in spec['output_dtypes']]
        def forward(*args,_backend=backend,_dtype=dtype):
            return tuple(x.to(t) for x,t in zip(_backend(*args),_dtype))
        model=solution.system.global_model if stage=='global' else solution.system.table_encoder
        model.forward=forward

