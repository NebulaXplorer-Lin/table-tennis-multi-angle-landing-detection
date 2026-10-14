"""Local FP32 ONNX Runtime adapter; official TensorRT solution stays intact."""
import os
from pathlib import Path

import numpy as np

from . import solution as official
from .ball_detector import BallDetector


class Solution(official.Solution):
    def prepare(self, context):
        import torch
        import onnxruntime as ort

        if not torch.cuda.is_available():
            raise RuntimeError('Local CUDA GPU unavailable; no automatic CPU fallback.')
        # Load the CUDA/cuDNN DLLs shipped with the matching CUDA PyTorch wheel.
        ort.preload_dlls()
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ONNX Runtime CUDA provider unavailable.')
        gap = int(os.environ.get('LOCAL_MIN_GAP', str(official.MIN_GAP)))
        if gap < 1:
            raise ValueError('MIN_GAP must be a positive integer')
        official.MIN_GAP = gap
        model_path = Path(__file__).resolve().parent / 'weights/ball.onnx'
        if os.name == 'nt':
            model_path = Path('\\\\?\\' + str(model_path))
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(context.output_dir) / 'ort_profile')
        self.session = ort.InferenceSession(model_path.read_bytes(), sess_options=options,
            providers=[('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0}), 'CPUExecutionProvider'])
        self.session.disable_fallback()
        if 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('CUDA session initialization failed; refusing CPU-only execution.')
        input_name = self.session.get_inputs()[0].name
        self.det = BallDetector(lambda x: self.session.run(None, {input_name: x})[0], model_path)
        self.det.detect(np.zeros((288, 512, 3), np.uint8), 512, 288)
        # Verify at least the actual convolution workload executed on the GPU.
        import json
        profile_path = self.session.end_profiling()
        events = json.loads(Path(profile_path).read_text(encoding='utf-8'))
        gpu_conv = [e for e in events if e.get('args', {}).get('provider') == 'CUDAExecutionProvider'
                    and e.get('args', {}).get('op_name') in ('Conv', 'FusedConv')]
        if not gpu_conv:
            raise RuntimeError('No CUDA convolution observed in ORT profiling.')
        context.info.update(backend='onnxruntime_cuda_fp32', min_gap=gap,
                            gpu=torch.cuda.get_device_name(0), ort_version=ort.__version__,
                            torch_version=torch.__version__, gpu_conv_events=len(gpu_conv))
        self.traj = []
        self.last_emit = -10_000
