"""Local CUDA inference with the official detector and tracker semantics."""

import numpy as np
import torch
import onnxruntime as ort

from .player_detector import IN_H, IN_W, PlayerDetector, weights_path
from .solution import CLASSES, Solution as OfficialSolution
from .tracker import IoUTracker


class Solution(OfficialSolution):
    def prepare(self, context):
        if not torch.cuda.is_available():
            raise RuntimeError("Local solution requires an accessible CUDA GPU")
        ort.preload_dlls()
        session = ort.InferenceSession(
            str(weights_path()),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime failed to initialize CUDA")
        session.disable_fallback()
        input_name = session.get_inputs()[0].name
        self.det = PlayerDetector(
            lambda x: session.run(None, {input_name: x})[0],
            input_size=(IN_H, IN_W), classes=CLASSES, layout="e2e",
        )
        self.det.detect(np.zeros((720, 1280, 3), np.uint8))
        self.tracker = IoUTracker()
        context.info["backend"] = "onnxruntime_cuda_local"
