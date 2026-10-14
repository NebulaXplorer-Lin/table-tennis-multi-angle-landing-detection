"""Local ONNX Runtime entry point for the exported YOLO detector."""

from pathlib import Path

import numpy as np
import torch  # Load the CUDA/cuDNN libraries shared with ONNX Runtime.
import onnxruntime as ort

from .solution import Solution as OfficialSolution
from .solution import CONF_TH
from .yolo_ball_detector import YoloBallDetector


class Solution(OfficialSolution):
    def prepare(self, context):
        if not torch.cuda.is_available():
            raise RuntimeError("Local solution requires an accessible CUDA GPU")
        ort.preload_dlls()
        model_path = Path(__file__).parent / "weights" / "yolo_ball_full_v1.onnx"
        session = ort.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime failed to initialize CUDA")
        session.disable_fallback()
        input_name = session.get_inputs()[0].name
        self.det = YoloBallDetector(lambda x: session.run(None, {input_name: x})[0], size=1280, score_threshold=CONF_TH)
        self.det.detect(np.zeros((1080, 1920, 3), np.uint8), 1920, 1080)
        self.traj = []
        self.last_emit = -10_000
        context.info["backend"] = "onnxruntime_cuda_local"
