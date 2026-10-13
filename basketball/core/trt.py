"""ONNX -> TensorRT engine：转换 / 缓存 / 加载。

engine 缓存路径：<trt_cache>/<tensorrt_version>/<gpu_name>/<stem>-<onnx_md5[:12]>.engine
按 TRT 版本 + GPU 型号 + onnx 内容哈希分目录；评测机首跑自动从 ONNX 重建（约几分钟，
只应发生在 prepare 阶段）。engine 反序列化失败（版本/GPU 不匹配）视为未命中缓存。

设备缓冲用 torch 分配（基础镜像自带 PyTorch cu118；本模块不依赖 torch 的算子）。
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

# 累计 engine 构建耗时（秒）；run_status 的 timing.engine_build_sec 读它
BUILD_SECONDS_TOTAL = 0.0


class TrtError(RuntimeError):
    """TRT 不可用 / engine 加载失败。"""


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        return out.stdout.decode("utf-8").strip().split("\n")[0] or "unknown-gpu"
    except Exception:
        return "unknown-gpu"


def model_hash(onnx_path: str | Path) -> str:
    h = hashlib.md5()
    with open(onnx_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def expected_engine_path(trt_cache: str | Path, onnx_path: str | Path) -> Path:
    import tensorrt as trt  # noqa: F401  仅取版本；失败由调用方处理

    stem = Path(onnx_path).stem
    return (Path(trt_cache) / trt.__version__ / gpu_name()
            / f"{stem}-{model_hash(onnx_path)}.engine")


def build_engine(onnx_path: str | Path, engine_path: str | Path,
                 input_name: str, shape: tuple[int, ...],
                 fp16: bool = True, workspace_gb: int = 1) -> None:
    """固定 shape（batch 1）+ 可选 FP16 编译 engine。"""
    global BUILD_SECONDS_TOTAL
    import tensorrt as trt

    t0 = time.time()

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise TrtError(f"ONNX 解析失败: {errs[:3]}")
    serialized = builder.build_serialized_network(network, config)
    if not serialized:
        raise TrtError("engine 构建失败")
    Path(engine_path).write_bytes(serialized)
    BUILD_SECONDS_TOTAL += time.time() - t0


class TrtModel:
    """单输入单输出 engine 的最小推理封装。"""

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt
        import torch

        self._torch = torch
        data = Path(engine_path).read_bytes()
        logger = trt.Logger(trt.Logger.WARNING)
        engine = trt.Runtime(logger).deserialize_cuda_engine(data)
        if engine is None:
            raise TrtError(f"engine 反序列化失败: {engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()

        self._input_idx, self._output_idx = None, None
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                if self._input_idx is not None:
                    raise TrtError("仅支持单输入模型")
                self._input_idx = i
            elif self._output_idx is None:
                self._output_idx = i
        if self._input_idx is None or self._output_idx is None:
            raise TrtError("找不到输入/输出 binding")
        self.input_shape = tuple(engine.get_binding_shape(self._input_idx))
        self.output_shape = tuple(engine.get_binding_shape(self._output_idx))
        self._stream = torch.cuda.Stream()

    def infer(self, x_np):
        """x_np: float32 ndarray，shape 需与 engine 输入一致。返回输出 ndarray。"""
        torch = self._torch
        x = torch.from_numpy(np_contiguous(x_np)).cuda()
        out = torch.empty(self.output_shape, dtype=torch.float32, device="cuda")
        if tuple(x.shape) != self.input_shape:  # 固定 shape engine 无需设置
            name = self.engine.get_binding_name(self._input_idx)
            self.context.set_input_shape(name, tuple(x.shape))
        bindings = [0] * self.engine.num_bindings
        bindings[self._input_idx] = int(x.data_ptr())
        bindings[self._output_idx] = int(out.data_ptr())
        ok = self.context.execute_async_v2(bindings, self._stream.cuda_stream)
        if not ok:
            raise TrtError("execute_async_v2 失败")
        self._stream.synchronize()
        return out.cpu().numpy()


def np_contiguous(x):
    import numpy as np

    if not x.flags["C_CONTIGUOUS"]:
        x = x.copy(order="C")
    return x


def load_or_build(onnx_path: str | Path, trt_cache: str | Path,
                  input_name: str, shape: tuple[int, ...],
                  fp16: bool = True, log=print):
    """优先加载缓存 engine；未命中/不匹配则从 ONNX 重建。返回 (TrtModel, "cached"|"built")。"""
    ep = expected_engine_path(trt_cache, onnx_path)
    if ep.exists():
        try:
            return TrtModel(ep), "cached"
        except TrtError as e:
            log(f"TRT engine 缓存不可用，将重建: {ep}（{e}）")
    log(f"TRT engine 不存在，从 ONNX 构建: {ep}\n"
        f"  building TensorRT engine, this may take several minutes（约 5 分钟，属正常，勿中断）")
    build_engine(onnx_path, ep, input_name, shape, fp16=fp16)
    return TrtModel(ep), "built"
