"""静态检查 ball.onnx；写出可复核的图结构、逐层清单和计算量。

依赖 onnx、numpy；不执行模型，不修改权重。MAC 仅统计 Conv 和二维 MatMul。
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def shape(value):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param or None
            for d in value.type.tensor_type.shape.dim]


def serialize(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        result = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if array.size <= 32:
            result["values"] = array.tolist()
        return result
    if isinstance(value, (tuple, list)):
        return [serialize(x) for x in value]
    return value


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "pingpang/participant/weights/ball.onnx")
    parser.add_argument("--output-dir", type=Path, default=root / "docs")
    args = parser.parse_args()
    model = onnx.load(str(args.model))
    onnx.checker.check_model(model, full_check=True)
    fixed = copy.deepcopy(model)
    for value in fixed.graph.input:
        value.type.tensor_type.shape.dim[0].dim_value = 1
    inferred = onnx.shape_inference.infer_shapes(fixed, strict_mode=True, data_prop=True)
    shapes = {v.name: shape(v) for v in [*inferred.graph.input, *inferred.graph.value_info,
                                       *inferred.graph.output]}
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    groups = defaultdict(lambda: {"initializer_elements": 0, "conv_matmul_macs": 0})
    consumed = set()
    nodes, layers = [], []
    for index, node in enumerate(model.graph.node):
        parts = node.name.split("/")
        group = parts[2].split(".")[0] if node.name.startswith("/model/") else "other"
        params = 0
        for name in node.input:
            if name in weights and name not in consumed:
                params += int(weights[name].size)
                consumed.add(name)
        groups[group]["initializer_elements"] += params
        macs = 0
        if node.op_type == "Conv":
            output = shapes[node.output[0]]
            assert all(isinstance(x, int) and x > 0 for x in output), output
            macs = math.prod(output) * math.prod(weights[node.input[1]].shape[1:])
        elif node.op_type == "MatMul":
            left, right = shapes[node.input[0]], list(weights[node.input[1]].shape)
            assert len(left) == len(right) == 2 and left[1] == right[0], (left, right)
            macs = left[0] * left[1] * right[1]
        groups[group]["conv_matmul_macs"] += macs
        item = {"index": index, "name": node.name, "op_type": node.op_type,
                "inputs": list(node.input), "outputs": list(node.output),
                "output_shapes_batch1": [shapes.get(x) for x in node.output],
                "attributes": {a.name: serialize(helper.get_attribute_value(a)) for a in node.attribute},
                "initializer_shapes": {x: list(weights[x].shape) for x in node.input if x in weights},
                "conv_matmul_macs_batch1": macs}
        nodes.append(item)
        if node.op_type in {"Conv", "MatMul", "Resize"}:
            layers.append(item)
    total_macs = sum(x["conv_matmul_macs"] for x in groups.values())
    summary = {
        "model": str(args.model.resolve()), "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "file_bytes": args.model.stat().st_size, "producer": model.producer_name,
        "producer_version": model.producer_version, "ir_version": model.ir_version,
        "opsets": {x.domain or "ai.onnx": x.version for x in model.opset_import},
        "metadata": {x.key: x.value for x in model.metadata_props},
        "inputs": [{"name": v.name, "shape": shape(v), "dtype": v.type.tensor_type.elem_type} for v in model.graph.input],
        "outputs": [{"name": v.name, "shape": shape(v), "dtype": v.type.tensor_type.elem_type} for v in model.graph.output],
        "node_count": len(nodes), "operator_counts": dict(Counter(n.op_type for n in model.graph.node)),
        "initializer_count": len(weights), "initializer_elements": sum(int(w.size) for w in weights.values()),
        "initializer_bytes": sum(w.nbytes for w in weights.values()),
        "initializer_dtypes": dict(Counter(str(w.dtype) for w in weights.values())),
        "all_initializers_finite": all(bool(np.isfinite(w).all()) for w in weights.values()),
        "unused_initializers": sorted(set(weights) - consumed),
        "conv_matmul_macs_batch1": total_macs, "conv_matmul_flops_2_per_mac": 2 * total_macs,
        "mac_exclusions": "bias, activations, residual/fusion Add, SE Mul, pooling, Resize, shape operations, transfers",
        "groups": dict(groups), "checker": "full_check passed", "shape_inference": "strict batch=1 passed",
        "analysis_versions": {"onnx": onnx.__version__, "numpy": np.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ball_onnx_graph.json").write_text(
        json.dumps({"summary": summary, "nodes": nodes}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# ball.onnx 算子层清单", "", "由 inspect_ball_onnx.py 生成。序号为 ONNX 图中 0 基节点索引；shape 推断时固定 batch=1。", "",
             "只列 Conv、MatMul、Resize；完整连接与其他算子见 ball_onnx_graph.json。MAC 不包括激活、偏置、插值及逐元素运算。", "",
             "| 节点 | 名称 | 类型 | 权重形状 | 输出形状 | MAC |", "| --- | --- | --- | --- | --- | ---: |"]
    for n in layers:
        lines.append(f"| {n['index']} | `{n['name']}` | {n['op_type']} | `{list(n['initializer_shapes'].values())}` | `{n['output_shapes_batch1']}` | {n['conv_matmul_macs_batch1']:,} |")
    (args.output_dir / "ball_onnx_layers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
