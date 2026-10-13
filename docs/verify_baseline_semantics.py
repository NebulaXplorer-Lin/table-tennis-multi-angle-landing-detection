"""仅用标准库抽取并执行现有 _emit，验证报告中的时序结论，不运行神经网络。"""
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass
class Landing:
    frame_id: int
    x: float
    y: float


def main():
    source = Path(__file__).resolve().parents[1] / "pingpang/participant/solution.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in
                         {"MIN_GAP", "FALL_AMP", "RISE_AMP"} for t in n.targets)]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Solution")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_emit")
    namespace = {"Landing": Landing}
    exec(compile(ast.Module(body=constants + [method], type_ignores=[]), str(source), "exec"), namespace)
    emit = namespace["_emit"]

    def run(points, future, last=-10_000):
        state = SimpleNamespace(traj=[(f, 100.0, y, 0.9) for f, y in points], last_emit=last)
        return [asdict(x) for x in emit(state, future)]

    cases = [
        ("normal_waits_for_two_future_points", [(1, 10), (2, 15), (3, 10)], 2, -10_000, []),
        ("tail_allows_one_future_point", [(1, 10), (2, 15), (3, 10)], 1, -10_000, [1]),
        ("second_future_y_not_checked", [(1, 10), (2, 15), (3, 10), (4, 100)], 2, -10_000, [1]),
        ("missing_frames_not_rejected", [(1, 10), (20, 15), (40, 10), (60, 10)], 2, -10_000, [19]),
        ("nine_frame_gap_suppressed", [(10, 10), (11, 15), (12, 10), (13, 10)], 2, 2, []),
        ("ten_frame_gap_allowed", [(11, 10), (12, 15), (13, 10), (14, 10)], 2, 2, [11]),
        ("last_point_cannot_be_peak_even_at_finish", [(1, 10), (2, 15)], 1, -10_000, []),
    ]
    results = []
    for name, points, future, last, expected in cases:
        actual = run(points, future, last)
        assert [x["frame_id"] for x in actual] == expected, (name, actual, expected)
        results.append({"case": name, "passed": True, "output": actual})
    print(json.dumps({"scope": "existing _emit only; no ONNX/GPU/video inference",
                      "passed": len(results), "cases": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
