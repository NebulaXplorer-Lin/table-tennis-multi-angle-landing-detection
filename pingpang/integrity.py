"""完整性自校验：检测 run.sh / core / run.py / validate.py 等关键脚本是否被篡改。

原理：
  - 发布前调用 generate() 扫描关键文件算 SHA256，写入 .integrity.json
  - run.sh 启动时调用 verify()，重新扫描同样文件比对 hash
  - 参赛者只覆盖 participant/，不会覆盖 .integrity.json

被校验的文件不包括 participant/（参赛者合法可改）和 public_data/（数据）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 受保护范围：core/、scripts/ 的全部 .py，加顶层入口与说明。
# 从实际文件系统收集，避免列表中不存在的文件被静默跳过。
PROTECTED = sorted(
    [str(p.relative_to(ROOT)) for p in (ROOT / "core").rglob("*.py")]
    + [str(p.relative_to(ROOT)) for p in (ROOT / "scripts").rglob("*.py")]
    + ["run.py", "run.sh", "validate.py", "integrity.py", "README.md"]
)

INTEGRITY_FILE = ROOT / ".integrity.json"


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def generate() -> dict:
    """构建期调用：扫描所有受保护文件，生成 hash 清单。"""
    hashes = {}
    for rel in PROTECTED:
        p = ROOT / rel
        if p.exists():
            hashes[rel] = _hash_file(p)
    return {"protected": hashes}


def verify() -> None:
    """运行期调用：比对当前文件 hash 与构建期记录的 hash。
    不一致则直接退出（exit 4），run.sh 应在启动 run.py 前最先调用本函数。
    """
    if not INTEGRITY_FILE.exists():
        # 容器内必须有 .integrity.json，否则视为环境异常
        print("FATAL: .integrity.json 不存在，镜像可能未通过 build.sh 正规构建",
              file=sys.stderr)
        sys.exit(4)

    expected = json.loads(INTEGRITY_FILE.read_text(encoding="utf-8"))
    expected_hashes = expected.get("protected", {})

    violations = []
    for rel, expected_hash in expected_hashes.items():
        p = ROOT / rel
        if not p.exists():
            violations.append(f"{rel}: 文件缺失")
            continue
        actual = _hash_file(p)
        if actual != expected_hash:
            violations.append(f"{rel}: hash 不匹配（被篡改）")

    if violations:
        print("FATAL: 完整性校验失败，以下关键脚本被篡改：", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print("镜像构建后不允许修改 run.sh / core / run.py 等受保护文件。", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    # CLI: python3 integrity.py --generate | --verify
    if "--generate" in sys.argv:
        doc = generate()
        INTEGRITY_FILE.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"已生成 {INTEGRITY_FILE}（{len(doc['protected'])} 个文件）")
    elif "--verify" in sys.argv:
        verify()
        print("完整性校验通过")
    else:
        print("用法: integrity.py --generate | --verify", file=sys.stderr)
        sys.exit(2)
