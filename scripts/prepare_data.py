#!/usr/bin/env python3
"""Rebuild the project's original datasets using Python 3.10+ and curl."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import zlib

try:
    from . import download_ttnet
except ImportError:
    import download_ttnet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = Path(__file__).with_name("data_manifest.json")
NAMES = ("ttnet", "pp_train_data", "pp_public_data")
ALIASES = {"ttnet": ("openttgames",), "pp_train_data": ("pp_tarin_data",),
           "pp_public_data": ("public_data",)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entries(name):
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["datasets"][name]


def checked_path(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"越界路径：{relative}")
    return path


def verify(root, name):
    records = entries(name)
    print(f"校验 {name}：{len(records)} 个源文件（SHA256）", flush=True)
    for entry in records:
        path = checked_path(root, entry["path"])
        if not path.is_file():
            raise ValueError(f"缺少文件：{path}")
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"文件与本机基准不一致：{path}；请检查数据来源，不会覆盖已有文件")
    return records


def existing(name):
    for path in [DATA / name, *(DATA / alias for alias in ALIASES[name])]:
        if path.is_dir():
            return path
    return None


def canonicalize(name):
    target = DATA / name
    source = existing(name)
    if source and source != target:
        if source.is_symlink() or target.is_symlink():
            raise ValueError(f"请先处理目录符号链接：{source} / {target}")
        source.rename(target)  # Same filesystem: no copying of large videos.
        source.symlink_to(name, target_is_directory=True)
        print(f"目录统一：{source} → {target}（保留旧路径链接）", flush=True)
    return target


def find_source(name, explicit=None):
    if explicit:
        return explicit
    names = (name, *ALIASES[name])
    for directory in (ROOT, DATA, DATA / "_archives"):
        for stem in names:
            for suffix in (".zip", ".tar", ".tar.gz", ".tgz"):
                path = directory / (stem + suffix)
                if path.is_file():
                    return str(path)
        for stem in names:
            path = directory / stem
            if path.is_dir():
                return str(path)
    if name == "pp_public_data":
        if (ROOT / "pingpang/public_data").is_dir():
            return str(ROOT / "pingpang/public_data")
        if (ROOT / "participant.zip").is_file():
            return str(ROOT / "participant.zip")
    raise ValueError(
        f"找不到 {name}。请把同名 ZIP/文件夹放到项目根目录，"
        f"或使用 --{'train' if name == 'pp_train_data' else 'public'}-source 路径/URL。"
        "可在本机运行 --export-pp 导出可迁移的数据包。"
    )


def validate_archive(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC 校验失败：{bad}")
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            archive.getmembers()
    else:
        raise ValueError(f"不是有效的 ZIP/TAR 压缩包：{path}")


def extract_archive(path, target):
    """Validate all archive paths and types before extracting any member."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            for member in members:
                checked_path(target, member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not
                                         (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError(f"ZIP 包含链接/特殊文件：{member.filename}")
            archive.extractall(target)
    else:
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            for member in members:
                checked_path(target, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"TAR 包含链接/特殊文件：{member.name}")
            archive.extractall(target)


def locate_dataset(root, name):
    marker = "annotations" if name == "pp_train_data" else "manifest.json"
    candidates = [root, *(p.parent for p in root.rglob(marker))]
    valid = []
    for path in dict.fromkeys(candidates):
        if name == "pp_train_data":
            ok = (path / "annotations").is_dir() and (path / "rallies_videos").is_dir()
        else:
            ok = (path / "manifest.json").is_file() and (path / "gt_public.jsonl").is_file()
            if ok:
                doc = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                ok = doc.get("name") == "pingpang_landing_v1_public"
        if ok:
            valid.append(path)
    if len(valid) != 1:
        raise ValueError(f"{root} 中需要恰好一份 {name}，找到 {len(valid)} 份")
    return valid[0]


def copy_dataset(source, target, name):
    records = verify(source, name)  # Validate the entire input before any writes.
    for entry in records:
        src = checked_path(source, entry["path"])
        dst = checked_path(target, entry["path"])
        if dst.exists():
            if not dst.is_file() or dst.stat().st_size != entry["size"] or sha256(dst) != entry["sha256"]:
                raise ValueError(f"已有文件不一致，不会覆盖：{dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        partial = dst.with_name(dst.name + ".prepare.part")
        shutil.copyfile(src, partial)
        partial.replace(dst)
    print(f"已准备：{target}", flush=True)


def prepare_pp(name, source, attempts):
    target = canonicalize(name)
    if source.startswith(("http://", "https://")):
        if shutil.which("curl") is None:
            raise ValueError("下载需要系统 curl")
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        local = DATA / "_archives" / f"{name}-{key}.archive"
        download_ttnet.download(source, local, attempts, validator=validate_archive)
    else:
        local = Path(source).expanduser().resolve()
    if local.is_dir():
        copy_dataset(locate_dataset(local, name), target, name)
    elif local.is_file():
        validate_archive(local)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=DATA) as staging:
            extracted = Path(staging)
            extract_archive(local, extracted)
            copy_dataset(locate_dataset(extracted, name), target, name)
    else:
        raise ValueError(f"数据来源不存在：{local}")
    if name == "pp_public_data" and (ROOT / "pingpang").is_dir():
        legacy = ROOT / "pingpang/public_data"
        if not legacy.exists() and not legacy.is_symlink():
            legacy.symlink_to("../data/pp_public_data", target_is_directory=True)
            print(f"公开验证路径：{legacy} → {target}", flush=True)


def verify_extracted(root):
    for split, games in download_ttnet.GAMES.items():
        for game in games:
            path = root / split / "annotations" / f"{game}.zip"
            target = path.with_suffix("")
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    extracted = checked_path(target, member.filename)
                    if not extracted.is_file() or extracted.stat().st_size != member.file_size:
                        raise ValueError(f"标注缺失/大小不符：{extracted}；重新运行准备脚本解压")
                    crc = 0
                    with extracted.open("rb") as stream:
                        for block in iter(lambda: stream.read(1 << 20), b""):
                            crc = zlib.crc32(block, crc)
                    if crc != member.CRC:
                        raise ValueError(f"解压标注 CRC 不符：{extracted}")


def export_pp(destination):
    destination = destination.expanduser().resolve()
    sources = {name: existing(name) for name in NAMES[1:]}
    for name, source in sources.items():
        if source is None:
            raise ValueError(f"请先准备 {name}，然后再导出")
        verify(source, name)
        if (destination / f"{name}.zip").exists():
            raise ValueError(f"导出文件已存在，不会覆盖：{destination / (name + '.zip')}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        target = destination / f"{name}.zip"
        partial = target.with_suffix(".zip.part")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
            for entry in entries(name):
                archive.write(checked_path(source, entry["path"]), f"{name}/{entry['path']}")
        partial.replace(target)
        print(f"已导出：{target}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--train-source", help="PP 训练集目录、ZIP/TAR 或 HTTP(S) URL")
    parser.add_argument("--public-source", help="PP 公开集目录、ZIP/TAR、participant.zip 或 URL")
    parser.add_argument("--base-url", default=download_ttnet.BASE_URL, help="TTNet 下载源")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--check-only", action="store_true", help="只检查数据身份和解压标注，不写文件")
    parser.add_argument("--dry-run", action="store_true", help="只展示准备计划，不写文件或下载")
    parser.add_argument("--export-pp", type=Path, metavar="DIR", help="仅导出本机两份 PP ZIP，带到新环境使用")
    parser.add_argument("--prepare-training", action="store_true", help="准备原始数据后生成现有 TTNet ball_v2 训练帧和窗口")
    parser.add_argument("--workers", type=int, default=4, help="训练数据生成进程数")
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.workers < 1:
        parser.error("--attempts 和 --workers 必须大于 0")
    if sum((args.dry_run, args.check_only, args.export_pp is not None)) > 1:
        parser.error("--dry-run、--check-only、--export-pp 不能同时使用")
    if args.prepare_training and (args.check_only or args.export_pp is not None or
                                  "pp_train_data" not in args.datasets):
        parser.error("--prepare-training 需要准备 pp_train_data，不能与只检查/导出同时使用")
    if args.export_pp is not None:
        export_pp(args.export_pp)
        return 0
    if args.check_only:
        for name in args.datasets:
            source = existing(name)
            if source is None:
                raise ValueError(f"缺少数据目录：{DATA / name}")
            verify(source, name)
            if name == "ttnet":
                verify_extracted(source)
        print("数据校验通过：与本机原始数据基准一致。", flush=True)
        return 0

    # Resolve PP sources before starting the large TTNet download.
    sources = {}
    for name in args.datasets:
        if name != "ttnet":
            explicit = args.train_source if name == "pp_train_data" else args.public_source
            sources[name] = find_source(name, explicit)
    for name in args.datasets:
        print(f"{name}：{sources.get(name, args.base_url)} → {DATA / name}", flush=True)
    if args.dry_run:
        if "ttnet" in args.datasets:
            download_ttnet.main(["--output-dir", str(DATA / "ttnet"), "--base-url", args.base_url, "--dry-run"])
        return 0
    DATA.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        if name == "ttnet":
            target = canonicalize(name)
            result = download_ttnet.main(["--output-dir", str(target), "--base-url", args.base_url,
                                         "--attempts", str(args.attempts)])
            if result:
                return result
            verify(target, name)
        else:
            prepare_pp(name, sources[name], args.attempts)
    if args.prepare_training:
        subprocess.run([sys.executable, "-m", "ttnet.data_tools.prepare", "--workers", str(args.workers)],
                       cwd=ROOT, check=True)
    print("数据准备完成；可运行 --check-only 校验，或 --export-pp DIR 导出 PP 数据包。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断；保留下载/复制进度，重新运行可继续。", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, tarfile.TarError,
            subprocess.CalledProcessError) as error:
        print(f"准备失败：{error}", file=sys.stderr)
        sys.exit(1)
