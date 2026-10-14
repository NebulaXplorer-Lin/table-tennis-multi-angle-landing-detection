#!/usr/bin/env python3
"""Download TTNet's OpenTTGames videos and annotations (Python 3.10+, curl)."""

import argparse
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://lab.osai.ai/datasets/openttgames/data/"
GAMES = {
    "training": [f"game_{i}" for i in range(1, 6)],
    "test": [f"test_{i}" for i in range(1, 8)],
}


def validate_file(path):
    """Reject incomplete ZIPs and HTML error pages saved as dataset files."""
    if path.stat().st_size == 0:
        raise ValueError(f"文件为空：{path}")
    if path.name.endswith((".zip", ".zip.part")):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"ZIP 校验失败：{path} 中的 {bad}")
    else:
        with path.open("rb") as stream:
            header = stream.read(12)
        if header[4:8] != b"ftyp":
            raise ValueError(f"不是有效的 MP4 文件头：{path}")


def download(url, destination, attempts, validator=validate_file):
    if destination.exists():
        validator(destination)
        print(f"已存在，跳过下载：{destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, attempts + 1):
        print(f"下载 [{attempt}/{attempts}]：{url}\n保存：{destination}", flush=True)
        # Start a new curl process on each retry so the resume offset is recalculated.
        result = subprocess.run([
            "curl", "--fail", "--location", "--show-error", "--progress-bar",
            "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "120",
            "--continue-at", "-", "--output", str(partial), url,
        ], check=False)
        if result.returncode == 0:
            validator(partial)
            partial.replace(destination)
            return
        if result.returncode == 33:
            raise RuntimeError(
                f"服务器不支持此断点续传请求；保留了 {partial}。"
                "请更换支持 Range 的下载源，或移走该 .part 文件后重新下载。"
            )
        if attempt < attempts:
            print("下载失败，5 秒后重试；已下载内容保留。", flush=True)
            time.sleep(5)
    raise RuntimeError(f"下载失败：{url}（curl={result.returncode}）；再次运行可续传 {partial}")


def extract_annotations(path):
    # Archives share filenames (ball_markup.json, events_markup.json, masks).
    # Give each game its own directory so later archives cannot overwrite it.
    destination = path.parent / path.stem
    with zipfile.ZipFile(path) as archive:
        # Check every member before writing any, including existing symlink parents.
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"ZIP 包含越界路径：{member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"ZIP 包含符号链接：{member.filename}")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
    print(f"已解压：{path} → {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="下载 TTNet / OpenTTGames 数据集到项目 data/ttnet（约 35 GB）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/ttnet",
                        help="数据集保存目录；默认不受当前工作目录影响")
    parser.add_argument("--split", choices=["all", "train", "test"], default="all",
                        help="选择训练集、测试集或全部")
    parser.add_argument("--only", choices=["all", "videos", "annotations"], default="all",
                        help="选择视频、标注或两者")
    parser.add_argument("--games", nargs="+", metavar="NAME",
                        help="仅下载指定比赛，例如 game_1 game_2 或 test_2")
    parser.add_argument("--base-url", default=BASE_URL, help="包含同名 MP4/ZIP 的下载源")
    parser.add_argument("--attempts", type=int, default=5, help="每个文件最多下载尝试次数")
    parser.add_argument("--no-extract", action="store_true", help="保留标注 ZIP，不解压")
    parser.add_argument("--dry-run", action="store_true", help="仅列出下载地址和保存路径")
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts 必须大于 0")
    splits = ["training", "test"] if args.split == "all" else [
        "training" if args.split == "train" else "test"
    ]
    available = {game for split in splits for game in GAMES[split]}
    if args.games and not set(args.games).issubset(available):
        parser.error(f"--games 必须属于当前 split：{', '.join(sorted(available))}")
    if not args.dry_run and shutil.which("curl") is None:
        parser.error("需要 curl，请先安装系统 curl")

    output = args.output_dir.expanduser().resolve()
    tasks = []
    for split in splits:
        for game in GAMES[split]:
            if args.games and game not in args.games:
                continue
            # Small annotations first, then large videos.
            for kind, extension in [("annotations", "zip"), ("videos", "mp4")]:
                if args.only not in ("all", kind):
                    continue
                filename = f"{game}.{extension}"
                tasks.append((args.base_url.rstrip("/") + "/" + filename,
                              output / split / kind / filename))
    print(f"OpenTTGames：{len(tasks)} 个文件，保存到 {output}", flush=True)
    if args.dry_run:
        for url, destination in tasks:
            print(f"{url}\n  → {destination}")
        return 0

    failures = []
    for url, destination in tasks:
        try:
            download(url, destination, args.attempts)
            if destination.suffix == ".zip" and not args.no_extract:
                extract_annotations(destination)
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
            failures.append(destination.name)
            print(f"失败：{error}", file=sys.stderr, flush=True)
    if failures:
        print(f"未完成：{', '.join(failures)}。再次运行相同命令可续传。", file=sys.stderr)
        return 1
    print(f"下载完成：{output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断；保留 .part 文件，再次运行相同命令可续传。", file=sys.stderr)
        sys.exit(130)
