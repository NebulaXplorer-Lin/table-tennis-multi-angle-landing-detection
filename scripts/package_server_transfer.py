"""Package the current working tree, selected reference weights and data lists.

Uses the standard library. Does not train, upload, delete, or follow legacy
directory symlinks. A snapshot includes modified, untracked and ignored recipes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ('ttnet/src', 'configs', 'scripts', 'tools', 'docs', 'pingpang', 'basketball',
             'experiments', 'result')
CODE_FILES = ('ttnet/__init__.py', 'ttnet/README.md', 'ttnet/RESULTS.md', 'AGENT.md',
              'README.md', 'TaskDescription.md', 'what_to_do_next.md', '.gitignore',
              '.gitattributes', '.dockerignore', '.python-version', 'pyproject.toml',
              'uv.lock', 'Dockerfile', 'run_all.sh')
TEXT_SUFFIXES = {'.py', '.json', '.jsonl', '.yaml', '.yml', '.md', '.txt', '.sh',
                 '.toml', '.lock', '.csv', '.ini', '.cfg', '.svg', '.html'}
SKIP_DIRS = {'__pycache__', '.git', '.venv', '.pytest_cache', 'public_data', 'output',
             'out', 'trt_cache'}
WEIGHTS = {
    'global_best.pt': 'archive/20261015-102535_workspace/ttnet/output/20261014-103011_local_fixed3_ball/weights/global_best.pt',
    'local_best.pt': 'archive/20261015-102535_workspace/ttnet/output/20261014-103011_local_fixed3_ball/weights/best.pt',
    'table_legacy.pt': 'out/experiments/20261016_table_ar/table/best.pt',
    'event_tcn_raw_best.pt': 'out/experiments/20261016_table_ar/tcn_frames_lr0.0003_d0.1/checkpoints/best.pt',
    'event_tcn_smoothed_best.pt': 'out/experiments/20261017-112200_tcn_fast_refine/tcn_smoothed/checkpoints/best.pt',
}
DATA_DIRS = ('data/pp_train_data', 'data/pp_public_data', 'data/annotations',
             'data/datasets/pp_ball_v2', 'data/datasets/pp_ball_events_v1',
             'data/datasets/openttgames_event25_v3',
             'data/cache/frames/31be21667d34f45b0c7f',
             'data/cache/frames/d3299f59fa021c8c845e',
             'data/ttnet/training/videos', 'data/ttnet/test/videos')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def files_under(relative, *, code=False):
    root = ROOT / relative
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f'required real directory is missing: {root}')
    result = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(parent)/d).is_symlink()
                         and (not code or d not in SKIP_DIRS))
        for name in sorted(files):
            p = Path(parent)/name
            if p.is_symlink():
                raise ValueError(f'file symlink needs explicit resolution: {p}')
            if not p.is_file() or (code and p.suffix not in TEXT_SUFFIXES):
                continue
            if '\n' in str(p) or '\r' in str(p):
                raise ValueError(f'unsupported newline in file name: {p}')
            result.append(p)
    return result


def record(path, **extra):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=digest(path), **extra)


def add_json(archive, name, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def git(*args):
    p = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def verify(root):
    root = root.resolve()
    snapshot = json.loads((root/'MIGRATION_SNAPSHOT.json').read_text())
    checked = 0
    for row in snapshot['files']:
        path = root / row['path']
        if not path.is_file() or path.is_symlink() or digest(path) != row['sha256']:
            raise ValueError(f'snapshot file missing or changed: {path}')
        checked += 1
    weight_manifest = root/'weights/reference/manifest.json'
    if not weight_manifest.is_file():
        raise ValueError('extract reference_weights.tar.gz before verification')
    for row in json.loads(weight_manifest.read_text())['files']:
        path = root/row['destination']
        if not path.is_file() or path.is_symlink() or digest(path) != row['sha256']:
            raise ValueError(f'weight missing or changed: {path}')
        checked += 1
    print(json.dumps(dict(passed=True, files_checked=checked,
                         note='Code and reference weights only. Verify datasets separately.'), indent=2))


def package(output,profile='prepared'):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = sorted({ROOT/p for p in CODE_FILES} |
                  {p for directory in CODE_DIRS for p in files_under(directory, code=True)})
    missing = [str(p) for p in code if not p.is_file() or p.is_symlink()]
    if missing:
        raise ValueError(f'invalid source files: {missing}')
    snapshot = dict(schema=1, created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    git_head=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                    scope='Allowlisted current source including uncommitted and ignored recipes; not a Git clone.',
                    files=[record(p) for p in code])
    with tarfile.open(output/'code.tar.gz', 'w:gz') as archive:
        for path in code:
            archive.add(path, arcname='pingpong/'+path.relative_to(ROOT).as_posix(), recursive=False)
        add_json(archive, 'pingpong/MIGRATION_SNAPSHOT.json', snapshot)
    weight_records = [record(ROOT/source, destination='weights/reference/'+name)
                      for name,source in WEIGHTS.items()]
    weights = dict(schema=1, files=weight_records,
                   note='Reference checkpoints are byte-identical. Embedded historical configs retain provenance paths; use configs/server for new runs.')
    with tarfile.open(output/'reference_weights.tar.gz', 'w:gz') as archive:
        for row in weight_records:
            archive.add(ROOT/row['path'], arcname='pingpong/'+row['destination'], recursive=False)
        add_json(archive, 'pingpong/weights/reference/manifest.json', weights)
    directories=[d for d in DATA_DIRS if profile!='source-video' or not d.startswith('data/cache/frames/')]
    data_files = sorted({p for directory in directories for p in files_under(directory)})
    relative = [p.relative_to(ROOT).as_posix() for p in data_files]
    (output/'data.files0').write_bytes(b'\0'.join(p.encode() for p in relative)+b'\0')
    (output/'data.files.txt').write_text('\n'.join(relative)+'\n')
    stage1 = [p for p in relative if not p.startswith(('data/datasets/openttgames_event25_v3/',
                'data/cache/frames/d3299f59fa021c8c845e/', 'data/ttnet/'))]
    stage1_set = set(stage1)
    stage2 = [p for p in relative if p not in stage1_set]
    for name,paths in [('data_stage1.files0',stage1), ('data_stage2.files0',stage2)]:
        (output/name).write_bytes(b'\0'.join(p.encode() for p in paths)+b'\0')
    summary = dict(schema=1, profile=profile, source_root=str(ROOT), created_at=snapshot['created_at'],
                   code=dict(files=len(code), unpacked_bytes=sum(p.stat().st_size for p in code)),
                   weights=dict(files=len(weight_records), unpacked_bytes=sum(p['bytes'] for p in weight_records)),
                   data=dict(files=len(data_files), bytes=sum(p.stat().st_size for p in data_files),
                             stage1_bytes=sum((ROOT/p).stat().st_size for p in stage1),
                             stage2_bytes=sum((ROOT/p).stat().st_size for p in stage2),
                             directories=[dict(path=d,bytes=sum(p.stat().st_size for p in data_files if p.is_relative_to(ROOT/d))) for d in directories],
                             verification='For source-video, publish video_store indices on server first. Then run ttnet.cli data verify --full; raw videos were not rehashed while packaging.'),
                   archives=[dict(filename=p.name, bytes=p.stat().st_size, sha256=digest(p))
                             for p in (output/'code.tar.gz',output/'reference_weights.tar.gz')])
    (output/'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    artifacts = ['code.tar.gz','reference_weights.tar.gz','data.files0','data.files.txt',
                 'data_stage1.files0','data_stage2.files0','manifest.json']
    (output/'SHA256SUMS').write_text(''.join(f'{digest(output/name)}  {name}\n' for name in artifacts))
    print(json.dumps(dict(output=str(output),code_files=len(code),weight_files=len(weight_records),
                         data_files=len(data_files),data_gib=summary['data']['bytes']/2**30,
                         stage1_gib=summary['data']['stage1_bytes']/2**30,
                         stage2_gib=summary['data']['stage2_bytes']/2**30,
                         archive_mib={p:(output/p).stat().st_size/2**20 for p in artifacts[:2]}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--output', type=Path)
    group.add_argument('--verify-root', type=Path)
    parser.add_argument('--profile',choices=('prepared','source-video'),default='prepared')
    args = parser.parse_args()
    if args.verify_root:
        verify(args.verify_root)
    else:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
        package(args.output or ROOT/'out/migration'/f'{stamp}_server_transfer',args.profile)
