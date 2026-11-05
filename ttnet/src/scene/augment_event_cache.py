"""Append independent single-frame table features to an immutable ball cache.

No Global/Local or trajectory inference is repeated. Completion markers contain
both source and output hashes; an interrupted video is rebuilt before skipping.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch

from ttnet.common import PROJECT, jsonl, save, save_jsonl, sha
from ttnet.configuration import load_config
from ttnet.data_tools.geometry import letterbox
from ttnet.events.continuous import attach_pp
from ttnet.events.data import EventCorpus
from ttnet.events.engine import EventSystem, cache_identity, feature_cache_path
from ttnet.models.table_encoder import select_table
from ttnet.scene.visual_cache import restore_visual, serialize_visual, table_image_batch


def verify_base_identity(previous, expected):
    """Only adding a table frontend may differ; decoding cannot be changed."""
    if previous.get('schema') != 3 or previous.get('table_visual'):
        raise ValueError('base must be a ball-only schema3 feature cache')
    if previous.get('table', {}).get('checkpoint_sha256') is not None:
        raise ValueError('base must not contain a legacy table checkpoint')
    if expected.get('schema') != 4 or not expected.get('table_visual'):
        raise ValueError('target must contain a spatial table frontend')
    omit = {'schema', 'table_visual'}
    old = {k:v for k,v in previous.items() if k not in omit}
    new = {k:v for k,v in expected.items() if k not in omit}
    if old != new:
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        raise ValueError('base ball-cache identity mismatch: ' + ', '.join(changed))


def file_pair(directory, vid):
    return dict(npz=sha(directory/f'{vid}.npz'), records=sha(directory/f'{vid}.jsonl'))


def infer_tables(system, corpus, vid, ids, batch):
    spec = system.config['table_frontend']
    frames = corpus.videos[vid]['frames']
    result = {fid:None for fid in ids}
    valid = [fid for fid in ids if frames[fid].get('frame_valid', True)]
    with torch.inference_mode(), torch.autocast(device_type=system.device.type,
            enabled=system.config['amp'] and system.device.type == 'cuda'):
        for start in range(0, len(valid), batch):
            chosen = valid[start:start+batch]
            boxed = [letterbox(corpus.image(vid, fid), *spec['input'], padding_value=114) for fid in chosen]
            images = table_image_batch([im for im,_ in boxed], system.device)
            prediction, spatial = system.table_encoder(images)
            candidates = system.table_encoder.candidates(prediction, spec['confidence'], spec['nms_iou'])
            if len(candidates) != len(chosen) or len(spatial) != len(chosen):
                raise ValueError('table encoder returned an invalid batch')
            for fid, boxes, feature, (_, transform) in zip(chosen, candidates, spatial, boxed):
                result[fid] = system._visual_table(select_table(boxes, transform), feature, transform)
            if start % 512 == 0:
                print(f'table-only {vid}: {start}/{len(valid)}', flush=True)
    return result


def augment_cache(system, corpus, base, destination, batch=8):
    base, dest = Path(base).resolve(), Path(destination).resolve()
    if base == dest:
        raise ValueError('source cache cannot be the destination')
    if batch < 1:
        raise ValueError('batch must be positive')
    if system.config['table_frontend']['source'] != 'independent_yolo':
        raise ValueError('this augmentation uses the independent YOLO single-frame contract')
    previous = json.loads((base/'identity.json').read_text())
    expected = cache_identity(system, corpus)
    verify_base_identity(previous, expected)
    videos = [vid for part in ('train','val') for vid in corpus.split[part]]
    if len(videos) != len(set(videos)):
        raise ValueError('duplicate video in train/val split')
    sources = {vid:file_pair(base, vid) for vid in videos}
    precision = 'AMP' if system.config['amp'] and system.device.type == 'cuda' else 'FP32'
    dest.mkdir(parents=True, exist_ok=True)
    identity_path = dest/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != expected:
        raise ValueError('target feature cache identity mismatch')
    manifest_path = dest/'derivation.json'
    manifest = dict(schema='independent_table_cache_derivation_v1', status='preparing',
                    base=str(base), base_identity=previous, target_identity=expected,
                    implementation_sha256=sha(__file__), table_inference_batch=batch,
                    table_memory_format='contiguous_nchw',
                    table_inference_precision=precision, sources=sources, completed={})
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ('schema','base','base_identity','target_identity','implementation_sha256',
                    'table_inference_batch','table_inference_precision','table_memory_format','sources'):
            if old.get(key) != manifest[key]:
                raise ValueError('cache derivation changed on resume: ' + key)
        manifest = old
    elif any(dest.glob('*.npz')) or any(dest.glob('*.jsonl')):
        raise ValueError('destination has untracked cache files; choose a new directory')
    save(identity_path, expected)
    manifest['status'] = 'augmenting'
    save(manifest_path, manifest)
    system.configure('frozen', False)
    for vid in videos:
        marker = manifest['completed'].get(vid)
        if marker and all((dest/f'{vid}{suffix}').exists() for suffix in ('.npz','.jsonl')):
            if file_pair(dest, vid) == marker['outputs']:
                continue
        # A marker is committed only after NPZ, records and round-trip checks.
        # Mere file existence never makes an interrupted write complete.
        records = jsonl(base/f'{vid}.jsonl')
        with np.load(base/f'{vid}.npz', allow_pickle=False) as archive:
            arrays = {key:archive[key].copy() for key in archive.files}
        if any(key.startswith('table_') for key in arrays):
            raise ValueError('ball-only base unexpectedly contains table arrays')
        ids = [int(fid) for fid in arrays['frame_ids']]
        if ids != sorted(corpus.videos[vid]['frames']) or [r['frame_id'] for r in records] != ids:
            raise ValueError('base cache frame coverage/order mismatch: ' + vid)
        if arrays['features'].shape != (len(ids),256) or not np.isfinite(arrays['features']).all():
            raise ValueError('invalid base ball descriptors: ' + vid)
        tables = infer_tables(system, corpus, vid, ids, batch)
        enriched = [dict(rec, table=tables[rec['frame_id']]) for rec in records]
        extra, clean = serialize_visual(enriched, system.config['table_frontend']['channels'])
        if [{k:v for k,v in r.items() if k != 'table'} for r in clean] != [{k:v for k,v in r.items() if k != 'table'} for r in records]:
            raise AssertionError('table augmentation modified ball trajectory records')
        temporary = dest/f'{vid}.tmp.npz'
        np.savez_compressed(temporary, **arrays, **extra)
        with np.load(temporary, allow_pickle=False) as written:
            for key, value in arrays.items():
                np.testing.assert_array_equal(written[key], value)
            restored = {r['frame_id']:deepcopy(r) for r in clean}
            restore_visual(written, ids, restored, system.config['table_frontend']['channels'])
        if file_pair(base, vid) != sources[vid]:
            raise ValueError('source cache changed during augmentation: ' + vid)
        temporary.replace(dest/f'{vid}.npz')
        save_jsonl(dest/f'{vid}.jsonl', clean)
        manifest['completed'][vid] = dict(source=sources[vid], outputs=file_pair(dest, vid), frames=len(ids),
                                          ball_arrays_exact=True, ball_records_exact_excluding_table=True)
        save(manifest_path, manifest)
        print(f'table-augmented {vid}: {len(ids)}', flush=True)
    if set(manifest['completed']) != set(videos):
        raise AssertionError('incomplete cache derivation')
    manifest['status'] = 'completed'
    save(manifest_path, manifest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--output', type=Path, help='optional JSON result receipt')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT/'out/yolo_setup/event_table_config'))
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    torch.set_num_threads(4)
    cfg, _ = load_config(args.config)
    corpus = EventCorpus(cfg['dataset'])
    if cfg.get('sampling_protocol') == 'openttgames_pp_50_50':
        attach_pp(corpus, cfg['pp_annotations'])
    system = EventSystem(cfg, torch.device(args.device))
    destination = args.destination or feature_cache_path(system, corpus)
    result = augment_cache(system, corpus, args.base, destination, args.batch)
    receipt = dict(status='completed', cache=str(result), base=str(args.base.resolve()),
                   memory_format='contiguous_nchw',
                   derivation=str(result/'derivation.json'), batch=args.batch,
                   precision='AMP' if cfg['amp'] and system.device.type == 'cuda' else 'FP32')
    if args.output:
        save(args.output, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
