"""Publish an immutable table-label revision and old/new overlays; never train a model."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.review_table_labels import FrameStore, fit, grid, save_rgb, text, write_preview


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def valid_box(box, width, height):
    return (len(box) == 4 and np.isfinite(box).all()
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def apply_revisions(doc, revisions):
    """Only replace an exact, previously reviewed single table annotation."""
    result = deepcopy(doc)
    frames = {f['frame_id']: f for f in result['frames']}
    for revision in revisions:
        frame = frames[revision['frame_id']]
        tables = [a for a in frame['annotations'] if a['label'] == 'table']
        if len(tables) != 1 or tables[0]['bbox_xyxy'] != [revision['old_bbox_xyxy']]:
            raise ValueError(f'old table box mismatch: {doc["video_id"]}/{revision["frame_id"]}')
        box = revision['new_bbox_xyxy']
        if not valid_box(box, doc['frame_width'], doc['frame_height']):
            raise ValueError('replacement table box is invalid')
        tables[0]['bbox_xyxy'] = [box]
    return result


def comparison(rgb, revision):
    image = rgb.copy()
    for key, color in [('old_bbox_xyxy', (255, 163, 65)), ('new_bbox_xyxy', (65, 235, 100))]:
        b = np.rint(revision[key]).astype(int)
        cv2.rectangle(image, tuple(b[:2]), tuple(b[2:]), color, 2, cv2.LINE_AA)
    header = np.full((76, rgb.shape[1], 3), (17, 24, 32), np.uint8)
    text(header, f'{revision["video_id"]} | source frame {revision["frame_id"]} | {revision["split"]}', 23, size=.65)
    text(header, 'ORANGE: old '+str(revision['old_bbox_xyxy']), 46, (255,163,65), .55)
    text(header, 'GREEN: reannotated '+str(revision['new_bbox_xyxy']), 68, (65,235,100), .55)
    return np.vstack([header, image])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revisions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--review-output', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.revisions.read_text())
    base = PROJECT / spec['base_dataset']
    base_meta = json.loads((base / 'metadata/dataset.json').read_text())
    if base_meta['dataset_id'] != spec['base_dataset_id']:
        raise ValueError('base dataset identity mismatch')
    if args.output.exists() or args.review_output.exists():
        raise FileExistsError('use new output directories; revisions are immutable')
    revisions = {(r['video_id'], r['frame_id']): r for r in spec['revisions']}
    if len(revisions) != len(spec['revisions']):
        raise ValueError('duplicate revision key')
    # Validate every source before publishing any files.
    for r in revisions.values():
        source = PROJECT / r['annotation']
        if sha(source) != r['source_annotation_sha256']:
            raise ValueError(f'source annotation hash mismatch: {source}')
        apply_revisions(json.loads(source.read_text()), [r])
    out, review = args.output, args.review_output
    (out / 'labels').mkdir(parents=True)
    (out / 'splits').mkdir()
    review.mkdir(parents=True)
    records, source_hashes, revised_docs, changed, stats = {}, {}, {}, [], {}
    for part in ('train', 'val'):
        split_path = base / f'splits/{part}.jsonl'
        shutil.copyfile(split_path, out / f'splits/{part}.jsonl')
        rows = [json.loads(s) for s in split_path.read_text().splitlines()]
        target = []
        for video in rows:
            vid = video['video_id']
            source = PROJECT / video['annotation']
            source_hashes[video['annotation']] = sha(source)
            relevant = [r for r in revisions.values() if r['video_id'] == vid]
            if any(r['split'] != part or r['annotation'] != video['annotation'] for r in relevant):
                raise ValueError('revision belongs to a different split or source')
            doc = apply_revisions(json.loads(source.read_text()), relevant)
            export_path = out / 'annotations' / Path(video['annotation']).relative_to('data/pp_train_data/annotations')
            write_json(export_path, doc)
            if relevant:
                revised_docs[vid] = doc
                changed.extend((r['video_id'], r['frame_id']) for r in relevant)
            labels = [json.loads(s) for s in (base / f'labels/{vid}.jsonl').read_text().splitlines()]
            timestamps = {r['frame_id']: r['timestamp_sec'] for r in labels}
            for frame in doc['frames']:
                boxes = [b for a in frame['annotations'] if a['label'] == 'table' for b in a['bbox_xyxy']]
                if len(boxes) > 1 or (boxes and not valid_box(boxes[0], doc['frame_width'], doc['frame_height'])):
                    raise ValueError(f'invalid source table label: {vid}/{frame["frame_id"]}')
                revised = (vid, frame['frame_id']) in revisions
                target.append(dict(video_id=vid, frame_id=frame['frame_id'], split=part,
                                   timestamp_sec=timestamps[frame['frame_id']],
                                   resolution=dict(width=doc['frame_width'], height=doc['frame_height']),
                                   table_status='positive' if boxes else 'unknown',
                                   table_bbox_xyxy=boxes[0] if boxes else None,
                                   annotation_source=spec['annotation_method'] if revised else 'original_dataset',
                                   revision_version=spec['version'] if revised else None))
        (out / f'labels/{part}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in target))
        records[part] = target
        stats[part] = dict(videos=len(rows), frames=len(target),
                           positive=sum(r['table_status']=='positive' for r in target),
                           unknown=sum(r['table_status']=='unknown' for r in target),
                           revised=sum(r['revision_version'] is not None for r in target))
    if set(changed) != set(revisions):
        raise ValueError('not every requested revision was applied')
    label_hashes = {part: sha(out / f'labels/{part}.jsonl') for part in ('train', 'val')}
    split_hashes = {part: sha(out / f'splits/{part}.jsonl') for part in ('train', 'val')}
    dataset_id = hashlib.sha256(json.dumps(dict(version=spec['version'], base_dataset_id=base_meta['dataset_id'],
                    label_hashes=label_hashes, split_hashes=split_hashes, revisions_sha256=sha(args.revisions)), sort_keys=True).encode()).hexdigest()
    reference = json.loads((base / 'frame_store.json').read_text())
    # Resolve the existing store correctly even when the output directory has another depth.
    import os
    reference['path'] = os.path.relpath((base / reference['path']).resolve(), out.resolve())
    reference['logical_dataset_id'] = dataset_id
    write_json(out / 'frame_store.json', reference)
    write_json(out / 'metadata/revisions.json', spec)
    write_json(out / 'metadata/source_annotations.json', source_hashes)
    meta = dict(schema='pp_table_dataset_v1', name=spec['version'], dataset_id=dataset_id, status='completed',
                task='table', split_id=spec['split_id'], split_hashes=split_hashes, label_hashes=label_hashes,
                base_dataset_id=base_meta['dataset_id'], stats=stats, revision_spec_sha256=sha(args.revisions),
                target_definition=spec['target_definition'], exhaustive_visual_review=False,
                human_review_status=spec['human_review_status'], coverage='confirmed_wrong_region_errors_only')
    write_json(out / 'metadata/dataset.json', meta)
    (out / 'README.md').write_text(f'''# {spec['version']}\n\n主球台台面修订数据，dataset_id `{dataset_id}`。\n\n仅替换04_004第115–120帧六个错误桌框，六帧均属于val；train监督未改变。固定114/40划分、球和事件标签不变，缺框保持unknown。台面边界由Codex看原图与相邻帧重标，尚未经过独立人工复审；其余原始框未全量精修。\n\n`labels/train.jsonl`、`labels/val.jsonl`供桌框训练读取；`annotations/`保留154份原格式JSON，只修改六个table bbox。`frame_store.json`复用现有画面。来源、修订和哈希见metadata。\n\n读取：`annotations('val', dataset=Path('{out}'))`。训练入口：`python -m ttnet.scene.train --table-dataset {out} --checkpoint <Global权重> --run <新实验目录>`。本次没有启动模型训练。\n''')
    store = FrameStore(base)
    tiles = []
    for r in spec['revisions']:
        rendered = comparison(store.get_rgb(r['video_id'], r['frame_id']), r)
        name = f'{r["video_id"]}_{r["frame_id"]:06d}.jpg'
        save_rgb(review / name, rendered)
        tiles.append(fit(rendered, 800, 500))
    save_rgb(review / 'before_after.jpg', grid(tiles, 3))
    for doc in revised_docs.values():
        write_preview(review, store, doc, highlight_original_anomalies=False,
                      label_name='reannotated dataset label', filename=f'{doc["video_id"]}_reannotated_labels.mp4')
    cards = ''.join(f'<article><h2>{r["video_id"]} · 原帧 {r["frame_id"]}</h2><a href="{r["video_id"]}_{r["frame_id"]:06d}.jpg"><img src="{r["video_id"]}_{r["frame_id"]:06d}.jpg"></a><p>旧框 {r["old_bbox_xyxy"]} → 新框 {r["new_bbox_xyxy"]}</p></article>' for r in spec['revisions'])
    (review / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>桌面重新标注：新旧框对照</title><style>body{max-width:1400px;margin:24px auto;padding:0 20px;background:#111b25;color:#edf2f7;font:16px/1.7 system-ui}img,video{width:100%;height:auto}a{color:#8acbff}article{margin:24px 0;padding:16px;background:#1d2c39;border-radius:8px}</style><h1>桌面重新标注：新旧框对照</h1><p>橙色：错误原始标注；绿色：重新标注的台面框。均为标签，不是模型预测。04_004六帧仍属验证集。遮挡边界参考相邻帧，修订由Codex完成，尚未经过独立人工复审。</p><p>'''+html.escape(str(out))+''' · <a href="manifest.json">修订记录</a> · <a href="validation.json">校验结果</a></p><a href="before_after.jpg"><img src="before_after.jpg"></a>'''+cards+'''<h2>完整视频：绿色为当前修订版标签</h2><video controls preload="metadata" src="04_004_reannotated_labels.mp4"></video></html>''')
    write_json(review / 'manifest.json', dict(dataset=meta, revisions=spec['revisions'], sources_unchanged=True))
    for path, digest in source_hashes.items():
        if sha(PROJECT / path) != digest:
            raise RuntimeError('original annotation changed')
    print(json.dumps(dict(dataset=str(out), dataset_id=dataset_id, review=str(review), stats=stats), ensure_ascii=False))


if __name__ == '__main__':
    main()
