"""Merge coarse CVAT event exports into an immutable copy of the PP ball split.

Only explicitly tagged frames provide event evidence. Unmarked frames remain
unknown, Net stays masked, and ball centers never become contact annotations.
No images are decoded or copied; the existing array store is referenced.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from zoneinfo import ZoneInfo

from ttnet.common import DATA, PROJECT, identity, jsonl, save, save_jsonl, sha

SCHEMA = 'ttnet_ball_events_v1.0'
LABELS = {
    'table_contact':'bounce', 'table_bounce':'bounce', 'touch_table':'bounce', 'table':'bounce',
    'net_contact':'net', 'net hit':'net', 'touch_net':'net', 'net':'net',
    'racket_hit':'racket_hit', 'floor_bounce':'floor_bounce',
    'ignore':'ignore', 'uncertain_contact':'ignore',
}
POLICY = dict(version=1,frame_index_base=0,time_alignment='exact original frame ID; preserve source PTS',
    unmarked='unknown; no implicit negatives',net='preserve explicit tags; all Net supervision masked pending review',
    hard_negatives='only explicit racket_hit/floor_bounce frames supervise Bounce=0',
    ignore='ignore/uncertain_contact overrides all event supervision on that frame',
    timing='coarse manual frame; no shifts, expansion, smoothing or timing correction',
    contact_coordinates='unannotated; ball center is a separate reference only',
    continuous_validation_ready=False,label_aliases=LABELS)


def timestamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def video_id_for(path, task_name='', task_source='', declared=None):
    """Resolve names/metadata, never guess a video from its length or CVAT task ID."""
    found=set()
    if declared:
        if not re.fullmatch(r'\d{2}_\d{3}',declared):raise ValueError(f'invalid video ID: {declared}')
        found.add(declared)
    group=next((p.name for p in path.parents if re.fullmatch(r'\d{2}',p.name)),None)
    for name in (path.stem,path.parent.name,task_name,task_source):
        match=re.search(r'(?<!\d)(\d{2})_(\d{3})(?!\d)',name)
        if match:found.add('_'.join(match.groups()))
        compact=re.fullmatch(r'(\d{2})(\d{3})(?:\.mp4)?',name)
        if compact:found.add('_'.join(compact.groups()))
        clip=re.fullmatch(r'(\d{3})(?:\.mp4)?',name)
        if clip and group:found.add(f'{group}_{clip[1]}')
    if len(found)!=1:raise ValueError(f'cannot uniquely identify {path}: {sorted(found)}')
    return next(iter(found))


def label_tag(fid,label,source='manual'):
    name=label.strip().lower()
    if name not in LABELS:raise ValueError(f'unsupported event label: {label!r}')
    if type(fid) is not int or fid<0:raise ValueError(f'invalid event frame ID: {fid}')
    return dict(frame_id=fid,raw_label=label,kind=LABELS[name],source=source)


def parse_xml(data, path):
    root=ET.fromstring(data)
    if root.tag!='annotations':raise ValueError(f'not a CVAT export: {path}')
    meta=root.find('meta/task')
    if meta is None:meta=root.find('meta/job')
    if meta is None:raise ValueError(f'missing CVAT metadata: {path}')
    if (meta.findtext('frame_filter') or '') not in ('','step=1'):
        raise ValueError(f'event export uses non-native frame stride: {path}')
    count=int(meta.findtext('size'))
    if int(meta.findtext('start_frame') or 0)!=0 or int(meta.findtext('stop_frame') or count-1)!=count-1:
        raise ValueError(f'event export is not a full original-frame timeline: {path}')
    size=root.find('meta/original_size')
    if size is None:size=meta.find('original_size')
    resolution=dict(width=int(size.findtext('width')),height=int(size.findtext('height'))) if size is not None else None
    tags=[];frame_ids=[]
    if root.findall('track'):raise ValueError(f'event tracks need explicit review, not interpolation: {path}')
    for im in root.findall('image'):
        fid=int(im.get('id'));frame_ids.append(fid)
        name=re.search(r'frame_(\d+)',im.get('name',''))
        if name and int(name[1])!=fid:raise ValueError(f'frame name/id mismatch: {path}:{fid}')
        dims=dict(width=int(im.get('width')),height=int(im.get('height')))
        if resolution is None:resolution=dims
        if dims!=resolution:raise ValueError(f'variable export resolution: {path}')
        for tag in im:
            if tag.tag!='tag':raise ValueError(f'unsupported event shape {tag.tag}: {path}')
            tags.append(label_tag(fid,tag.get('label'),tag.get('source','manual')))
    if len(frame_ids)!=len(set(frame_ids)) or any(not 0<=i<count for i in frame_ids):
        raise ValueError(f'invalid/duplicate export frame IDs: {path}')
    return dict(video_id=video_id_for(path,meta.findtext('name') or '',meta.findtext('source') or ''),
        frame_count=count,resolution=resolution,source_sha256=None,tags=tags,
        represented_frames=len(frame_ids),format='cvat_xml',task_id=meta.findtext('id'))


def parse_json(document,path):
    if document.get('frame_index_base')!=0:raise ValueError(f'nonzero frame index base: {path}')
    dm=document['data_meta'];count=document['frame_count']
    if type(count) is not int or count<1 or dm['size']!=count or dm.get('start_frame',0)!=0 or dm.get('stop_frame',count-1)!=count-1:
        raise ValueError(f'JSON export frame count/range differs: {path}')
    if dm.get('frame_filter','') not in ('','step=1') or dm.get('deleted_frames'):
        raise ValueError(f'JSON export has skipped/deleted frames: {path}')
    ca=document['cvat_annotations']
    if any(ca.get(k) for k in ('shapes','tracks','intervals')):raise ValueError(f'non-tag JSON annotations: {path}')
    tags=[label_tag(e['frame_id'],e['type'],e.get('source','manual')) for e in document['events']]
    names={l['id']:l['name'] for l in document['labels']}
    raw=[label_tag(t['frame'],names[t['label_id']],t.get('source','manual')) for t in ca['tags']]
    if Counter((t['frame_id'],t['raw_label']) for t in raw)!=Counter((t['frame_id'],t['raw_label']) for t in tags):
        raise ValueError(f'JSON event summary differs from CVAT tags: {path}')
    dims={ (f['width'],f['height']) for f in dm['frames'] }
    if len(dims)!=1:raise ValueError(f'ambiguous JSON resolution: {path}')
    w,h=next(iter(dims))
    return dict(video_id=video_id_for(path,document.get('task_name',''),declared=document['video_id']),
        frame_count=count,resolution=dict(width=w,height=h),source_sha256=document['source_sha256'],
        tags=tags,represented_frames=count,format='cvat_json',task_id=document.get('task_id'))


def collect_exports(source):
    source=Path(source);exports=defaultdict(list);inventory=[];indexes=[]
    for path in sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in ('.zip','.xml','.json')):
        record=dict(path=path.relative_to(source).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
        if path.suffix.lower()=='.zip':
            with zipfile.ZipFile(path) as archive:
                names=[n for n in archive.namelist() if n.lower().endswith('.xml')]
                if len(names)!=1:raise ValueError(f'expected one CVAT XML in {path}')
                data=archive.read(names[0]);record['member']=names[0];record['xml_sha256']=hashlib.sha256(data).hexdigest()
            parsed=parse_xml(data,path)
        elif path.suffix.lower()=='.xml':parsed=parse_xml(path.read_bytes(),path)
        else:
            doc=json.loads(path.read_text(encoding='utf-8-sig'))
            if 'video_id' not in doc:
                if path.name!='manifest.json' or 'tasks' not in doc:raise ValueError(f'unknown JSON file: {path}')
                inventory.append(dict(record,role='export_manifest'));indexes.append(doc);continue
            parsed=parse_json(doc,path)
        inventory.append(dict(record,video_id=parsed['video_id'],role='annotation'))
        parsed['provenance']=record;exports[parsed['video_id']].append(parsed)
    for doc in indexes:
        for vid,task in doc['tasks'].items():
            if vid not in exports:raise ValueError(f'manifest video has no export: {vid}')
            hashes={e['source_sha256'] for e in exports[vid] if e['source_sha256']}
            if hashes and hashes!={task['source_sha256']}:raise ValueError(f'manifest source hash conflict: {vid}')
    if not exports:raise ValueError('no event exports found')
    return exports,inventory


def reconcile(vid,exports,metadata):
    signatures=[]
    for export in exports:
        if export['frame_count']!=metadata['actual_frames'] or export['resolution']!=metadata['resolution']:
            raise ValueError(f'frame count or resolution differs from ball data: {vid}')
        if export['source_sha256'] and export['source_sha256']!=metadata['video_sha256']:
            raise ValueError(f'annotation source video hash mismatch: {vid}')
        if any(t['frame_id']>=metadata['actual_frames'] for t in export['tags']):raise ValueError(f'event frame out of range: {vid}')
        signatures.append(sorted(set((t['frame_id'],t['raw_label'],t['kind']) for t in export['tags'])))
    if any(s!=signatures[0] for s in signatures[1:]):
        raise ValueError(f'conflicting duplicate XML/ZIP/JSON annotations: {vid}; resolve before merging')
    preferred=next((e for e in exports if e['format']=='cvat_json'),exports[0])
    unique={}
    for tag in preferred['tags']:unique.setdefault((tag['frame_id'],tag['raw_label']),tag)
    tags=sorted(unique.values(),key=lambda t:(t['frame_id'],t['raw_label']))
    warnings=[]
    if not tags:warnings.append('empty_event_export_all_frames_unknown')
    if len(preferred['tags'])!=len(tags):warnings.append('duplicate_same_frame_label_deduplicated')
    if any(e['represented_frames']==0 for e in exports):warnings.append('export_has_no_image_entries')
    return tags,dict(formats=sorted({e['format'] for e in exports}),representations=len(exports),
        raw_tag_count=len(preferred['tags']),unique_tag_count=len(tags),
        source_hash_verified_in_export=any(e['source_sha256'] for e in exports),
        provenance=[e['provenance'] for e in exports],warnings=warnings)


def frame_annotation(tags):
    kinds=sorted({t['kind'] for t in tags});raw=sorted({t['raw_label'] for t in tags})
    status=['unknown','unknown'];target=[None,None];supervised=[False,False];issues=[]
    if 'bounce' in kinds:status[0]='positive';target[0]=1.;supervised[0]=True
    if 'net' in kinds:status[1]='positive';target[1]=1.;issues.append('net_pending_review')
    negative=bool({'racket_hit','floor_bounce'} & set(kinds))
    if negative:
        if 'bounce' in kinds:status[0]='conflict';target[0]=None;supervised[0]=False;issues.append('bounce_and_hard_negative_conflict')
        else:status[0]='negative';target[0]=0.;supervised[0]=True
    if 'ignore' in kinds:
        status=['ignore','ignore'];target=[None,None];supervised=[False,False]
        if len(kinds)>1:issues.append('ignore_overrides_other_tags')
    return dict(kinds=kinds,raw_labels=raw,class_order=['bounce','net'],status=status,
        target=target,supervised=supervised,quality='coarse_manual' if tags else 'unknown',issues=issues)


def merge(source,ball_root,output,project=PROJECT):
    source=Path(source).resolve();ball_root=Path(ball_root).resolve();output=Path(output).resolve();project=Path(project)
    if output.exists():raise FileExistsError(f'use a new dataset version: {output}')
    if source==output or source in output.parents or output in source.parents or output==ball_root or ball_root in output.parents or output in ball_root.parents:
        raise ValueError('output must be separate from annotation source and original ball dataset')
    base=json.loads((ball_root/'metadata/dataset.json').read_text())
    split_hashes={p:sha(ball_root/f'splits/{p}.jsonl') for p in ('train','val')}
    if split_hashes!=base['split_hashes']:raise ValueError('frozen ball split file hashes changed')
    splits={p:jsonl(ball_root/f'splits/{p}.jsonl') for p in ('train','val')}
    entries={};groups={}
    for part,rows in splits.items():
        for entry in rows:
            vid=entry['video_id']
            if vid in entries or entry['split']!=part:raise ValueError('video split leakage')
            if entry['group'] in groups and groups[entry['group']]!=part:raise ValueError('match group split leakage')
            groups[entry['group']]=part;entries[vid]=entry
    exports,inventory=collect_exports(source)
    if set(exports)-set(entries):raise ValueError(f'event videos outside frozen PP data: {sorted(set(exports)-set(entries))}')
    metadata={};annotations={};audit={};input_hashes={}
    for vid,entry in entries.items():
        meta=json.loads((ball_root/f'metadata/videos/{vid}.json').read_text());metadata[vid]=meta
        if meta['video_sha256']!=entry['video_sha256'] or sha(project/entry['video'])!=entry['video_sha256']:
            raise ValueError(f'original source video content changed: {vid}')
        if vid in exports:annotations[vid],audit[vid]=reconcile(vid,exports[vid],meta)
        else:annotations[vid]=[];audit[vid]=dict(warnings=['missing_event_export_all_frames_unknown'],provenance=[],unique_tag_count=0,raw_tag_count=0)
        input_hashes[vid]=sha(ball_root/f'labels/{vid}.jsonl')
        if input_hashes[vid]!=meta['labels_sha256']:raise ValueError(f'ball label content changed: {vid}')
    ref=json.loads((ball_root/'frame_store.json').read_text())
    store=(ball_root/ref['path']).resolve()
    store_meta=json.loads((store/'manifest.json').read_text())
    if store_meta['status']!='completed' or store_meta['storage_id']!=ref['storage_id']:raise ValueError('frame store identity mismatch')
    dataset_id=identity(dict(base_dataset_id=base['dataset_id'],base_label_hashes=input_hashes,
        split_hashes=split_hashes,source_inventory=inventory,policy=POLICY,implementation_sha256=sha(__file__)))
    staging=output.with_name(output.name+'.partial')
    if staging.exists():raise FileExistsError(f'previous incomplete merge exists: {staging}')
    staging.mkdir(parents=True)
    save(staging/'metadata/dataset.json',dict(schema=SCHEMA,status='building',dataset_id=dataset_id))
    try:
        (staging/'splits').mkdir();(staging/'windows').mkdir()
        for part in splits:shutil.copyfile(ball_root/f'splits/{part}.jsonl',staging/f'splits/{part}.jsonl')
        # Window bytes and frame pixels are unchanged. Only lightweight metadata is copied.
        for path in (ball_root/'windows').glob('*.jsonl'):shutil.copyfile(path,staging/'windows'/path.name)
        frame_ref=dict(ref,path=os.path.relpath(store,output),logical_dataset_id=dataset_id,base_dataset_id=base['dataset_id'])
        save(staging/'frame_store.json',frame_ref)
        save(staging/'metadata/event_policy.json',POLICY)
        save(staging/'metadata/event_sources.json',dict(root=str(source),files=inventory,source_identity=identity(inventory)))
        per_split={};all_issues=[]
        for part,entries_part in splits.items():
            combined=[];events_part=[];counts=Counter();raw_counts=Counter();video_counts=Counter()
            for entry in entries_part:
                vid=entry['video_id'];rows=jsonl(ball_root/f'labels/{vid}.jsonl')
                if [r['frame_id'] for r in rows]!=list(range(metadata[vid]['actual_frames'])):raise ValueError(f'ball frames not contiguous: {vid}')
                by_frame=defaultdict(list)
                for tag in annotations[vid]:by_frame[tag['frame_id']].append(tag)
                events=[];updated=[]
                for row in rows:
                    if row['video_id']!=vid or row['split']!=part:raise ValueError(f'ball row identity mismatch: {vid}')
                    fid=row['frame_id'];annotation=frame_annotation(by_frame[fid])
                    counts.update(['bounce_'+annotation['status'][0],'net_'+annotation['status'][1]])
                    counts['bounce_supervised']+=int(annotation['supervised'][0]);counts['net_supervised']+=int(annotation['supervised'][1])
                    if annotation['kinds']:
                        counts['tagged_frames']+=1;raw_counts.update(annotation['raw_labels'])
                        event=dict(video_id=vid,frame_id=fid,timestamp_sec=row['timestamp_sec'],split=part,
                            **annotation,contact_xy=None,contact_xy_status='not_annotated',
                            ball_center_reference=row['ball_centers'][0] if row['ball_centers'] else None,
                            ball_position_label_status=row['label_status'])
                        events.append(event)
                        if annotation['issues']:all_issues.append(dict(video_id=vid,frame_id=fid,issues=annotation['issues'],kinds=annotation['kinds']))
                    updated.append(dict(row,event_annotation=annotation))
                combined.extend(updated);events_part.extend(events)
                save_jsonl(staging/f'labels/{vid}.jsonl',updated);save_jsonl(staging/f'events/{vid}.jsonl',events)
                vm=dict(metadata[vid],base_labels_sha256=input_hashes[vid],labels_sha256=sha(staging/f'labels/{vid}.jsonl'),
                    events_sha256=sha(staging/f'events/{vid}.jsonl'),
                    event_annotations=dict(dataset_id=dataset_id,frames_with_tags=len(events),**audit[vid]))
                save(staging/f'metadata/videos/{vid}.json',vm)
                video_counts['videos']+=1;video_counts['with_tags']+=bool(events)
            combined.sort(key=lambda r:(r['video_id'],r['frame_id']));events_part.sort(key=lambda r:(r['video_id'],r['frame_id']))
            save_jsonl(staging/f'labels/{part}.jsonl',combined);save_jsonl(staging/f'events/{part}.jsonl',events_part)
            per_split[part]=dict(frames=len(combined),event_counts=dict(counts),raw_label_counts=dict(raw_counts),**video_counts)
        report=dict(status='merged_coarse_annotations',dataset_id=dataset_id,base_dataset_id=base['dataset_id'],
            original_split_preserved=True,split_hashes=split_hashes,source_files=len(inventory),export_videos=len(exports),
            total_videos=len(entries),frames=sum(x['frames'] for x in per_split.values()),per_split=per_split,
            warnings=[dict(video_id=v,warning=w) for v,a in audit.items() for w in a['warnings']],
            per_frame_issues=all_issues,duplicate_representations=sum(max(0,len(e)-1) for e in exports.values()),
            ready_for_continuous_selection=False,contact_coordinates_annotated=False,net_supervision_enabled=False,
            interpretation='Sparse coarse event evidence only. Unmarked frames remain unknown. Needs exhaustive review before continuous FP/threshold/checkpoint selection.')
        save(staging/'metadata/event_merge.json',report)
        save(staging/'metadata/dataset.json',dict(base,schema=SCHEMA,status='completed',dataset_id=dataset_id,
            base_dataset_id=base['dataset_id'],base_dataset_path=str(ball_root),finished=timestamp(),
            merge_implementation_sha256=sha(__file__),
            event_source_identity=identity(inventory),event_annotations=dict(quality='coarse_manual',continuous_validation_ready=False),
            event_hashes={p:sha(staging/f'events/{p}.jsonl') for p in splits},
            label_hashes={p:sha(staging/f'labels/{p}.jsonl') for p in splits}))
        summary=['# PP 位置与粗事件合并','','状态：合并完成；未标注事件帧为 unknown，连续事件选型仍需核验。','',
            '| 集合 | 视频 | 帧 | Bounce 正帧 | 明确 Bounce 负帧 | Bounce unknown | Net 标记正帧（屏蔽监督） |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for part,r in per_split.items():
            c=r['event_counts'];summary.append(f"| {part} | {r['videos']} | {r['frames']} | {c.get('bounce_positive',0)} | {c.get('bounce_negative',0)} | {c.get('bounce_unknown',0)} | {c.get('net_positive',0)} |")
        summary+=['','沿用原视频划分与完整球框/球心标签，复用原 NPY 帧库。所有事件仍为粗帧标记；contact_xy 未标注，原球心单独保存为参考。',
            '重复 XML/ZIP/JSON 已核对一致后去重；ignore 覆盖同帧事件监督。','',
            f"来源视频：{len(exports)}/{len(entries)}；空标注/去重提示见 [审计](metadata/event_merge.json)。"]
        (staging/'README.md').write_text('\n'.join(summary)+'\n')
        staging.replace(output)
        return report
    except BaseException as error:
        save(staging/'metadata/failure.json',dict(status='failed',error=repr(error),updated=timestamp()))
        raise


def verify(merged,ball_root):
    merged=Path(merged);ball_root=Path(ball_root)
    meta=json.loads((merged/'metadata/dataset.json').read_text());base=json.loads((ball_root/'metadata/dataset.json').read_text())
    if meta['status']!='completed' or meta['base_dataset_id']!=base['dataset_id']:raise ValueError('merged dataset identity mismatch')
    frames=videos=0;supervised=Counter()
    for part in ('train','val'):
        if sha(merged/f'splits/{part}.jsonl')!=sha(ball_root/f'splits/{part}.jsonl'):raise ValueError('split changed')
        original=jsonl(ball_root/f'labels/{part}.jsonl');rows=jsonl(merged/f'labels/{part}.jsonl')
        if len(rows)!=len(original):raise ValueError('merged row count changed')
        if sha(merged/f'labels/{part}.jsonl')!=meta['label_hashes'][part]:raise ValueError('merged labels changed')
        if sha(merged/f'events/{part}.jsonl')!=meta['event_hashes'][part]:raise ValueError('merged events changed')
        expected={}
        for a,b in zip(rows,original):
            stripped={k:v for k,v in a.items() if k!='event_annotation'}
            if stripped!=b:raise ValueError('ball positions, labels, order or timestamps changed')
            ev=a['event_annotation']
            if ev!=frame_annotation([label_tag(a['frame_id'],label) for label in ev['raw_labels']]):raise ValueError('event mask or label policy changed')
            supervised['bounce']+=ev['supervised'][0];supervised['net']+=ev['supervised'][1]
            if not ev['kinds'] and (any(ev['supervised']) or ev['target']!=[None,None]):raise ValueError('implicit negative introduced')
            if ev['supervised'][1]:raise ValueError('unreviewed Net is supervised')
            if 'ignore' in ev['kinds'] and any(ev['supervised']):raise ValueError('ignore is supervised')
            expected.setdefault(a['video_id'],[]).append(a)
        for entry in jsonl(merged/f'splits/{part}.jsonl'):
            vid=entry['video_id'];vm=json.loads((merged/f'metadata/videos/{vid}.json').read_text())
            if jsonl(merged/f'labels/{vid}.jsonl')!=expected[vid] or sha(merged/f'labels/{vid}.jsonl')!=vm['labels_sha256']:
                raise ValueError('per-video labels differ')
            if sha(merged/f'windows/{vid}.jsonl')!=sha(ball_root/f'windows/{vid}.jsonl'):raise ValueError('temporal windows changed')
            if sha(merged/f'events/{vid}.jsonl')!=vm['events_sha256']:raise ValueError('per-video events changed')
            events=jsonl(merged/f'events/{vid}.jsonl')
            if any(e['contact_xy'] is not None for e in events):raise ValueError('contact coordinate invented')
            videos+=1
        frames+=len(rows)
    for root in (merged,ball_root):
        ref=json.loads((root/'frame_store.json').read_text())
        if root==merged:new_store=(root/ref['path']).resolve();new_id=ref['storage_id']
        else:
            if (root/ref['path']).resolve()!=new_store or ref['storage_id']!=new_id:raise ValueError('frame store changed')
    return dict(passed=True,dataset_id=meta['dataset_id'],videos=videos,frames=frames,
        ball_labels_unchanged=True,split_bytes_unchanged=True,temporal_windows_unchanged=True,
        frame_store_reused=True,supervised_frames=dict(supervised))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PROJECT/'event_annotations')
    p.add_argument('--ball-dataset',type=Path,default=DATA);p.add_argument('--output',type=Path,default=PROJECT/'data/datasets/pp_ball_events_v1')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:print(json.dumps(verify(a.output,a.ball_dataset),ensure_ascii=False,indent=2))
    else:
        result=merge(a.source,a.ball_dataset,a.output)
        checks=verify(a.output,a.ball_dataset);save(a.output/'metadata/merge_verification.json',checks)
        print(json.dumps(dict(dataset=str(a.output),status=result['status'],per_split=result['per_split'],verification=checks),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
