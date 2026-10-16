"""Derive training labels from the user's confirmation of exhaustive Bounce tags.

Original coarse labels stay immutable. Contact coordinates are explicitly marked
as ball-position GT references, never as newly drawn contact annotations.
"""
from collections import defaultdict
import json
from pathlib import Path

from ttnet.common import DATA, PROJECT, identity, jsonl, save, sha

# Coarse camera orientation buckets reviewed from the saved group contact sheet.
# These are visual metadata, not calibrated camera angles.
VIEWS = {
    'front': '00 01 06 07 08 12 14 15 16 18 19 23 24'.split(),
    'diagonal': '03 09 11 17 20 21'.split(),
    'side': '02 04 05 10 13 22'.split(),
}


def prepare(output, merged=PROJECT/'data/datasets/pp_ball_events_v1'):
    output=Path(output);merged=Path(merged)
    if output.exists():raise FileExistsError(output)
    meta=json.loads((DATA/'metadata/dataset.json').read_text())
    mm=json.loads((merged/'metadata/dataset.json').read_text())
    source_audit=json.loads((merged/'metadata/event_merge.json').read_text())
    empty={r['video_id'] for r in source_audit['warnings'] if r['warning']=='empty_event_export_all_frames_unknown'}
    view={g:v for v,groups in VIEWS.items() for g in groups}
    videos=[]
    for part in ('train','val'):
        labels=defaultdict(list)
        for row in jsonl(merged/f'labels/{part}.jsonl'):labels[row['video_id']].append(row)
        for entry in jsonl(DATA/f'splits/{part}.jsonl'):
            vid=entry['video_id'];rows=labels[vid];n=len(rows);events=[];ignored=[]
            for row in rows:
                ann=row['event_annotation'];fid=row['frame_id']
                if 'ignore' in ann['kinds'] or ann['status'][0]=='conflict':ignored.append([fid,fid]);continue
                if ann['status'][0]!='positive':continue
                xy=row['ball_centers'][0] if row['label_status']=='positive' and row['ball_centers'] else None
                events.append(dict(kind='bounce',frame_id=fid,x=xy[0] if xy else None,y=xy[1] if xy else None,
                    coordinate_source='ball_position_gt_at_event_frame' if xy else 'missing_ball_position_gt',
                    spatial_valid=xy is not None,timing_quality='coarse_manual'))
            if vid in empty:events=[];ignored=[[0,n-1]]
            videos.append(dict(video_id=vid,split=part,match_group=entry['group'],view=view[entry['group']],
                view_source='agent_visual_orientation_review',source=entry['video'],source_sha256=entry['video_sha256'],
                status='reviewed',reviewed_by='user_confirmation_2026-10-16',net_reviewed=False,
                reviewed_intervals=[[0,n-1]] if vid not in empty else [],unknown_intervals=ignored,events=events,
                note='User confirmed Bounce exhaustive; racket/floor negative tags may be missing. Empty export remains unknown.'))
    document=dict(schema=1,version='pp_bounce_confirmed_v1',ball_dataset_id=meta['dataset_id'],
        ball_split_hashes={p:sha(DATA/f'splits/{p}.jsonl') for p in ('train','val')},videos=videos,
        confirmation=dict(date='2026-10-16',statement='触台已标全，主要漏了挥拍／落地等负样本标签',
            coarse_dataset_id=mm['dataset_id'],unmarked_bounce='negative outside ignore and empty exports',
            net='masked',spatial_metric='event-frame original ball GT reference; missing coordinates excluded only from spatial scoring',
            view_review='out/results/20261016_training_preflight/camera_groups.jpg'))
    save(output,document);return document


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=prepare(a.output);print(identity(d))
