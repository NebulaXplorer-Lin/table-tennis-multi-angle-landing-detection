"""Small real-frame acceptance of batched YOLO caches against online inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from ttnet.common import PROJECT, save, sha
from ttnet.configuration import load_config
from ttnet.events.continuous import attach_pp
from ttnet.events.data import EventCorpus
from ttnet.events.engine import EventSystem
from ttnet.scene.augment_event_cache import infer_tables
from ttnet.scene.visual_cache import restore_visual, serialize_visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=8)
    args = parser.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR',str(PROJECT/'out/yolo_setup/event_table_config'))
    os.environ.setdefault('YOLO_OFFLINE','true')
    torch.set_num_threads(2)
    cfg,_ = load_config(args.config)
    corpus = EventCorpus(cfg['dataset']);attach_pp(corpus,cfg['pp_annotations'])
    system = EventSystem(cfg,torch.device(args.device));system.configure('frozen',False)
    cases = [('pp_02_001',0),('pp_02_001',1),('pp_22_001',0),('pp_22_001',1),
             ('pp_22_004',0),('pp_22_004',76),('pp_22_004',130),('pp_22_004',304),
             ('pp_12_001',66),('pp_03_004',90),('pp_21_004',63)]
    class MixedFrames:
        videos = dict(acceptance=dict(frames={i:{} for i in range(len(cases))}))
        def image(self, vid, fid):
            source_video, source_frame = cases[fid]
            return corpus.image(source_video,source_frame)
    mixed = MixedFrames()
    groups = dict(acceptance=list(range(len(cases))))
    # Cases include normal boxes, group22 misses and validation scores near 0.1.
    rows = []
    tolerances = dict(feature_atol=2e-3,feature_rtol=2e-3,box_atol_pixels=.1,score_atol=1e-3)
    for vid,ids in groups.items():
        captured = []
        original = system.table_encoder.candidates
        def candidates(*values,**kwargs):
            result = original(*values,**kwargs)
            captured.extend([r.detach().float().cpu() for r in result])
            return result
        with patch.object(system.table_encoder,'candidates',side_effect=candidates):
            batched = infer_tables(system,mixed,vid,ids,args.batch)
        batch_candidates = captured
        captured = []
        records = {fid:dict(frame_id=fid) for fid in ids}
        with torch.inference_mode(),torch.autocast(device_type=system.device.type,
                enabled=cfg['amp'] and system.device.type=='cuda'), \
             patch.object(system.table_encoder,'candidates',side_effect=candidates):
            system.attach_table_visual(mixed,vid,ids,records)
        online_candidates = captured
        arrays,clean = serialize_visual([dict(frame_id=fid,table=batched[fid]) for fid in ids],128)
        restored = {r['frame_id']:r for r in clean};restore_visual(arrays,ids,restored,128)
        for index,fid in enumerate(ids):
            a,b = batched[fid],records[fid]['table']
            features = {}
            for key in ('roi','context'):
                first,second = a['visual'][key],b['visual'][key]
                features[key] = dict(max_abs=float((first-second).abs().max()),
                    mean_abs=float((first-second).abs().mean()),
                    normalized_l2=float((first-second).norm()/second.norm().clamp_min(1e-8)),
                    allclose_2e_4=bool(torch.allclose(first,second,atol=2e-4,rtol=2e-4)),
                    within_tolerance=bool(torch.allclose(first,second,atol=2e-3,rtol=2e-3)))
                torch.testing.assert_close(first,restored[fid]['table']['visual'][key],atol=0,rtol=0)
            ac,bc = batch_candidates[index],online_candidates[index]
            same_count = len(ac)==len(bc)
            valid_equal = a['valid']==b['valid']
            box_error = max(abs(x-y) for x,y in zip(a['bbox_xyxy'],b['bbox_xyxy'])) if a['valid'] and b['valid'] else None
            score_error = abs(a['score']-b['score'])
            source_video, source_frame = cases[fid]
            candidate_box_error = float((ac[:,:4]-bc[:,:4]).abs().max()) if same_count and len(ac) else None
            candidate_score_error = float((ac[:,4]-bc[:,4]).abs().max()) if same_count and len(ac) else None
            row = dict(video=source_video,frame_id=source_frame,batch=args.batch,batched_candidate_count=len(ac),online_candidate_count=len(bc),
                same_candidate_count=same_count,valid_equal=valid_equal,batched_valid=a['valid'],online_valid=b['valid'],
                batched_score=a['score'],online_score=b['score'],score_abs_error=score_error,box_max_abs_pixels=box_error,
                candidate_box_max_abs_input_pixels=candidate_box_error,candidate_score_max_abs=candidate_score_error,
                features=features,serialization_exact=True)
            row['passed'] = bool(same_count and valid_equal and score_error<=tolerances['score_atol'] and
                (box_error is None or box_error<=tolerances['box_atol_pixels']) and all(v['within_tolerance'] for v in features.values()))
            row['single_frame_exact'] = bool(same_count and valid_equal and score_error==0 and
                (box_error is None or box_error==0) and all(v['max_abs']==0 for v in features.values()))
            if args.batch==1:
                row['passed'] = row['passed'] and row['single_frame_exact']
            rows.append(row)
    result = dict(passed=all(r['passed'] for r in rows),device=str(system.device),amp=cfg['amp'],
        checkpoint_sha256=sha(cfg['table_frontend']['checkpoint']),config=str(args.config),tolerances=tolerances,
        batch=args.batch,cases_count=len(cases),
        note='Real validation frames, including near-threshold group22 and observed online/cache discrepancy cases; numerical acceptance only, no accuracy tuning or timing.',cases=rows)
    save(args.output,result)
    print(json.dumps(result),flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
