"""Add table predictions to a verified frozen rule cache without rerunning Local."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ttnet.common import DATA,EVENT_DATA,save,save_jsonl,jsonl,sha
from ttnet.events.data import EventCorpus
from ttnet.events.continuous import attach_pp
from ttnet.events.engine import EventSystem,cache_identity,feature_cache_path
from ttnet.data_tools.geometry import letterbox,letterbox_transform
from ttnet.training.dataset import normalize
from ttnet.scene.table import annotations


def augment(cfg,base,table_features,device):
    corpus=EventCorpus(EVENT_DATA);attach_pp(corpus,cfg['pp_annotations'])
    system=EventSystem(cfg,device);system.configure('frozen',False)
    previous=json.loads((base/'identity.json').read_text());expected=cache_identity(system,corpus)
    for key in ('dataset_id','global_sha256','local_sha256','tracker','config','extraction_batch','sampling'):
        if previous[key]!=expected[key]:raise ValueError(f'base cache mismatch: {key}')
    dest=feature_cache_path(system,corpus);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=expected:raise ValueError('target cache mismatch')
    save(dest/'identity.json',expected)
    oracle={**annotations('train')[0],**annotations('val')[0]} if cfg.get('table_source')=='oracle' else {}
    derivations=[]
    for part in ('train','val'):
        for vid in corpus.split[part]:
            src=base/f'{vid}.npz';recsrc=base/f'{vid}.jsonl'
            derivations.append(dict(video=vid,feature_sha256=sha(src),records_sha256=sha(recsrc)))
            if (dest/f'{vid}.npz').exists() and (dest/f'{vid}.jsonl').exists():continue
            records=jsonl(recsrc);meta=corpus.videos[vid]['meta'];frames=corpus.videos[vid]['frames'];resolution=dict(width=meta['width'],height=meta['height'])
            native_vid=vid.removeprefix('pp_')
            pp=table_features/f'{native_vid}.npz'
            if not cfg.get('table_checkpoint'):
                for rec in records:rec['table']=None
            elif cfg.get('table_source')=='oracle':
                for rec in records:
                    b=oracle.get((native_vid,rec['frame_id']));rec['table']=dict(bbox_xyxy=b,valid=True,source='oracle') if b else None
            elif pp.exists():
                with np.load(pp) as z:spatial=torch.from_numpy(z['features']).to(device);ids=z['frame_ids'].tolist()
                with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    boxes=torch.cat([system.table_head(s) for s in spatial.split(128)]).float().cpu().numpy()
                from ttnet.scene.table import decode_table
                t=letterbox_transform(meta['width'],meta['height'],*cfg['global_input'])
                mapping={i:decode_table(b,t,resolution) for i,b in zip(ids,boxes)}
                for rec in records:rec['table']=mapping[rec['frame_id']]
            else:
                # Supply true neighboring frames at sparse-anchor boundaries, matching
                # the existing chronological source extraction (never zero a real neighbor).
                missing={i for rec in records for i in (rec['frame_id']-1,rec['frame_id']+1) if 0<=i<meta['n_frames'] and i not in frames}
                extra={}
                if missing:
                    import av
                    from fractions import Fraction
                    from ttnet.events.prepare import nearest_index
                    with av.open(str(meta['source'])) as container:
                        stream=container.streams.video[0];rate=Fraction(stream.average_rate);stream.thread_type='AUTO'
                        wanted={nearest_index(Fraction(i)*rate/Fraction(str(meta['fps']))):i for i in missing}
                        first=None
                        for sid,frame in enumerate(container.decode(stream)):
                            ts=float(frame.pts*frame.time_base)
                            if first is None:first=ts
                            if sid in wanted:
                                fid=wanted[sid];extra[fid]=frame.to_ndarray(format='rgb24');frames[fid]=dict(timestamp_sec=ts-first,frame_valid=True)
                            if sid>=max(wanted):break
                    if len(extra)!=len(missing):raise ValueError('missing source context')
                with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    for start in range(0,len(records),16):
                        batch=records[start:start+16];images=[];valid=[];offsets=[];transforms=[]
                        for rec in batch:
                            fid=rec['frame_id'];ns=[i if i in frames and frames[i].get('frame_valid',True) else None for i in (fid-1,fid,fid+1)]
                            # Base cache retains sampled NPY pixels; same precedence as streaming extraction.
                            boxes=[letterbox(extra[i] if i in extra else corpus.image(vid,i),*cfg['global_input']) if i is not None else (np.zeros((cfg['global_input'][1],cfg['global_input'][0],3),np.uint8),None) for i in ns]
                            if ns[1] is None:raise ValueError('invalid center in table cache')
                            images.append(np.stack([normalize(im) for im,t in boxes]));transforms.append(boxes[1][1]);valid.append([i is not None for i in ns]);offsets.append([frames[i]['timestamp_sec']-frames[fid]['timestamp_sec'] if i is not None else 0. for i in ns])
                        system._run(system.global_model,torch.from_numpy(np.stack(images)).to(device),torch.tensor(valid,device=device),torch.tensor(offsets,device=device))
                        for i,rec in enumerate(batch):rec['table']=system.table_record(transforms[i],resolution,i)
            # Copy data, never mutate or hardlink existing cache records.
            import shutil
            tmp=dest/f'{vid}.tmp.npz';shutil.copyfile(src,tmp);tmp.replace(dest/f'{vid}.npz');save_jsonl(dest/f'{vid}.jsonl',records)
            print(f'table-augmented {vid}: {len(records)}',flush=True)
    save(dest/'derivation.json',dict(base=str(base),base_identity=previous,sources=derivations,note='Frozen locator decode unchanged; added independent table head. Full-video parity checked separately.'))
    return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--table-features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4)
    cfg=json.loads(args.config.read_text());dest=augment(cfg,args.base,args.table_features,torch.device('cuda'));save(args.output,dict(cache=str(dest)))

if __name__=='__main__':main()
