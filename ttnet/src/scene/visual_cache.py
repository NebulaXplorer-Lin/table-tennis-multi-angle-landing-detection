"""Table descriptor contract shared by online frames, NPZ caches and event windows."""
from __future__ import annotations

import numpy as np
import torch


VISUAL_KEYS = ('roi','context','roi_valid','context_valid')


def table_image_batch(images, device):
    """Match online CHW.copy() layout before CUDA/AMP detector execution.

    Stacking transposed HWC views preserves channel-last physical strides.
    Equal pixel values with that layout can select different FP16 kernels.
    """
    array = np.ascontiguousarray(np.stack([image.transpose(2,0,1) for image in images]))
    return torch.from_numpy(array).to(device).float()/255.


def trajectory_records(records):
    """Log prediction metadata; spatial tensors belong in the NPZ feature cache."""
    for record in records:
        item = dict(record)
        if item.get('table') is not None:
            item['table'] = {k:v for k,v in item['table'].items() if k!='visual'}
        yield item


def validate_frontend(spec):
    if set(spec) != {'source','checkpoint','channels','input','confidence','nms_iou','roi'}:
        raise ValueError('invalid table_frontend fields')
    if spec['source'] not in ('independent_cnn','independent_yolo','shared_global') or not spec['checkpoint']:
        raise ValueError('table front end needs a source and trained checkpoint')
    if type(spec['channels']) is not int or spec['channels'] < 1:
        raise ValueError('invalid table channels')
    if len(spec['input'])!=2 or any(type(i) is not int or i<32 or i%32 for i in spec['input']):
        raise ValueError('invalid table input size')
    if not 0<=spec['confidence']<=1 or not 0<spec['nms_iou']<=1:
        raise ValueError('invalid table thresholds')
    if set(spec['roi'])!={'width_scale','height_scale','min_height'} or any(v<=0 for v in spec['roi'].values()):
        raise ValueError('invalid ROI configuration')


def window_visual(ids, records, channels, device):
    values = {key:[] for key in VISUAL_KEYS}
    for fid in ids:
        record = records.get(fid,{}) if fid is not None else {}
        payload = (record.get('table') or {}).get('visual',{})
        for key,size in (('roi',channels*16),('context',channels)):
            value = payload.get(key)
            tensor = torch.zeros(size,device=device) if value is None else torch.as_tensor(value,device=device,dtype=torch.float32)
            if tensor.shape!=(size,):
                raise ValueError(f'invalid table {key} descriptor shape')
            values[key].append(tensor)
        for key in ('roi_valid','context_valid'):
            values[key].append(bool(payload.get(key,False)) and record.get('frame_valid',True))
    return {key:torch.stack(value)[None] if key in ('roi','context') else torch.tensor([value],device=device,dtype=torch.bool)
            for key,value in values.items()}


def serialize_visual(records, channels):
    """Move tensors into NPZ fields and return JSON-compatible copied records."""
    fields = window_visual(list(range(len(records))),dict(enumerate(records)),channels,'cpu')
    arrays = {'table_'+k:v[0].detach().cpu().numpy() for k,v in fields.items()}
    return arrays,list(trajectory_records(records))


def restore_visual(archive, ids, records, channels):
    expected = dict(roi=(len(ids),channels*16),context=(len(ids),channels),
                    roi_valid=(len(ids),),context_valid=(len(ids),))
    # NpzFile.__getitem__ decompresses the entire member on every access.
    # Read each array once, before indexing frames, to keep restoration linear.
    arrays = {}
    available = set(archive)
    for key,shape in expected.items():
        if 'table_'+key not in available:
            raise ValueError('missing or malformed table cache field: '+key)
        arrays[key] = archive['table_'+key]
        if arrays[key].shape!=shape:
            raise ValueError('missing or malformed table cache field: '+key)
    for index,fid in enumerate(ids):
        table = records[int(fid)].get('table')
        if table is None:
            table = dict(valid=False,bbox_xyxy=None,score=0.,age_sec=0.,source='predicted')
            records[int(fid)]['table'] = table
        table['visual'] = {key:torch.from_numpy(arrays[key][index].copy()).float() if key in ('roi','context')
                           else bool(arrays[key][index]) for key in VISUAL_KEYS}


def batch_windows(inputs, device):
    """Batch legacy 3-tuples and new structured table 4-tuples identically."""
    def combine(values):
        if isinstance(values[0],dict):
            return {key:combine([v[key] for v in values]) for key in values[0]}
        return torch.cat(values).to(device,non_blocking=True)
    return [combine([item[k] for item in inputs]) for k in range(len(inputs[0]))]
