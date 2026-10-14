from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ttnet.common import DATA,identity,jsonl,now,save,sha
from ttnet.data_tools.geometry import crop_image,crop_transform,letterbox
from ttnet.training.dataset import normalize,read_rgb
from ttnet.training.scales import input_size
from .inference import load_model


def export_one(checkpoint,dataset,dest,stage):
    torch.set_num_threads(4)
    model,info=load_model(checkpoint,torch.device('cpu'))
    if info['dataset_id']!=json.loads((dataset/'metadata/dataset.json').read_text())['dataset_id']:raise ValueError('export dataset identity differs')
    w,h=input_size(info.get('config'),stage)
    model.export_pool_kernel=(h//4,w//4)
    dummy=torch.zeros((1,3,h,w))
    path=dest/f'{stage}.onnx'
    with torch.inference_mode():
        torch.onnx.export(model,dummy,path,input_names=['input'],output_names=['heatmap','features'],
                          opset_version=17,dynamic_axes={'input':{0:'batch'},'heatmap':{0:'batch'},'features':{0:'batch'}},dynamo=False)
    graph=onnx.load(path);onnx.checker.check_model(graph,full_check=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    rows=jsonl(dataset/'labels/val.jsonl')
    chosen=[];seen=set()
    for r in rows:
        key=(r['resolution']['width'],r['resolution']['height'],r['label_status'])
        if key not in seen:chosen.append(r);seen.add(key)
    errors=[];inputs=[np.zeros((1,3,h,w),np.float32)]
    for r in chosen[:8]:
        if stage=='global':rgb,_=letterbox(read_rgb(dataset/r['frame_path']),w,h)
        else:
            raw=read_rgb(dataset/r['frame_path']);x,y=r['ball_centers'][0] if r['ball_centers'] else (0,0)
            rgb=crop_image(raw,crop_transform(x,y,r['resolution']['width'],r['resolution']['height'],w,
                                               info.get('config',{}).get('local_crop_side_1080',256)))
        inputs.append(normalize(rgb)[None])
    inputs.append(np.concatenate([inputs[-1],inputs[-1]],axis=0))
    for x in inputs:
        with torch.inference_mode():expected=[v.numpy() for v in model(torch.from_numpy(x))]
        actual=session.run(None,{'input':x})
        deltas=[float(np.max(np.abs(a-b))) for a,b in zip(actual,expected)]
        if not all(np.allclose(a,b,atol=2e-4,rtol=2e-4) for a,b in zip(actual,expected)):
            raise RuntimeError(f'ONNX parity failed: {deltas}')
        errors.append(dict(batch=len(x),heatmap_max_abs=deltas[0],features_max_abs=deltas[1]))
    return dict(stage=stage,checkpoint_sha256=sha(checkpoint),onnx_sha256=sha(path),opset=17,
                input_shape=[None,3,h,w],output_shapes=[[None,1,h//2,w//2],[None,128]],threshold=info['threshold'],
                parity=dict(backend='ONNX Runtime CPU FP32 vs PyTorch CPU FP32',atol=2e-4,rtol=2e-4,cases=errors,passed=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,default=DATA)
    p.add_argument('--global-checkpoint',type=Path,required=True);p.add_argument('--local-checkpoint',type=Path)
    args=p.parse_args();dest=args.run/'exports';dest.mkdir(exist_ok=True)
    records=[export_one(args.global_checkpoint,args.dataset,dest,'global')]
    if args.local_checkpoint:records.append(export_one(args.local_checkpoint,args.dataset,dest,'local'))
    save(dest/'manifest.json',dict(exported=now(),dataset_id=json.loads((args.dataset/'metadata/dataset.json').read_text())['dataset_id'],
                                  models=records,normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225],rgb=True),
                                  note='ORT parity checked; target competition TensorRT compatibility not yet measured'))
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
