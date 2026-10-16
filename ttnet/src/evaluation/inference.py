from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ttnet.data_tools.geometry import crop_point, map_point
from ttnet.training.dataset import BallDataset
from ttnet.models.localizer import BallLocalizer
from ttnet.models.build import build_localizer
from .metrics import accept, decode_heatmap
from ttnet.temporal import move_inputs,temporal_mode


def load_model(path, device):
    checkpoint = torch.load(path,map_location="cpu",weights_only=False)
    model = build_localizer(checkpoint.get("config")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model,checkpoint


@torch.inference_mode()
def predict(model, dataset, split, device, stage="global", predictions=None, batch=8, workers=4, amp=False, features=False,
            input_size=None,crop_side_at_1080=256,input_cache=None,window_half_span_sec=1/30,local_cache=None):
    ds=BallDataset(dataset,split,stage,predictions=predictions,all_frames=True,input_size=input_size,crop_side_at_1080=crop_side_at_1080,
                   temporal_mode=getattr(model,"temporal_mode","single"),input_cache=input_cache,window_half_span_sec=window_half_span_sec,local_cache=local_cache)
    loader=DataLoader(ds,batch_size=batch,num_workers=workers,shuffle=False,pin_memory=device.type=="cuda")
    results,descriptors=[],[]
    model.eval()
    for x,_,_,indices in loader:
        with torch.autocast(device_type=device.type,enabled=amp and device.type=="cuda"):
            heat,descriptor=model(*move_inputs(x,device))
        heat=heat.float().cpu().numpy()
        if features:
            descriptors.extend(descriptor.float().cpu().numpy())
        for k,index in enumerate(indices.tolist()):
            row=ds.rows[index]
            x,y,conf=decode_heatmap(heat[k])
            transform=None
            if stage=="global":
                x,y=map_point(x,y,ds.transform_for_row(row),inverse=True)
            else:
                g=(predictions or {}).get((row["video_id"],row["frame_id"]))
                if g is None or not g["valid"]:
                    results.append(None)
                    if features:
                        descriptors[len(results)-1]=np.zeros(128,np.float32)
                    continue
                transform=ds.transform_for_row(row)
                x,y=crop_point(x,y,transform,inverse=True)
            results.append(dict(video_id=row["video_id"],frame_id=row["frame_id"],timestamp_sec=row["timestamp_sec"],
                                x=x,y=y,confidence=conf,source=stage,crop_transform=transform))
    return ds.rows,results,np.asarray(descriptors,np.float32) if features else None


def prediction_map(rows, candidates, threshold):
    return {(r["video_id"],r["frame_id"]):accept(p,threshold,r["resolution"]) for r,p in zip(rows,candidates)}
