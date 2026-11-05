"""Prepare the two table routes and check real forward paths without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ttnet.common import PROJECT, identity, save, sha
from ttnet.configuration import load_config
from ttnet.data_tools.prepare_table import TableCorpus, TableDataset, collate_table, sample_epoch
from ttnet.models.table_encoder import select_table, spatial_visual
from ttnet.scene.train_visual import construct, forward, load_recipe, parameter_groups
from ttnet.scene import train_light
from ttnet.models.table_light import NativeTableEncoder


def check_route(cfg, corpus):
    """CPU eval forward/loss/ROI only. No optimizer, backward, BN update or checkpoint."""
    torch.set_num_threads(2)
    if cfg['source']=='independent_cnn':
        model,global_model=NativeTableEncoder(cfg['input']),None
    else:
        model,global_model = construct(cfg,torch.device('cpu'))
    model.eval()
    r = next(r for r in corpus.positives('train') if (r['table_bbox_xyxy'][3]-r['table_bbox_xyxy'][1])/r['resolution']['height'] < .06)
    batch = collate_table([TableDataset(corpus,[r],cfg)[0]])
    with torch.inference_mode():
        prediction,spatial = forward(model,global_model,batch,torch.device('cpu'))
        loss,parts = model.loss(prediction,batch['target'])
        kwargs = dict(confidence=cfg['confidence'],nms_iou=cfg['nms_iou'])
        if global_model is not None:
            kwargs['input_size'] = cfg['input']
        candidates = model.candidates(prediction,**kwargs)
        table = select_table(candidates[0],batch['transform'][0])
        visual = spatial_visual(spatial[0],table,batch['transform'][0])
    if not torch.isfinite(loss) or not torch.isfinite(spatial).all() or spatial.shape[1] != cfg['channels']:
        raise ValueError('invalid table forward contract')
    return dict(source=cfg['source'],frame=[r['video_id'],r['frame_id']],input=list(batch['images'].shape),
                spatial=list(spatial.shape),roi=list(visual['roi'].shape),context=list(visual['context'].shape),
                initial_loss=float(loss),loss_parts={k:float(v) for k,v in parts.items()},prediction=table,
                parameter_groups=[dict(name=g['name'],lr=g['lr'],weight_decay=g['weight_decay'],parameters=sum(p.numel() for p in g['params'])) for g in (train_light.parameter_groups(model,cfg) if cfg['source']=='independent_cnn' else parameter_groups(model,cfg))],
                note='Untrained table head; loss/prediction is an interface check, not accuracy evidence.')


def prepare(output, check_models=False):
    if output.exists():
        raise FileExistsError(output)
    corpus = TableCorpus(PROJECT/'data/datasets/pp_table_reannotated_v1')
    recipes = dict(independent_cnn=train_light.load_recipe(PROJECT/'configs/table/independent_cnn.json'),
                   shared_B=load_recipe(PROJECT/'configs/table/shared_B.json'))
    configs = {p.stem:load_config(p)[0] for p in sorted((PROJECT/'configs/experiments').glob('event_table_*.json'))}
    b = configs['event_table_B_F0']
    weights = {key:dict(path=b[key],sha256=sha(b[key])) for key in ('global_checkpoint','local_checkpoint')}
    selected = torch.load(b['local_checkpoint'],map_location='cpu',weights_only=False)
    if selected['global_checkpoint_sha256'] != weights['global_checkpoint']['sha256']:
        raise ValueError('B Global/Local are not a trained pair')
    report = dict(status='prepared_no_training',table_annotations=corpus.identity,stats=corpus.audit(),
                  B_locator_pair=weights,event_configs=list(configs),recipes=recipes,
                  checks=[],training_started=False,optimizer_updates=0,
                  remaining=['Train and evaluate the two native/shared table routes; historical YOLO preparation is retired.',
                             'Lock the final locator evaluation and table checkpoints before generating full event feature caches.',
                             'Future joint fine-tuning with table auxiliary loss is intentionally rejected by the current frozen-event path.'])
    if check_models:
        report['checks'] = [check_route(cfg,corpus) for cfg in recipes.values()]
    output.mkdir(parents=True)
    # Generate the same 80 epoch centre-frame schedule for both routes, not RGB caches.
    command = [sys.executable,'-m','ttnet.data_tools.prepare_table','--dataset',str(corpus.root),'--output',str(output/'sampling')]
    subprocess.run(command,cwd=PROJECT,check=True,capture_output=True,text=True)
    report['sampling_manifest_sha256'] = sha(output/'sampling/manifest.json')
    report['pretrained'] = {name:dict(path=cfg['pretrained'],available=Path(cfg['pretrained']).is_file(),
                                    sha256=sha(cfg['pretrained']) if Path(cfg['pretrained']).is_file() else None)
                            for name,cfg in recipes.items() if cfg['pretrained']}
    save(output/'preparation.json',report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-models',action='store_true')
    args = p.parse_args()
    result = prepare(args.output,args.check_models)
    print(json.dumps(dict(status=result['status'],stats=result['stats'],checks=result['checks']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
