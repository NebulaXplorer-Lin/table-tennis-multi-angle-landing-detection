"""Native Ultralytics table training on the immutable frame store, without RGB expansion."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_CONFIG_DIR']=str(Path(__file__).resolve().parents[3]/'out/yolo_setup/table_native_config')
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[3]/'out/yolo_setup/matplotlib')

import cv2
import numpy as np
import torch

from ttnet.common import PROJECT, SRC, identity, save, save_jsonl, sha
from ttnet.data_tools.prepare_table import TableCorpus, is_thin
from ttnet.events.train import atomic_checkpoint
from ttnet.models.table_encoder import ultralytics_api, load_table_checkpoint
from ttnet.scene.table_mlp_control import now
from ttnet.scene.train_visual import evaluate

api=ultralytics_api()
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import SETTINGS
from ultralytics.utils.torch_utils import unwrap_model
import ultralytics.engine.trainer as native_trainer


class FrameStoreDataset(YOLODataset):
    """Use native YOLO geometry/HSV/Mosaic; replace only image and label loading."""
    def __init__(self, corpus, part, **kwargs):
        self.corpus=corpus;self.part=part
        self.table_rows=corpus.positives(part)
        self.path_rows={str(corpus.root/'virtual_yolo'/part/f'{r["video_id"]}_{r["frame_id"]:06d}.jpg'):r for r in self.table_rows}
        super().__init__(img_path=part,**kwargs)

    def get_img_files(self, img_path):
        return list(self.path_rows)

    def get_labels(self):
        result=[]
        for name in self.im_files:
            r=self.path_rows[name];w,h=r['resolution']['width'],r['resolution']['height'];x1,y1,x2,y2=r['table_bbox_xyxy']
            result.append(dict(im_file=name,shape=(h,w),cls=np.zeros((1,1),np.float32),
                               bboxes=np.array([[(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h]],np.float32),
                               segments=[],keypoints=None,normalized=True,bbox_format='xywh'))
        return result

    def load_image(self, i, rect_mode=True):
        if self.ims[i] is not None:
            return self.ims[i],self.im_hw0[i],self.im_hw[i]
        r=self.path_rows[self.im_files[i]]
        image=self.corpus.image(r['video_id'],r['frame_id'])[:,:,::-1].copy()  # native YOLO loads BGR
        h0,w0=image.shape[:2]
        if rect_mode:
            ratio=self.imgsz/max(h0,w0)
            if ratio!=1:
                image=cv2.resize(image,(min(math.ceil(w0*ratio),self.imgsz),min(math.ceil(h0*ratio),self.imgsz)),interpolation=cv2.INTER_LINEAR)
        elif (h0,w0)!=(self.imgsz,self.imgsz):
            image=cv2.resize(image,(self.imgsz,self.imgsz),interpolation=cv2.INTER_LINEAR)
        if self.augment:
            self.ims[i],self.im_hw0[i],self.im_hw[i]=image,(h0,w0),image.shape[:2]
            self.buffer.append(i)
            if 1<len(self.buffer)>=self.max_buffer_length:
                j=self.buffer.pop(0)
                self.ims[j],self.im_hw0[j],self.im_hw[j]=None,None,None
        return image,(h0,w0),image.shape[:2]


class TableTrainer(DetectionTrainer):
    def get_dataset(self):
        self.table_corpus=TableCorpus(Path(self.args.data))
        return dict(path=str(self.table_corpus.root),train='train',val='val',nc=1,names={0:'main_table'},channels=3)

    def build_dataset(self, img_path, mode='train', batch=None):
        return FrameStoreDataset(self.table_corpus,mode,imgsz=self.args.imgsz,augment=mode=='train',hyp=deepcopy(self.args),
                                 rect=False,batch_size=batch,stride=32,pad=0.,cache=False,data=self.data,task='detect',prefix=mode+': ')

    def preprocess_batch(self, batch):
        batch=super().preprocess_batch(batch)
        if not getattr(self,'table_examples_saved',False):
            dest=self.save_dir.parent/'augmentation_examples';dest.mkdir(exist_ok=True)
            records=[]
            for i in range(min(4,len(batch['img']))):
                image=(batch['img'][i].detach().float().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)[:,:,::-1].copy()
                boxes=batch['bboxes'][batch['batch_idx']==i].detach().float().cpu().numpy()
                for cx,cy,w,h in boxes:
                    cv2.rectangle(image,(round((cx-w/2)*image.shape[1]),round((cy-h/2)*image.shape[0])),
                                  (round((cx+w/2)*image.shape[1]),round((cy+h/2)*image.shape[0])),(0,255,0),2)
                name=f'native_batch_{i:02d}.jpg'
                if not cv2.imwrite(str(dest/name),image):
                    raise OSError(dest/name)
                records.append(dict(image=name,boxes=len(boxes)))
            save(dest/'manifest.json',dict(kind='actual_native_training_batch_after_HSV_mosaic_affine',examples=records))
            self.table_examples_saved=True
        return batch

    def final_eval(self):
        # Each epoch already has native AP validation. Preserve optimizer/EMA in last.pt;
        # the common full-frame table evaluation runs below using the saved best bundle.
        return None


def run(root, config_path, device, resume=False):
    if root.exists() and not resume:
        raise FileExistsError(root)
    cfg=json.loads(config_path.read_text())
    if cfg['schema']!='table_yolo_native_v1' or api.__version__!=cfg['ultralytics_version']:
        raise ValueError('native YOLO recipe/version mismatch')
    dataset=(PROJECT/cfg['dataset']).resolve();pretrained=(PROJECT/cfg['pretrained']).resolve()
    if sha(pretrained)!=cfg['pretrained_sha256']:
        raise ValueError('YOLO COCO pretrained checkpoint checksum mismatch')
    corpus=TableCorpus(dataset)
    code=('scene/train_yolo_native.py','scene/train_visual.py','models/table_encoder.py','data_tools/prepare_table.py','data_tools/geometry.py','data_tools/frame_store.py')
    native_files=('cfg/default.yaml','engine/trainer.py','models/yolo/detect/train.py','data/base.py','data/dataset.py','data/augment.py','utils/loss.py')
    dependencies=dict(table_annotations=corpus.identity,pretrained_sha256=sha(pretrained),global_sha256=None,
                      implementation={p:sha(SRC/p) for p in code},ultralytics={p:sha(Path(api.__file__).parent/p) for p in native_files})
    if resume and (json.loads((root/'recipe.json').read_text())!=cfg or json.loads((root/'dependencies.json').read_text())!=dependencies):
        raise ValueError('YOLO resume recipe, dataset or code changed')
    root.mkdir(parents=True,exist_ok=True);save(root/'recipe.json',cfg);save(root/'dependencies.json',dependencies)
    for part in ('train','val'):
        rows=corpus.positives(part)
        save_jsonl(root/f'dataset/{part}.jsonl',[dict(video_id=r['video_id'],frame_id=r['frame_id'],thin=is_thin(r),bbox_xyxy=r['table_bbox_xyxy'],resolution=r['resolution']) for r in rows])
    save(root/'dataset/manifest.json',dict(stats=corpus.audit(),table_annotations=corpus.identity,
         storage='read existing FrameStore; no expanded RGB image dataset',unknown='excluded from native train/AP; retained in final full-frame inference'))
    # Existing CUDA AMP evidence from this session avoids native AMP's extra model download/check.
    native_trainer.check_amp=lambda model:True
    SETTINGS.update({key:False for key in ('sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','tensorboard','wandb') if key in SETTINGS})
    overrides=dict(cfg['train'],task='detect',mode='train',model=str(pretrained),data=str(dataset),device=device,
                   project=str(root),name='native',exist_ok=True,verbose=False)
    if resume:
        checkpoint=root/'native/weights/last.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        overrides['resume']=str(checkpoint)
    trainer=TableTrainer(overrides=overrides)
    eval_cfg=dict(schema='table_visual_recipe_v1',source='independent_yolo',dataset=str(dataset),pretrained=str(pretrained),
                  global_checkpoint=None,input=[640,640],channels=128,seed=cfg['train']['seed'],amp=cfg['train']['amp'],
                  confidence=cfg['evaluation']['confidence'],nms_iou=cfg['evaluation']['nms_iou'],micro_batch=cfg['evaluation']['micro_batch'],
                  workers=cfg['train']['workers'],augmentation={},letterbox_value=114,selection='native_mAP50_95',native_recipe=cfg)
    save(root/'config.json',eval_cfg)
    started=json.loads((root/'task.json').read_text()).get('started',now()) if resume and (root/'task.json').exists() else now()
    status=dict(status='preparing',started=started,run=str(root),native_run=str(root/'native'),selection='native_mAP50_95',dataset_stats=corpus.audit(),
                source='YOLO11s COCO pretrained; native trainer/loss/HSV/Mosaic',weight_sha256=sha(pretrained))
    history=json.loads((root/'history.json').read_text()) if resume and (root/'history.json').exists() else []
    best_export=max([r['fitness'] for r in history],default=-1.) if (root/'best.pt').exists() else -1.
    def update(**values):
        status.update(values,updated=now());save(root/'task.json',status);save(root/'run.json',status)
    def epoch_start(t):
        t.table_step=0;update(status='training',epoch=t.epoch+1,steps=len(t.train_loader))
    def batch_end(t):
        t.table_step+=1
        if t.table_step==1 or t.table_step%50==0:
            update(status='training',epoch=t.epoch+1,step=t.table_step,steps=len(t.train_loader),
                   loss=[float(v) for v in t.tloss.detach().cpu().flatten()],optimizer_updates=int(t.ema.updates))
    def epoch_end(t):
        update(status='validating',epoch=t.epoch+1)
    def fit_end(t):
        nonlocal best_export
        values={k:float(v) for k,v in t.metrics.items()};fit=float(t.fitness)
        row=dict(epoch=t.epoch+1,fitness=fit,metrics=values,optimizer_updates=int(t.ema.updates))
        if history and history[-1]['epoch']==row['epoch']:
            history[-1]=row
        else:
            history.append(row)
        save(root/'history.json',history)
        if fit>best_export:
            model=unwrap_model(t.ema.ema)
            head=model.model[-1];channels=head.cv2[0][0].conv.in_channels
            if channels!=128:
                raise ValueError('unexpected YOLO11s P3 channel contract')
            state={f'detector.{k}':(v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()).clone() for k,v in model.state_dict().items()}
            atomic_checkpoint(root/'best.pt',dict(schema='table_visual_v1',config=eval_cfg,architecture=deepcopy(model.yaml),model=state,
                              global_sha256=None,dependencies=dependencies,epoch=t.epoch+1,native_fitness=fit,native_metrics=values))
            best_export=fit
        update(status='training',epoch=t.epoch+1,latest_native_metrics=values,best_fitness=best_export)
        print('table native epoch',json.dumps(row),flush=True)
    trainer.add_callback('on_train_epoch_start',epoch_start);trainer.add_callback('on_train_batch_end',batch_end)
    trainer.add_callback('on_train_epoch_end',epoch_end);trainer.add_callback('on_fit_epoch_end',fit_end)
    update();trainer.train()
    update(status='evaluating_best')
    model,state=load_table_checkpoint(root/'best.pt',device='cuda:0' if str(device)=='0' else device)
    val=evaluate(model,None,corpus,eval_cfg,next(model.parameters()).device,root/'evaluation',overlay_mode='worst_per_video')
    train=evaluate(model,None,corpus,eval_cfg,next(model.parameters()).device,part='train')
    save(root/'metrics.json',dict(train=train,val=val,best_epoch=state['epoch'],native_fitness=state['native_fitness'],native_metrics=state['native_metrics'],
                                checkpoint_sha256=sha(root/'best.pt'),table_annotations=corpus.identity))
    comparisons={}
    for name,path in [('B 4x4 MLP','20261104-142741_B_legacy_mlp_control/B'),('B thin16x4 MLP','20261104-145747_B_mlp_thin'),('independent MLP color','20261104-160636_independent_mlp_color')]:
        m=json.loads((PROJECT/'out/experiments'/path/'metrics.json').read_text())['val'];b=m['by_bucket']
        comparisons[name]=dict(frame=m.get('frame_mean_iou',m.get('mean_iou')),video=m['video_mean_iou'],thin=b.get('thin',b.get('thin:all'))['mean_iou'],iou50=m['iou50'])
    comparisons['YOLO11s native']=dict(frame=val['mean_iou'],video=val['video_mean_iou'],thin=val['by_bucket']['thin:all']['mean_iou'],iou50=val['iou50'])
    save(root/'comparison.json',comparisons)
    lines=['# Native YOLO11s table recipe','','Best by native mAP50-95; common table IoU below. Combined recipe comparison.','',
           '| Recipe | Val frame IoU | Val video IoU | Thin IoU | IoU>=0.5 fraction |','|---|---:|---:|---:|---:|']
    for name,m in comparisons.items():
        lines.append(f'| {name} | {m["frame"]:.4f} | {m["video"]:.4f} | {m["thin"]:.4f} | {m["iou50"]:.4f} |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    update(status='completed',finished=now(),best_epoch=state['epoch'],metrics=str(root/'metrics.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=PROJECT/'configs/table/yolo11s_native.json')
    p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='0');p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1);existed=args.run.exists()
    try:
        run(args.run,args.config,args.device,args.resume)
    except BaseException as error:
        if args.run.exists() and (not existed or args.resume):
            save(args.run/'task.json',dict(status='failed',error=repr(error),updated=now()))
        raise
