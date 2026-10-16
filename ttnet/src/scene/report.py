"""Table pred/label overlays, covering every labeled validation frame."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from ttnet.common import DATA,save
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.data_tools.geometry import letterbox_transform
from ttnet.scene.table import decode_table,annotations


def render(run):
    corpus=BallCorpus(DATA,'val');labels,_=annotations('val');rows=json.loads((run/'val_predictions.json').read_text());by={}
    for r in rows:by.setdefault(r['video_id'],{})[r['frame_id']]=r
    dest=run/'overlays';dest.mkdir(exist_ok=True);errors=[]
    for vid,frames in by.items():
        meta=corpus.videos[vid]['meta'];t=letterbox_transform(meta['width'],meta['height']);size=(960,round(960*meta['height']/meta['width']));scale=960/meta['width']
        writer=cv2.VideoWriter(str(dest/f'{vid}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),meta['fps'],size)
        if not writer.isOpened():raise OSError('video writer failed')
        for fid in corpus.videos[vid]['frames']:
            im=cv2.cvtColor(corpus.image(vid,fid),cv2.COLOR_RGB2BGR);r=frames.get(fid)
            if r:
                b=decode_table(r['predicted_box'],t,meta)['bbox_xyxy'];gt=labels[(vid,fid)]
                for box,color in ((b,(0,0,255)),(gt,(0,255,0))):cv2.rectangle(im,tuple(round(x) for x in box[:2]),tuple(round(x) for x in box[2:]),color,3)
                note=f'{vid} frame {fid} IoU={r["iou"]:.3f} GREEN label / RED pred'
                if r['iou']<.5:errors.append(dict(video=vid,frame=fid,iou=r['iou'],visualization=f'{vid}.mp4'))
            else:note=f'{vid} frame {fid} table label UNKNOWN'
            im=cv2.resize(im,size);cv2.rectangle(im,(0,0),(size[0],35),(0,0,0),-1);cv2.putText(im,note,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);writer.write(im)
            if r and (fid==min(frames) or fid==min(frames,key=lambda f:frames[f]['iou'])):cv2.imwrite(str(dest/f'{vid}_{fid}.jpg'),im)
        writer.release()
    save(dest/'errors.json',errors)
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>Table: green label / red prediction</h1>'+''.join(f'<h2>{v}</h2><video controls width="960" src="{v}.mp4"></video>' for v in by))
    save(run/'run.json',dict(status='evaluated',metrics=str(run/'metrics.json'),visualization=str(dest/'index.html')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);render(p.parse_args().run)
