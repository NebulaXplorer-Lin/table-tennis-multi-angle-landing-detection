"""Render measured detections and GT as labelled images; no new model inference."""
import json
from pathlib import Path
import sys
import html
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run_yolo_ball import ROOT, match

EXP='20261013-163117_yolo11n_ball-640'
DEST=ROOT/'out/experiments'/EXP/'visual_review'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BG='#101820';WHITE='#eef3f8';MUTED='#bbc8d5'
GREEN='#42e39a';RED='#ff6b73';BLUE='#61baff';ORANGE='#ffbf57'

def font(n):return ImageFont.truetype(FONT,n)

def label(draw,xy,text,color,size=17):
    x,y=xy;f=font(size);b=draw.textbbox((x,y),text,font=f)
    draw.rectangle((b[0]-4,b[1]-3,b[2]+4,b[3]+3),fill=BG)
    draw.text((x,y),text,font=f,fill=color)

def iou(a,b):
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter+1e-9))

def analyze(r):
    gt=np.array(r['boxes']);p=np.array(r['predicted_boxes']).reshape(-1,4)
    matches=match(gt,p,10);pairs={j:(i,d) for i,j,d in matches}
    info=[]
    for j,box in enumerate(p):
        nearest=int(np.argmin(np.linalg.norm((gt[:,:2]+gt[:,2:])/2-(box[:2]+box[2:])/2,axis=1)))
        i,d=pairs[j] if j in pairs else (nearest,float(np.linalg.norm((gt[nearest,:2]+gt[nearest,2:])/2-(box[:2]+box[2:])/2)))
        info.append(dict(tp=j in pairs,distance=d,iou=iou(gt[i],box),score=r['confidence'][j]))
    missed=[i for i in range(len(gt)) if i not in {i for i,j,d in matches}]
    return info,missed

def render(r,target,focus=None):
    im=Image.open(ROOT/r['image']).convert('RGB');info,missed=analyze(r)
    gt=np.array(r['boxes']);p=np.array(r['predicted_boxes']).reshape(-1,4)
    if focus is None:focus=0 if len(p) else None
    anchor=p[focus] if focus is not None else gt[0]
    # Include both target and nearest labelled ball; distant errors retain enough context.
    nearest=min(gt,key=lambda b:np.linalg.norm((b[:2]+b[2:]-anchor[:2]-anchor[2:])/2))
    union=np.array([min(anchor[0],nearest[0]),min(anchor[1],nearest[1]),max(anchor[2],nearest[2]),max(anchor[3],nearest[3])])
    side=max(104,float(max(union[2:]-union[:2]))+70);cx,cy=(union[:2]+union[2:])/2
    crop=(max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(im.width,int(cx+side/2)),min(im.height,int(cy+side/2)))
    canvas=Image.new('RGB',(1440,740),BG);d=ImageDraw.Draw(canvas)
    title='检对球心' if info and all(x['tp'] for x in info) else ('含误检' if info else '漏检：模型没有输出框')
    d.text((24,14),title,font=font(29),fill=WHITE)
    d.text((24,59),f"视频 {Path(r['video']).parent.name}/{Path(r['video']).name}  ·  第 {r['frame']} 帧（0 基）  ·  conf ≥ 0.001",font=font(19),fill=MUTED)
    def panel(src,origin,width,height,cropbox):
        x,y=origin; src=src.crop(cropbox);scale=min(width/src.width,height/src.height)
        sw,sh=round(src.width*scale),round(src.height*scale);ox=x+(width-sw)//2;oy=y+(height-sh)//2
        canvas.paste(src.resize((sw,sh),Image.Resampling.LANCZOS),(ox,oy))
        def project(b):return [(b[0]-cropbox[0])*scale+ox,(b[1]-cropbox[1])*scale+oy,(b[2]-cropbox[0])*scale+ox,(b[3]-cropbox[1])*scale+oy]
        for i,b in enumerate(gt):
            q=project(b)
            if q[2]<ox or q[0]>ox+sw or q[3]<oy or q[1]>oy+sh:continue
            color=ORANGE if i in missed else BLUE
            # Dashes distinguish human ground truth from solid predicted boxes.
            x1,y1,x2,y2=q
            for xx in np.arange(x1,x2,8):d.line((xx,y1,min(xx+4,x2),y1),fill=color,width=2);d.line((xx,y2,min(xx+4,x2),y2),fill=color,width=2)
            for yy in np.arange(y1,y2,8):d.line((x1,yy,x1,min(yy+4,y2)),fill=color,width=2);d.line((x2,yy,x2,min(yy+4,y2)),fill=color,width=2)
            text='GT ball 漏检' if i in missed else 'GT ball'
            label(d,(max(ox,min(q[0],ox+sw-140)),min(oy+sh-25,q[3]+28)),text,color,17)
        for j,b in enumerate(p):
            q=project(b)
            if q[2]<ox or q[0]>ox+sw or q[3]<oy or q[1]>oy+sh:continue
            color=GREEN if info[j]['tp'] else RED;d.rectangle(q,outline=color,width=3)
            text=f"P{j+1} ball {info[j]['score']:.4f} {'检对' if info[j]['tp'] else '误检'}"
            label(d,(max(ox,min(q[0],ox+sw-260)),max(oy+3,q[1]-29-j*30)),text,color,17)
    d.text((24,98),'原图 + 预测框',font=font(20),fill=WHITE)
    d.text((940,98),'局部放大（框位置未调整）',font=font(20),fill=WHITE)
    panel(im,(24,135),884,498,(0,0,im.width,im.height))
    panel(im,(940,135),476,498,crop)
    if info:
        selected=info[focus or 0]
        d.text((24,654),f"P{(focus or 0)+1}  类别 ball（乒乓球）  ·  置信度 {selected['score']:.6f}  ·  球心误差 {selected['distance']:.2f}px  ·  框 IoU {selected['iou']:.3f}",font=font(22),fill=WHITE)
        d.text((24,696),f"球心一对一匹配 ≤10px：{'通过' if selected['tp'] else '未通过'}    |    框 IoU ≥0.5：{'达到' if selected['iou']>=.5 else '未达到'}    |    绿色实线=检对球心；红色实线=误检；蓝/橙虚线=人工标注",font=font(16),fill=MUTED)
    else:
        d.text((24,654),'模型没有输出 ball 框；橙色虚线是人工标注的位置，不是模型预测。',font=font(23),fill=ORANGE)
        d.text((24,696),'原实验 conf=0.25 时，全部 5,302 张验证帧均无预测框。此图使用事后诊断阈值 0.001。',font=font(16),fill=MUTED)
    canvas.save(target,quality=92)
    return canvas

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in (ROOT/'out/experiments'/EXP/'eval/score_diagnostic/predictions.jsonl').read_text().splitlines()]
    detected=[r for r in rows if r['predicted_boxes']]
    empty=[r for r in rows if not r['predicted_boxes']]
    selected=list(detected)
    for group in sorted({r['group'] for r in empty}):
        group_rows=[r for r in empty if r['group']==group]
        selected.extend([group_rows[0],group_rows[len(group_rows)//2]])
    gallery=[]
    for r in selected:
        info,missed=analyze(r)
        status='fp' if any(not i['tp'] for i in info) else ('tp' if info else 'fn')
        name=Path(r['image']).stem+'.jpg';render(r,DEST/name)
        gallery.append(dict(image=name,status=status,video=r['video'],frame=r['frame']))
    candidates=[(r,analyze(r)[0]) for r in detected]
    good=max((r for r,i in candidates if len(i)==1 and i[0]['tp'] and i[0]['iou']>=.5),key=lambda r:max(r['confidence']))
    loose=next(r for r,i in candidates if any(t['tp'] and t['iou']<.5 for t in i))
    wrong=max((r for r,i in candidates if any(not t['tp'] for t in i)),key=lambda r:max(t['distance'] for t in analyze(r)[0] if not t['tp']))
    examples=[('correct',good),('center_only',loose),('wrong',wrong),('missed',empty[0])]
    for name,r in examples:
        info,_=analyze(r);focus=next((j for j,t in enumerate(info) if (not t['tp'] if name=='wrong' else t['tp'])),None)
        render(r,DEST/f'example_{name}.jpg',focus)
    cards=[]
    for item in gallery:
        tag={'tp':'检对球心','fp':'含误检','fn':'完全漏检'}[item['status']]
        cards.append(f'<article data-kind="{item["status"]}"><h2>{tag} · {html.escape(Path(item["image"]).stem)}</h2><a href="{item["image"]}" target="_blank"><img loading="lazy" src="{item["image"]}" alt="{tag}，原图与放大区域，带类别置信度和正确性标签"></a></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLO 球定位 · 逐帧可视化</title><style>body{font-family:system-ui,sans-serif;background:#101820;color:#eef3f8;margin:24px auto;padding:0 20px;max-width:1440}p{line-height:1.8;color:#bbc8d5}button{font:inherit;padding:10px 18px;margin:4px;border-radius:6px;border:1px solid #738394;background:#1e2c38;color:white;cursor:pointer}button[aria-pressed=true]{background:#e5edf4;color:#101820}img{width:100%;height:auto}article{margin:32px 0}h2{font-size:18px}a{color:#61baff}</style><h1>模型究竟框对了什么？</h1><p>每个模型预测框都有 <b>ball（乒乓球）+ 置信度 + 检对/误检</b>。绿色实线：球心定位通过（≤10 原图像素，且一对一匹配）；红色实线：预测未匹配到球；蓝色虚线：已匹配人工标注；橙色虚线：漏检的人工标注。<b>虚线框不是模型画出的框。</b></p><p>此处展示事后诊断阈值 conf=0.001 的全部 64 张有输出帧（69 个预测框），以及按来源抽取的 12 张完全漏检帧。主实验 conf=0.25 下全部验证帧无输出。球心正确不一定代表框大小准确，每张图另外显示 IoU 是否达到 0.5。TP/FP 是评估标签，模型本身只输出 ball 类别、分数和坐标。未标注目标可能影响误检判断。</p><nav aria-label="筛选结果"><button data-filter="all" aria-pressed="true">全部</button><button data-filter="tp" aria-pressed="false">检对球心</button><button data-filter="fp" aria-pressed="false">含误检</button><button data-filter="fn" aria-pressed="false">完全漏检</button></nav><p id="count" aria-live="polite"></p>'''+''.join(cards)+'''<script>const buttons=[...document.querySelectorAll('button')],cards=[...document.querySelectorAll('article')];function choose(kind){let n=0;cards.forEach(c=>{c.hidden=kind!=='all'&&c.dataset.kind!==kind;if(!c.hidden)n++});buttons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.filter===kind)));document.getElementById('count').textContent='当前显示 '+n+' 张 · 点击图片查看原尺寸'}buttons.forEach(b=>b.onclick=()=>choose(b.dataset.filter));choose('all');</script></html>'''
    (DEST/'index.html').write_text(page)
    (DEST/'manifest.json').write_text(json.dumps(gallery,indent=2,ensure_ascii=False))
    print(json.dumps(dict(directory=str(DEST),detected_frames=len(detected),images=len(gallery),categories={k:sum(r['status']==k for r in gallery) for k in ['tp','fp','fn']}),ensure_ascii=False))

if __name__=='__main__':main()
