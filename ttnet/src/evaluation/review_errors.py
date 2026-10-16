"""Review saved validation errors without running inference again."""
from __future__ import annotations

from ttnet.data_tools.frame_store import FrameStore

import argparse
from collections import Counter
from fractions import Fraction
import html
import json
from pathlib import Path

import av
import cv2

from ttnet.common import DATA,jsonl,save,now
from ttnet.data_tools.geometry import letterbox
from .evaluate import overlay
from .metrics import accept


CAUSES = {'gt_outside_roi':'GT 在 Local 裁剪区域外', 'local_error':'GT 在 ROI 内，Local 定位偏差',
          'global_fallback_error':'GT 在 ROI 内，Local 无有效结果，回退 Global 仍有偏差',
          'no_global_roi':'Global 未通过阈值，无 Local 裁剪区域'}


def encode_clip(path, rows, records, dataset, target, global_threshold):
    frame_store=FrameStore(dataset)
    fps=max(1,round(rows[0]['nominal_fps']/3))
    with av.open(str(path),mode='w') as output:
        stream=output.add_stream('libx264',rate=fps)
        stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p'
        stream.options={'crf':'20','preset':'veryfast'}
        for i,row in enumerate(rows):
            source=frame_store.row_rgb(row)[:,:,::-1].copy()
            if source is None:raise RuntimeError(f'unreadable frame: {row["frame_path"]}')
            record=records[(row['video_id'],row['frame_id'])]
            painted,_,_=overlay(source,row,record['prediction'])
            g=accept(record['global_candidate'],global_threshold,row['resolution'])
            if g and g['valid']:
                cv2.drawMarker(painted,(round(g['x']),round(g['y'])),(0,220,255),cv2.MARKER_DIAMOND,20,2)
            cv2.putText(painted,'YELLOW: accepted Global | playback: 1/3 speed'+(' | SELECTED ERROR' if row['frame_id']==target else ''),
                        (10,82),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
            painted,_=letterbox(painted,1280,720)
            frame=av.VideoFrame.from_ndarray(painted,format='bgr24')
            frame.pts=i;frame.time_base=Fraction(1,fps)
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    with av.open(str(path)) as check:
        decoded=sum(1 for _ in check.decode(video=0))
    if decoded!=len(rows):raise RuntimeError('review video frame count differs')
    return dict(frames=len(rows),source_fps=rows[0]['nominal_fps'],playback_fps=fps,
                source_span_sec=rows[-1]['timestamp_sec']-rows[0]['timestamp_sec'],decode_checked=True)


def build(run, dataset=DATA):
    run=run.resolve();dataset=dataset.resolve()
    out=run/'evaluations/best_val'
    metrics=json.loads((out/'metrics.json').read_text())
    state=json.loads((run/'run.json').read_text())
    errors=json.loads((out/'errors.json').read_text())
    records={(r['video_id'],r['frame_id']):r for r in jsonl(run/'predictions/best_val.jsonl')}
    all_rows=jsonl(dataset/'labels/val.jsonl')
    groups={}
    for row in all_rows:groups.setdefault(row['video_id'],[]).append(row)
    enriched=[]
    for error in errors:
        record=records[(error['video_id'],error['frame_id'])]
        local=record['local_candidate'];pred=record['prediction']
        crop=local.get('crop_transform') if local else None
        x,y=record['ball_centers'][0]
        inside=bool(crop and crop['left']<=x<crop['left']+crop['side'] and crop['top']<=y<crop['top']+crop['side'])
        cause='no_global_roi' if not crop else ('gt_outside_roi' if not inside else
              ('local_error' if pred and pred['valid'] and pred['source']=='local' else 'global_fallback_error'))
        target=run/error['visualization']
        if not target.is_file():raise RuntimeError('missing saved error image')
        enriched.append(dict(error,cause=cause,cause_label=CAUSES[cause],timestamp_sec=record['timestamp_sec'],
                             confidence=pred['confidence'] if pred and pred['valid'] else None,
                             branch=pred['source'] if pred and pred['valid'] else 'missing',
                             global_confidence=record['global_candidate']['confidence'] if record['global_candidate'] else None,
                             gt_inside_roi=inside,image='../../'+error['visualization']))
    save(out/'error_review.json',enriched)
    counts=Counter(e['cause'] for e in enriched)
    selected=[]
    clip_dir=run/'visualizations/error_clips';clip_dir.mkdir(parents=True,exist_ok=True)
    for cause in CAUSES:
        choices=[e for e in enriched if e['cause']==cause]
        if not choices:continue
        e=choices[0]
        rows=[r for r in groups[e['video_id']] if abs(r['timestamp_sec']-e['timestamp_sec'])<=.75]
        dest=clip_dir/f"{cause}_{e['video_id']}_{e['frame_id']:06d}.mp4"
        info=encode_clip(dest,rows,records,dataset,e['frame_id'],metrics['global_threshold'])
        selected.append(dict(e,clip='../../'+str(dest.relative_to(run)),clip_info=info))
        print(f'clip {cause}: {e["video_id"]} frame {e["frame_id"]}, {len(rows)} frames',flush=True)
    save(out/'error_clips.json',selected)
    data=json.dumps(enriched,ensure_ascii=False).replace('<','\\u003c')
    videos=sorted({e['video_id'] for e in enriched})
    options=''.join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in videos)
    causes=''.join(f'<option value="{k}">{html.escape(v)}</option>' for k,v in CAUSES.items())
    clips=''.join(f'<article><h3>{html.escape(e["cause_label"])}</h3><video controls preload="metadata" src="{e["clip"]}"></video><p>{e["video_id"]} · 错误帧 {e["frame_id"]} · 原始 {e["clip_info"]["source_fps"]:.1f}fps，约 1/3 速度播放</p></article>' for e in selected)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三帧组合定位：错误检查</title>
<style>body{margin:24px;background:#101820;color:#eaf1f6;font:16px system-ui}h1,h2{font-weight:650}p{line-height:1.6}a{color:#7cc9ff}select,button{padding:8px;margin:4px;background:#23313e;color:white;border:1px solid #647586;border-radius:5px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}article{background:#1c2934;border-radius:8px;padding:12px}img,video{width:100%;height:auto}small{color:#bbcbd7}#count{margin:12px 0}.legend span{margin-right:16px}.gt{color:#3cf03c}.pred{color:#ff3c3c}.roi{color:#3cbeff}.global{color:#ffdc00}</style>
<h1>固定三帧 Global＋Local：错误检查</h1>
<p>第 13 轮最佳 Local，Global 第 20 轮；验证 F1@10px 85.08%。10px 之外计为定位错误。未知标签不计入错误。</p>
<p class="legend"><span class="gt">绿色：GT</span><span class="pred">红色：最终 Pred</span><span class="roi">蓝色：Local ROI</span><span class="global">黄色：Global（视频）</span></p>
<p>ERROR_COUNTS。分组依据 GT 是否在 ROI 内、最终结果的来源。点击图片可查看原分辨率；图片标题显示置信度、来源分支和误差。</p>
<h2>先看四类错误的前后帧慢放</h2><div class="grid">CLIPS</div>
<h2>全部错误帧</h2><label>类型 <select id="kind"><option value="">全部</option><option value="location_error">定位偏差</option><option value="miss">漏检</option></select></label>
<label>分组 <select id="cause"><option value="">全部</option>CAUSE_OPTIONS</select></label>
<label>视频 <select id="video"><option value="">全部</option>VIDEO_OPTIONS</select></label>
<label>排序 <select id="sort"><option value="frame">按视频和帧号</option><option value="error">误差从大到小</option><option value="confidence">置信度从高到低</option></select></label>
<div id="count"></div><button id="prev">上一页</button><button id="next">下一页</button><div id="images" class="grid"></div>
<script>const data=ERROR_DATA;let page=0;const size=48;const $=id=>document.getElementById(id);
function render(){let rows=data.filter(e=>(!$('kind').value||e.kind===$('kind').value)&&(!$('cause').value||e.cause===$('cause').value)&&(!$('video').value||e.video_id===$('video').value));const s=$('sort').value;rows.sort((a,b)=>s==='frame'?a.video_id.localeCompare(b.video_id)||a.frame_id-b.frame_id:s==='error'?(b.error_px??-1)-(a.error_px??-1):(b.confidence??-1)-(a.confidence??-1));const pages=Math.max(1,Math.ceil(rows.length/size));page=Math.min(page,pages-1);$('count').textContent=`匹配 ${rows.length} / ${data.length} 个错误，第 ${page+1}/${pages} 页`;$('prev').disabled=page===0;$('next').disabled=page>=pages-1;$('images').replaceChildren();for(const e of rows.slice(page*size,(page+1)*size)){const card=document.createElement('article'),a=document.createElement('a'),im=document.createElement('img'),title=document.createElement('h3'),p=document.createElement('p');a.href=e.image;a.target='_blank';im.src=e.image;im.loading='lazy';im.alt=e.video_id+' frame '+e.frame_id;a.append(im);title.textContent=e.video_id+' · frame '+e.frame_id+' · '+(e.kind==='miss'?'漏检':'定位偏差');p.textContent=e.cause_label+' | 来源 '+e.branch+' | conf '+(e.confidence===null?'—':e.confidence.toFixed(3))+' | 误差 '+(e.error_px===null?'—':e.error_px.toFixed(1)+'px');card.append(title,a,p);$('images').append(card)}}for(const id of ['kind','cause','video','sort'])$(id).onchange=()=>{page=0;render()};$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();</script></html>'''
    page=page.replace('ERROR_COUNTS',f'共 {len(errors)} 个错误：{sum(e["kind"]=="location_error" for e in errors)} 个定位偏差、{sum(e["kind"]=="miss" for e in errors)} 个漏检')
    page=page.replace('第 13 轮最佳 Local，Global 第 20 轮；验证 F1@10px 85.08%。',
                      f'最佳 Local 来自第 {state["best_epoch"]} 轮；验证 F1@10px {metrics["center"]["10"]["f1"]:.2%}。')
    page=page.replace('CLIPS',clips).replace('CAUSE_OPTIONS',causes).replace('VIDEO_OPTIONS',options).replace('ERROR_DATA',data)
    (out/'errors.html').write_text(page)
    lines=['# 组合定位错误检查','', '[打开错误图册和慢放视频](errors.html)','',
           '绿色 = GT；红色 = 最终 Pred；蓝色 = Local ROI；视频额外显示黄色 Global。所有帧均来自已保存原图，使用同一次 best_val 的预测和阈值，没有重新推理。','',
           '| 分组 | 错误帧数 |','|---|---:|']
    lines += [f'| {CAUSES[cause]} | {counts[cause]} |' for cause in CAUSES]
    lines += ['','图册覆盖全部评分错误；视频是各分组的一个示例，保留中心前后约 0.75 秒的真实帧，约 1/3 速度播放。','']
    for e in selected:
        lines += [f'- {e["cause_label"]}：[{e["video_id"]}，帧 {e["frame_id"]} 图片]({e["image"]}) · [慢放视频]({e["clip"]})']
    (out/'ERROR_REVIEW.md').write_text('\n'.join(lines)+'\n')
    save(out/'error_review_summary.json',dict(created=now(),error_count=len(errors),diagnostic_groups=dict(counts),
         global_checkpoint_sha256=metrics['global_checkpoint_sha256'],local_checkpoint_sha256=metrics['local_checkpoint_sha256'],clips=len(selected)))
    from ttnet.results import write_results
    write_results()
    return out/'errors.html'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,default=DATA)
    args=p.parse_args();print(build(args.run,args.dataset))


if __name__=='__main__':main()
