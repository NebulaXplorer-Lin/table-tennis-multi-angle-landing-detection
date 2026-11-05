"""Build a read-only table-label gallery from source-frame RGB and original boxes."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sys

import av
import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ttnet.data_tools.frame_store import FrameStore

GREEN = (65, 235, 100)
ORANGE = (255, 163, 65)
SPECIAL = {"04_004": [35, 100, 114, 115, 116, 117, 118, 119, 120, 121, 126, 195],
           "08_004": [0, 1, 15, 32, 33, 60, 129]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def boxes_at(frame):
    return [b for a in frame["annotations"] if a["label"] == "table" for b in a["bbox_xyxy"]]


def overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(1e-12, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter))


def save_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise OSError(path)


def marked(rgb, boxes, suspect=False):
    result = rgb.copy()
    for b in boxes:
        q = np.rint(b).astype(int)
        cv2.rectangle(result, tuple(q[:2]), tuple(q[2:]), ORANGE if suspect else GREEN, 2, cv2.LINE_AA)
    return result


def fit(rgb, width, height):
    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    resized = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
    canvas = np.full((height, width, 3), (17, 24, 32), np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def grid(tiles, columns):
    columns = min(columns, len(tiles))
    padded = list(tiles) + [np.full_like(tiles[0], (17, 24, 32))] * ((-len(tiles)) % columns)
    return np.vstack([np.hstack(padded[i:i + columns]) for i in range(0, len(padded), columns)])


def text(rgb, line, y, color=(235, 241, 248), size=.52):
    cv2.putText(rgb, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def tile(rgb, record, median):
    image = marked(rgb, record["boxes"], record["suspect"])
    # Include both the current box and typical-table context when a box is displaced.
    context = list(record["boxes"]) + ([median] if median is not None else [])
    if context:
        b = np.asarray(context)
        lo = np.floor(b[:, :2].min(0) - [70, 75]).astype(int)
        hi = np.ceil(b[:, 2:].max(0) + [70, 115]).astype(int)
        x1, y1 = np.maximum(lo, 0)
        x2, y2 = np.minimum(hi, [rgb.shape[1], rgb.shape[0]])
        crop = image[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, rgb.shape[1], rgb.shape[0]
        crop = image
    record["context_crop_xyxy"] = [int(v) for v in [x1, y1, x2, y2]]
    top = np.full((68, 640, 3), (17, 24, 32), np.uint8)
    color = ORANGE if record["suspect"] else GREEN
    text(top, f'{record["video_id"]}  frame {record["frame_id"]}  {record["split"]}', 24)
    coords = " | ".join(", ".join(f"{v:.2f}" for v in b) for b in record["boxes"]) or "UNKNOWN / no table label"
    text(top, coords, 49, color, .44)
    footer = np.full((28, 640, 3), (17, 24, 32), np.uint8)
    text(footer, "DETAIL: original table label; no inferred box", 20, size=.45)
    return np.vstack([top, fit(image, 640, 360), footer, fit(crop, 640, 220)])


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>桌框原始标注复核</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101820;color:#edf3f7;font:15px/1.6 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:24px}h1{margin:0 0 8px}p{color:#bdcbd7}a{color:#8acbff}
.bar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;background:#1c2a36;padding:14px;border-radius:8px}
button,select,input{font:inherit;background:#283e50;color:#fff;border:1px solid #526b7e;border-radius:5px;padding:5px 10px}
button{cursor:pointer}.info{margin:14px 0;color:#cfdeea}.green{color:#41eb64}.orange{color:#ffa341}
.viewer{position:relative;max-width:100%;margin:auto;line-height:0}.viewer img{display:block;width:100%;height:auto}
.viewer svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.hide svg{display:none}
details{margin-top:18px;background:#1b2935;padding:14px;border-radius:8px}summary{cursor:pointer}
.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:10px;margin:16px 0}
.thumbs button{padding:0;overflow:hidden;text-align:left}.thumbs img{width:100%;display:block}.thumbs span{padding:4px 8px;display:block}
.sheet{width:100%;height:auto}table{width:100%;border-collapse:collapse;margin-top:20px}td,th{text-align:left;border-bottom:1px solid #354b5c;padding:7px}
section{margin-top:22px}video{width:min(100%,1100px)}.scroll{overflow-x:auto}#zoom{width:140px}
</style><main><h1>桌框原始标注复核</h1>
<p>DATA_SUMMARY。所有框来自原始 JSON，帧号从 0 开始；缺标显示 UNKNOWN。图册覆盖代表帧，尚未逐帧审核全部数据。</p>
<p><span class="green">绿色：原始桌框</span>　<span class="orange">橙色：04_004 第 115–120 帧异常原框</span>。局部放大用于看桌面、桌沿和桌腿；裁剪范围参考同视频框分布，未生成新标签。</p>
<div class="bar"><label>视频 <select id="video"></select></label><button id="prev">← 上一帧</button><button id="next">下一帧 →</button>
<label><input id="overlay" type="checkbox" checked> 显示原始桌框</label><label>放大 <input id="zoom" type="range" min="100" max="250" step="25" value="100"></label></div>
<div id="meta" class="info"></div><div class="scroll"><div id="viewer" class="viewer"><img id="photo"><svg id="boxes" xmlns="http://www.w3.org/2000/svg"></svg></div></div>
<div id="thumbs" class="thumbs"></div><details open><summary>当前视频：全图＋桌子附近放大</summary><a id="sheetLink"><img id="sheet" class="sheet" loading="lazy"></a></details>
<section><h2>04_004 第 115–120 帧连续复核</h2><p>六帧逐帧展开，橙色为原始异常框；具体结论见 <a href="review.md">复核记录</a>。</p>
<a href="04_004_115-120.jpg"><img class="sheet" src="04_004_115-120.jpg" loading="lazy"></a>
<details><summary>04_004 完整叠图视频（196 帧，按原始 PTS 编码）</summary><video controls preload="metadata" src="04_004_original_labels.mp4"></video></details></section>
<section><h2>25 个源场景组代表帧</h2><p>用于检查框选语义与视角差异；每组一个代表帧，不代表全组已审核。</p><a href="groups_overview.jpg"><img class="sheet" src="groups_overview.jpg" loading="lazy"></a></section>
<section><h2>全部视频索引</h2><p><a href="manifest.json">来源、原始坐标和文件哈希</a> · <a href="frames.csv">抽帧清单</a> · <a href="review.md">复核记录</a></p>
<table><thead><tr><th>视频</th><th>集合／视角</th><th>有框／总帧</th><th>本次代表帧数</th></tr></thead><tbody id="index"></tbody></table></section></main>
<script>const DATA=GALLERY_DATA;let active=0,frame=0;const $=s=>document.getElementById(s);
const priority=['04_004','02_002','02_003','04_007','08_004','13_007','15_002','12_003','24_003'];
DATA.videos.sort((a,b)=>{let x=priority.indexOf(a.video_id),y=priority.indexOf(b.video_id);return (x<0?999:x)-(y<0?999:y)||a.video_id.localeCompare(b.video_id)});
for(const [i,v] of DATA.videos.entries()){let o=document.createElement('option');o.value=i;o.textContent=`${v.video_id} · ${v.split} · ${v.view} · ${v.labeled_frames}/${v.total_frames} 帧有框`;$('video').append(o);
let tr=document.createElement('tr');let td=document.createElement('td'),a=document.createElement('a');a.href='#';a.textContent=v.video_id;a.onclick=e=>{e.preventDefault();select(i);window.scrollTo({top:0,behavior:'smooth'})};td.append(a);tr.append(td);
for(const t of [`${v.split} / ${v.view}`,`${v.labeled_frames} / ${v.total_frames}`,v.frames.length]){let c=document.createElement('td');c.textContent=t;tr.append(c)}$('index').append(tr)}
function select(i){active=+i;frame=0;$('video').value=active;let v=DATA.videos[active];$('sheet').src=v.sheet;$('sheetLink').href=v.sheet;$('thumbs').replaceChildren();
for(const [j,f] of v.frames.entries()){let b=document.createElement('button'),im=document.createElement('img'),s=document.createElement('span');im.src=f.image;im.loading='lazy';s.textContent=`帧 ${f.frame_id}${f.boxes.length?'':' · 缺标'}`;b.append(im,s);b.onclick=()=>{frame=j;draw()};$('thumbs').append(b)}draw()}
function draw(){let v=DATA.videos[active],f=v.frames[frame];$('photo').src=f.image;$('photo').alt=`${v.video_id} 原帧 ${f.frame_id}`;let svg=$('boxes');svg.setAttribute('viewBox',`0 0 ${v.width} ${v.height}`);svg.replaceChildren();
for(const b of f.boxes){let r=document.createElementNS('http://www.w3.org/2000/svg','rect');r.setAttribute('x',b[0]);r.setAttribute('y',b[1]);r.setAttribute('width',b[2]-b[0]);r.setAttribute('height',b[3]-b[1]);r.setAttribute('fill','none');r.setAttribute('stroke',f.suspect?'#ffa341':'#41eb64');r.setAttribute('stroke-width',2);r.setAttribute('vector-effect','non-scaling-stroke');svg.append(r)}
$('meta').textContent=`${v.video_id} · 原帧 ${f.frame_id} · ${v.width}×${v.height} · PTS ${f.timestamp_sec.toFixed(4)}s · ${f.boxes.length?'xyxy: '+f.boxes.map(b=>b.map(x=>x.toFixed(2)).join(', ')).join(' | '):'UNKNOWN：没有桌框标注'} · ${frame+1}/${v.frames.length}`;
history.replaceState(null,'','#'+v.video_id+':'+f.frame_id)}
$('video').onchange=e=>select(e.target.value);$('prev').onclick=()=>{frame=(frame-1+DATA.videos[active].frames.length)%DATA.videos[active].frames.length;draw()};$('next').onclick=()=>{frame=(frame+1)%DATA.videos[active].frames.length;draw()};
$('overlay').onchange=e=>$('viewer').classList.toggle('hide',!e.target.checked);$('zoom').oninput=e=>{$('viewer').style.width=e.target.value+'%';$('viewer').style.maxWidth='none'};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()});
let [vid,fid]=location.hash.slice(1).split(':'),i=DATA.videos.findIndex(v=>v.video_id===vid);select(i<0?0:i);let j=DATA.videos[active].frames.findIndex(f=>f.frame_id===+fid);if(j>=0){frame=j;draw()}
</script></html>'''


def write_preview(dest, store, doc, *, highlight_original_anomalies=True,
                  label_name='original label', filename=None):
    """Encode every source frame with its original timestamp, including VFR gaps."""
    vid = doc['video_id']
    pts = np.load(store.root / vid / 'timestamps.npy', allow_pickle=False)
    with av.open(str(dest / (filename or f'{vid}_original_labels.mp4')), 'w', options={'movflags': '+faststart'}) as out:
        stream = out.add_stream('libx264', rate=30)
        stream.width, stream.height = doc['frame_width'], doc['frame_height'] + 60
        stream.pix_fmt = 'yuv420p'
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.time_base = Fraction(1, 90000)
        stream.options = {'crf': '20', 'preset': 'fast'}
        for frame in doc['frames']:
            fid = frame['frame_id']
            bs = boxes_at(frame)
            suspect = highlight_original_anomalies and vid == '04_004' and 115 <= fid <= 120
            rgb = marked(store.get_rgb(vid, fid), bs, suspect)
            header = np.full((60, rgb.shape[1], 3), (17, 24, 32), np.uint8)
            text(header, f'{vid} | source frame {fid} | PTS {pts[fid]:.4f}s | {label_name}', 23, size=.6)
            text(header, 'xyxy '+str(bs) if bs else 'UNKNOWN: missing table annotation', 48, ORANGE if suspect else GREEN, .52)
            item = av.VideoFrame.from_ndarray(np.vstack([header, rgb]), format='rgb24')
            item.time_base = Fraction(1, 90000)
            item.pts = round(float(pts[fid]) * 90000)
            for packet in stream.encode(item):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'images').mkdir()
    (dest / 'sheets').mkdir()
    cv2.setNumThreads(2)
    frozen_path = PROJECT / 'ttnet/src/configs/frozen_split.json'
    frozen = json.loads(frozen_path.read_text())
    confirmed = json.loads((PROJECT / 'data/annotations/pp_bounce_confirmed_v1.json').read_text())
    views = {v['video_id']: v['view'] for v in confirmed['videos']}
    store = FrameStore(PROJECT / 'data/datasets/pp_ball_v2')
    videos, flat, group_tiles, bad_tiles, sources = [], [], {}, [], {}
    for ix, source in enumerate(frozen['videos']):
        vid = source['video_id']
        annotation = PROJECT / source['annotation']
        before = sha(annotation)
        doc = json.loads(annotation.read_text())
        frames = {f['frame_id']: f for f in doc['frames']}
        known = [fid for fid, f in frames.items() if boxes_at(f)]
        unknown = [fid for fid, f in frames.items() if not boxes_at(f)]
        median = np.median([boxes_at(frames[f])[0] for f in known], axis=0).tolist() if known else None
        chosen = {0, doc['num_frames'] // 2, doc['num_frames'] - 1}
        if known:
            chosen.update([known[0], known[len(known) // 2], known[-1]])
            chosen.add(min(known, key=lambda fid: overlap(boxes_at(frames[fid])[0], median)))
        if unknown:
            chosen.add(unknown[len(unknown) // 2])
        chosen.update(SPECIAL.get(vid, []))
        ids = np.load(store.root / vid / 'frame_ids.npy', allow_pickle=False)
        pts = np.load(store.root / vid / 'timestamps.npy', allow_pickle=False)
        timestamps = dict(zip(ids.tolist(), pts.tolist()))
        entry = dict(video_id=vid, split=source['split'], group=source['group'], view=views[vid],
                     width=doc['frame_width'], height=doc['frame_height'], total_frames=doc['num_frames'],
                     labeled_frames=len(known), annotation=source['annotation'], annotation_sha256=before,
                     source_video=source['video'], source_video_sha256=source['video_sha256'],
                     sheet=f'sheets/{vid}.jpg', frames=[])
        tiles = []
        for fid in sorted(chosen):
            rgb = store.get_rgb(vid, fid)
            if rgb.shape != (entry['height'], entry['width'], 3):
                raise ValueError(f'dimensions differ from annotation: {vid}/{fid}')
            bs = boxes_at(frames[fid])
            record = dict(video_id=vid, frame_id=fid, split=source['split'], timestamp_sec=timestamps[fid],
                          boxes=bs, suspect=vid == '04_004' and 115 <= fid <= 120,
                          image=f'images/{vid}_{fid:06d}.jpg')
            save_rgb(dest / record['image'], rgb)
            rendered = tile(rgb, record, median)
            tiles.append(rendered)
            if source['group'] not in group_tiles and known and fid == known[len(known) // 2]:
                thumb = fit(marked(rgb, bs), 480, 270)
                header = np.full((38, 480, 3), (17, 24, 32), np.uint8)
                text(header, f'group {source["group"]} | {vid}:{fid} | {views[vid]}', 24, size=.47)
                group_tiles[source['group']] = np.vstack([header, thumb])
            if record['suspect']:
                bad_tiles.append(rendered)
            entry['frames'].append(record)
            flat.append(record)
        save_rgb(dest / entry['sheet'], grid(tiles, 4))
        if sha(annotation) != before:
            raise RuntimeError('source annotation changed while reviewing')
        sources[source['annotation']] = before
        videos.append(entry)
        if (ix + 1) % 25 == 0:
            print(f'gallery {ix+1}/{len(frozen["videos"])} videos, {len(flat)} sampled frames', flush=True)
    blank = np.full((308, 480, 3), (17, 24, 32), np.uint8)
    thumbs = [group_tiles[k] for k in sorted(group_tiles)]
    thumbs += [blank] * ((-len(thumbs)) % 5)
    save_rgb(dest / 'groups_overview.jpg', np.vstack([np.hstack(thumbs[i:i+5]) for i in range(0, len(thumbs), 5)]))
    save_rgb(dest / '04_004_115-120.jpg', np.vstack([np.hstack(bad_tiles[:3]), np.hstack(bad_tiles[3:])]))
    # Preserve real source timestamps rather than treating every source as 25 FPS.
    vid = '04_004'
    source = next(v for v in frozen['videos'] if v['video_id'] == vid)
    doc = json.loads((PROJECT / source['annotation']).read_text())
    write_preview(dest, store, doc)
    manifest = dict(created_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                    purpose='original table label visual review; no annotation changes',
                    split_id=frozen['split_id'], split_manifest_sha256=sha(frozen_path),
                    script_sha256=sha(Path(__file__)), source_annotation_hashes=sources,
                    sampled_frames=len(flat), videos=videos, original_annotations_unchanged=True,
                    video_preview=dict(video_id=vid, frames=doc['num_frames'], preserves_source_pts=True),
                    limitations=['Representative sampling is not exhaustive visual review.',
                                 'Median boxes only determine context crops; they are never drawn or exported as labels.',
                                 'JPEG review copies are not training inputs.'])
    write_json(dest / 'manifest.json', manifest)
    with (dest / 'frames.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'frame_id', 'split', 'timestamp_sec', 'boxes_xyxy', 'suspect', 'image'])
        for r in flat:
            writer.writerow([r['video_id'], r['frame_id'], r['split'], r['timestamp_sec'], json.dumps(r['boxes']), r['suspect'], r['image']])
    summary = f'{len(videos)} 个视频、{len(flat)} 个代表帧；25 个源场景组'
    page = PAGE.replace('DATA_SUMMARY', summary).replace('GALLERY_DATA', json.dumps(manifest, ensure_ascii=False).replace('</', '<\\/'))
    (dest / 'index.html').write_text(page)
    (dest / 'review.md').write_text('# 桌框原始标注复核\n\n图册已生成，目视复核结论待补充。原始标注未修改。\n')
    print(json.dumps(dict(output=str(dest), videos=len(videos), sampled_frames=len(flat), groups=len(group_tiles)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
