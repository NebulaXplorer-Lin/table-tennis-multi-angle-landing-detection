"""Build a positive-labelled-only, source-grouped YOLO localisation pilot dataset."""
import hashlib
import json
from pathlib import Path
import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/pp_train_data'
DEST = ROOT / 'out/datasets/yolo_ball_v1'
SPLIT = ROOT / 'experiments/splits/yolo_ball_v1'
VAL_GROUPS = {'02', '03', '04', '12', '21', '22'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    SPLIT.mkdir(parents=True, exist_ok=True)
    records, audit = [], []
    for path in sorted((SOURCE / 'annotations').glob('*/*.json')):
        data = json.loads(path.read_text())
        group = path.parent.name
        split = 'val' if group in VAL_GROUPS else 'train'
        video = SOURCE / data['video_path']
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened(), video
        w, h = int(cap.get(3)), int(cap.get(4))
        assert (w, h) == (data['frame_width'], data['frame_height']), video
        frames = {f['frame_id']: f for f in data['frames']}
        assert len(frames) == len(data['frames'])
        index, kept, unknown, clipped = 0, 0, 0, 0
        (DEST / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DEST / 'labels' / split).mkdir(parents=True, exist_ok=True)
        while True:
            ok, image = cap.read()
            if not ok:
                break
            f = frames.get(index)
            boxes = [b for a in (f or {}).get('annotations', []) if a['label'] == 'ball' for b in a['bbox_xyxy']]
            if not boxes:
                unknown += 1
            elif split == 'val' or index % 5 == 0:
                lines, valid = [], []
                for b in boxes:
                    x1, y1, x2, y2 = b
                    assert x2 > x1 and y2 > y1, (video, index, b)
                    c = [max(0., x1), max(0., y1), min(float(w), x2), min(float(h), y2)]
                    assert c[2] > c[0] and c[3] > c[1], (video, index, b)
                    clipped += c != b
                    x1,y1,x2,y2 = c
                    lines.append(f'0 {(x1+x2)/2/w:.9f} {(y1+y2)/2/h:.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}')
                    valid.append(c)
                name = f'{group}_{path.stem}_{index:06d}'
                img = DEST / 'images' / split / (name + '.jpg')
                assert cv2.imwrite(str(img), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (DEST / 'labels' / split / (name + '.txt')).write_text('\n'.join(lines)+'\n')
                records.append(dict(split=split, group=group, video=str(video.relative_to(ROOT)), frame=index,
                                    width=w, height=h, boxes=valid, image=str(img.relative_to(ROOT))))
                if kept == 0:
                    for b in valid:
                        cv2.rectangle(image, tuple(map(round,b[:2])), tuple(map(round,b[2:])), (0,255,0), 2)
                    preview = DEST / 'label_previews'; preview.mkdir(exist_ok=True)
                    cv2.imwrite(str(preview / f'{group}_{path.stem}.jpg'), image)
                kept += 1
            index += 1
        cap.release()
        assert index == data['num_frames'], (video, index, data['num_frames'])
        assert all(0 <= i < index for i in frames)
        audit.append(dict(video=str(video.relative_to(ROOT)), group=group, split=split, frames=index,
                          selected=kept, unknown=unknown, clipped_boxes=clipped, width=w, height=h,
                          annotation_sha256=sha(path), video_sha256=sha(video)))
    for split in ['train','val']:
        lines = [json.dumps(r) for r in records if r['split']==split]
        (SPLIT / f'{split}.jsonl').write_text('\n'.join(lines)+'\n')
    (SPLIT / 'audit.json').write_text(json.dumps(audit, indent=2))
    cfg = dict(path=str(DEST), train='images/train', val='images/val', names={0:'ball'})
    (DEST / 'data.yaml').write_text(yaml.safe_dump(cfg))
    for split in ['train','val']:
        names = [str(ROOT/r['image']) for r in records if r['split']==split][:64 if split=='train' else 32]
        (DEST / f'smoke_{split}.txt').write_text('\n'.join(names)+'\n')
    (DEST / 'smoke.yaml').write_text(yaml.safe_dump(dict(cfg, train='smoke_train.txt', val='smoke_val.txt')))
    print(json.dumps(dict(samples={s:sum(r['split']==s for r in records) for s in ['train','val']},
                          videos=len(audit), frames=sum(r['frames'] for r in audit),
                          unknown=sum(r['unknown'] for r in audit), clipped=sum(r['clipped_boxes'] for r in audit)),indent=2))

if __name__ == '__main__':
    main()
