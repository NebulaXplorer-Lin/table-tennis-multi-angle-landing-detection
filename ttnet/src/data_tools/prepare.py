"""Lossless full-frame materialization, immutable source identities, resumable per video."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import json
from pathlib import Path
import shutil
import time

import av
import cv2
import numpy as np

from ttnet.common import DATA, PROJECT, ROOT, identity, now, save, save_jsonl, sha, jsonl
from .geometry import window_indices

SCHEMA = "ttnet_ball_v2.0"
POLICY = dict(schema=SCHEMA, original_format="png", png_compression=1,
              rgb_decode="PyAV rgb24",
              half_span_sec=1/30, slots=9, center_slot=4,
              unknown="missing ball annotation is unknown, not background",
              coordinate_space="original image pixels; input scaling belongs to training/inference")


def read_png(path):
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"unreadable PNG: {path}")
    return im


def process_video(args):
    entry, dataset, policy_hash = args
    cv2.setNumThreads(1)
    dataset = Path(dataset)
    key = entry["video_id"]
    meta_path = dataset / "metadata/videos" / f"{key}.json"
    label_path = dataset / "labels" / f"{key}.jsonl"
    window_path = dataset / "windows" / f"{key}.jsonl"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["identity"] != identity(entry) or old["policy_hash"] != policy_hash:
            raise RuntimeError(f"existing video identity differs: {key}; create a new dataset version")
        if old.get("complete") and label_path.exists() and window_path.exists():
            rows = jsonl(label_path)
            if len(rows) != old["actual_frames"] or len(jsonl(window_path)) != len(rows):
                raise RuntimeError(f"incomplete metadata: {key}")
            for r in rows:
                for field, dims in [("frame_path", (old["resolution"]["width"], old["resolution"]["height"]))]:
                    im = read_png(dataset/r[field])
                    if im.shape[1::-1] != dims:
                        raise RuntimeError(f"image shape mismatch: {key}")
            return old, True
    if (dataset / "frames").is_symlink():
        raise RuntimeError("shared original frames are immutable; recover missing metadata with migrate_raw, not by re-decoding")
    doc = json.loads((PROJECT / entry["annotation"]).read_text())
    frames = {f["frame_id"]: f for f in doc["frames"]}
    if len(frames) != len(doc["frames"]):
        raise RuntimeError(f"duplicate annotation frame: {key}")
    orig_dir = dataset / "frames" / key
    orig_dir.mkdir(parents=True, exist_ok=True)
    rows, times, total_bytes, clipped, preview_done = [], [], 0, 0, False
    with av.open(str(PROJECT / entry["video"])) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 1
        fps = Fraction(stream.average_rate)
        w, h = stream.width, stream.height
        if (w, h) != (doc["frame_width"], doc["frame_height"]):
            raise RuntimeError(f"annotation resolution mismatch: {key}")
        first_pts = None
        for fid, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                raise RuntimeError(f"missing PTS: {key}/{fid}")
            pts_sec = float(frame.pts * frame.time_base)
            if first_pts is None:
                first_pts = pts_sec
            timestamp = pts_sec-first_pts
            if times and timestamp <= times[-1]:
                raise RuntimeError(f"non-monotonic PTS: {key}/{fid}")
            times.append(timestamp)
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape[:2] != (h, w):
                raise RuntimeError(f"variable resolution: {key}/{fid}")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            original = orig_dir / f"{fid:06d}.png"
            # Atomic image writes; incomplete video can be safely reprocessed.
            for path, image in [(original, bgr)]:
                temp = path.with_name(path.stem + ".tmp.png")
                if not cv2.imwrite(str(temp), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError(f"failed image write: {path}")
                temp.replace(path)
                total_bytes += path.stat().st_size
            raw = frames.get(fid, {"frame_id": fid, "annotations": []})
            boxes = [b for a in raw.get("annotations", []) if a["label"] == "ball" for b in a["bbox_xyxy"]]
            valid, problems = [], []
            for box in boxes:
                if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                    problems.append("invalid_box")
                    continue
                x1, y1, x2, y2 = map(float, box)
                bounded = [max(0., x1), max(0., y1), min(float(w), x2), min(float(h), y2)]
                if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
                    problems.append("outside_image")
                    continue
                clipped += bounded != list(box)
                valid.append(bounded)
            status = "unknown" if not boxes else ("positive" if len(valid) == 1 and not problems and len(boxes) == 1 else "ignore")
            if len(boxes) > 1:
                problems.append("multiple_ball_boxes")
            centers = [[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in valid]
            row = dict(video_id=key, split=entry["split"], frame_id=fid, timestamp_sec=timestamp,
                       pts=frame.pts, time_base_num=frame.time_base.numerator, time_base_den=frame.time_base.denominator,
                       fps=float(fps), fps_num=fps.numerator, fps_den=fps.denominator,
                       resolution=dict(width=w, height=h), ball_boxes=valid, ball_centers=centers,
                       label_status=status, annotation_issues=problems, raw_annotations=raw.get("annotations", []),
                       frame_path=str(original.relative_to(dataset)))
            rows.append(row)
            if status == "positive" and not preview_done:
                preview = bgr.copy()
                for b, c in zip(valid, centers):
                    cv2.rectangle(preview, tuple(map(round, b[:2])), tuple(map(round, b[2:])), (0,255,0), 2)
                    cv2.drawMarker(preview, tuple(map(round, c)), (0,255,0), cv2.MARKER_CROSS, 14, 1)
                cv2.putText(preview, f"{key} frame {fid} | GT green | {w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,255,0), 2)
                p = dataset / "previews" / f"{key}.jpg"
                p.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(p), preview)
                preview_done = True
    if len(rows) != doc["num_frames"] or any(i < 0 or i >= len(rows) for i in frames):
        raise RuntimeError(f"frame count/range mismatch: {key}")
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1/float(fps)
    nominal = float(1/dt)
    for row in rows:
        row["nominal_fps"] = nominal
    windows = [dict(video_id=key, split=entry["split"], **window_indices(times, i)) for i in range(len(times))]
    save_jsonl(label_path, rows)
    save_jsonl(window_path, windows)
    meta = dict(**entry, identity=identity(entry), policy_hash=policy_hash, complete=True,
                actual_frames=len(rows), resolution=dict(width=w, height=h), fps=float(fps),
                fps_num=fps.numerator, fps_den=fps.denominator, nominal_fps=nominal,
                pts_audit=dict(first_pts_sec=first_pts, median_interval_sec=dt,
                               min_interval_sec=float(min(np.diff(times))) if len(times)>1 else None,
                               max_interval_sec=float(max(np.diff(times))) if len(times)>1 else None,
                               monotonic=True), label_counts={s:sum(r["label_status"]==s for r in rows) for s in ["positive", "unknown", "ignore", "negative"]},
                png_bytes=total_bytes, clipped_boxes=clipped, labels_sha256=sha(label_path),
                windows_sha256=sha(window_path), decode_version=av.__version__, image_encoder_version=cv2.__version__)
    save(meta_path, meta)
    return meta, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", type=int, default=0, help="convert only N diverse pilot videos; later rerun without this flag")
    args = parser.parse_args()
    source = PROJECT / "experiments/splits/yolo_ball_v1/audit.json"
    audits = json.loads(source.read_text())
    entries = []
    for a in audits:
        annotation = f"data/pp_train_data/annotations/{a['group']}/{Path(a['video']).stem}.json"
        entry = dict(video_id=f"{a['group']}_{Path(a['video']).stem}", group=a["group"], split=a["split"],
                     video=a["video"], annotation=annotation, video_sha256=sha(PROJECT/a["video"]),
                     annotation_sha256=sha(PROJECT/annotation))
        if entry["video_sha256"] != a["video_sha256"] or entry["annotation_sha256"] != a["annotation_sha256"]:
            raise RuntimeError(f"source changed from frozen pilot: {entry['video_id']}")
        entries.append(entry)
    if len(entries) != 154 or sum(e["split"]=="train" for e in entries)!=114 or sum(e["split"]=="val" for e in entries)!=40:
        raise RuntimeError("expected frozen 114/40 video partition")
    by_hash = {}
    for e in entries:
        prev = by_hash.get(e["video_sha256"])
        if prev and prev != e["split"]:
            raise RuntimeError("duplicate video crosses splits")
        by_hash[e["video_sha256"]] = e["split"]
    policy_hash = identity(POLICY)
    dataset_id = identity(dict(policy_hash=policy_hash, videos=entries))
    manifest_path = args.dataset / "metadata/dataset.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["dataset_id"] != dataset_id:
        raise RuntimeError("dataset identity differs; choose a new version directory")
    save(args.dataset / "metadata/processing_config.json", POLICY)
    for split in ["train", "val"]:
        save_jsonl(args.dataset / "splits" / f"{split}.jsonl", [e for e in entries if e["split"]==split])
    frozen = ROOT / "src/configs/frozen_split.json"
    frozen_obj = dict(split_id="ttnet_ball_v1_frozen_yolo_pilot", source_audit_sha256=sha(source),
                      videos=entries, videos_hash=identity(entries), train_videos=114, val_videos=40,
                      note="user-authorized reuse of historical scene-grouped split; no re-splitting")
    if frozen.exists() and json.loads(frozen.read_text()) != frozen_obj:
        raise RuntimeError("frozen split changed")
    save(frozen, frozen_obj)
    selected = entries
    if args.pilot:
        selected = []
        seen = set()
        for e in entries:
            with av.open(str(PROJECT/e["video"])) as c:
                s = c.streams.video[0]
                bucket = (s.width, s.height, min([25,30,60], key=lambda f:abs(f-float(s.average_rate))))
            if bucket not in seen:
                selected.append(e); seen.add(bucket)
            if len(selected) == args.pilot:
                break
    save(manifest_path, dict(schema=SCHEMA, dataset_id=dataset_id, policy_hash=policy_hash,
                             status="preparing", started=now(), total_videos=154, source_audit_sha256=sha(source)))
    start, metas = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(process_video, (e, str(args.dataset), policy_hash)):e for e in selected}
        for future in as_completed(pending):
            meta, reused = future.result()
            metas.append(meta)
            print(f"{len(metas)}/{len(selected)} {meta['video_id']} {'verified cache' if reused else 'saved'} "
                  f"{meta['actual_frames']} frames {meta['resolution']} fps={meta['nominal_fps']:.3f}", flush=True)
    all_meta = [json.loads(p.read_text()) for p in sorted((args.dataset/"metadata/videos").glob("*.json"))]
    complete = len(all_meta)==len(entries) and {m["video_id"] for m in all_meta}=={e["video_id"] for e in entries}
    for split in ["train", "val"]:
        save_jsonl(args.dataset/"labels"/f"{split}.jsonl", (r for m in all_meta if m["split"]==split for r in jsonl(args.dataset/"labels"/f"{m['video_id']}.jsonl")))
    total = sum(m["actual_frames"] for m in all_meta)
    size = sum(m["png_bytes"] for m in all_meta)
    report = dict(schema=SCHEMA, dataset_id=dataset_id, policy_hash=policy_hash,
                  status="completed" if complete else "pilot", finished=now(), videos=len(all_meta),
                  train_videos=sum(m["split"]=="train" for m in all_meta), val_videos=sum(m["split"]=="val" for m in all_meta),
                  frames=total, label_counts={s:sum(m["label_counts"][s] for m in all_meta) for s in ["positive","unknown","ignore","negative"]},
                  png_bytes=size, estimated_full_png_bytes=size/max(total,1)*23690,
                  elapsed_sec=time.perf_counter()-start, free_disk_bytes=shutil.disk_usage(args.dataset).free,
                  split_hashes={s:sha(args.dataset/"splits"/f"{s}.jsonl") for s in ["train","val"]},
                  decode_version=av.__version__, image_encoder_version=cv2.__version__, source_videos=entries,
                  source_audit_sha256=sha(source),
                  frame_storage="shared immutable originals" if (args.dataset/"frames").is_symlink() else "dataset-local originals")
    save(manifest_path, report)
    print(json.dumps({k:v for k,v in report.items() if k!="source_videos"}), flush=True)


if __name__ == "__main__":
    main()
