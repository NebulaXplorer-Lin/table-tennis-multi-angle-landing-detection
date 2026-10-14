"""Create the full positive-labelled YOLO corpus from pp_train_data.

Every training video and every frame that has a ball box is included.  Frames
without a ball annotation are deliberately not converted into YOLO negatives:
the supplied annotations do not establish that those frames are exhaustively
labelled.  The public set has landing points, not ball boxes, and is therefore
kept out of detector training entirely.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/pp_train_data"
DEST = ROOT / "out/datasets/yolo_ball_full_v1"
SPLIT = ROOT / "experiments/splits/yolo_ball_full_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if DEST.exists() or SPLIT.exists():
        raise RuntimeError(f"refusing to overwrite existing dataset/split: {DEST} / {SPLIT}")
    image_dir, label_dir = DEST / "images" / "train", DEST / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    SPLIT.mkdir(parents=True)
    records, audit = [], []
    for annotation in sorted((SOURCE / "annotations").glob("*/*.json")):
        doc = json.loads(annotation.read_text())
        video = SOURCE / doc["video_path"]
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != (doc["frame_width"], doc["frame_height"]):
            raise RuntimeError(f"resolution mismatch: {video}")
        by_frame = {f["frame_id"]: f for f in doc["frames"]}
        kept = unknown = index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            boxes = [b for a in by_frame.get(index, {}).get("annotations", []) if a["label"] == "ball" for b in a["bbox_xyxy"]]
            if not boxes:
                unknown += 1
                index += 1
                continue
            name = f"{annotation.parent.name}_{annotation.stem}_{index:06d}"
            lines, clipped = [], []
            for x1, y1, x2, y2 in boxes:
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(width), x2), min(float(height), y2)
                if x2 <= x1 or y2 <= y1:
                    raise RuntimeError(f"invalid box {video} frame={index}")
                clipped.append([x1, y1, x2, y2])
                lines.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
            image_path = image_dir / f"{name}.jpg"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"could not write {image_path}")
            (label_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
            records.append({"group": annotation.parent.name, "video": str(video.relative_to(ROOT)), "frame": index,
                            "width": width, "height": height, "boxes": clipped, "image": str(image_path.relative_to(ROOT))})
            kept += 1
            index += 1
        cap.release()
        if index != doc["num_frames"]:
            raise RuntimeError(f"frame count mismatch: {video}: {index} != {doc['num_frames']}")
        audit.append({"group": annotation.parent.name, "video": str(video.relative_to(ROOT)), "frames": index,
                      "positive_frames": kept, "unlabelled_or_no_ball_frames": unknown,
                      "annotation_sha256": digest(annotation), "video_sha256": digest(video)})
    (SPLIT / "train.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (SPLIT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    # The trainer needs a validation source.  This is explicitly the same full
    # train set and is used for loss/learning diagnostics only, never reported
    # as generalisation; the public end-to-end score is the external validation.
    (DEST / "data.yaml").write_text(
        f"path: {DEST}\ntrain: images/train\nval: images/train\nnames:\n  0: ball\n")
    print(json.dumps({"videos": len(audit), "positive_frames": len(records), "total_frames": sum(x["frames"] for x in audit)}, indent=2))


if __name__ == "__main__":
    main()
