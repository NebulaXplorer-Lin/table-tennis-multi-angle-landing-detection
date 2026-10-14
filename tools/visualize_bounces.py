"""Overlay final bounce predictions and GT; does not infer per-frame ball positions."""
import argparse
import json
import subprocess
import shutil
from pathlib import Path

import cv2


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="v000001")
    parser.add_argument("--data", type=Path, default=Path("/participant/pingpang/public_data"))
    parser.add_argument("--predictions", type=Path, default=Path("/participant/pingpang/scripts/out_check_solution/predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/participant/output/error_review"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    meta = next(v for v in manifest["videos"] if v["video_id"] == args.video_id)
    source = args.data / meta["path"]
    gt = [r for r in read_jsonl(args.data / "gt_public.jsonl") if r["video_id"] == args.video_id]
    pred = [r for r in read_jsonl(args.predictions) if r["video_id"] == args.video_id]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / (args.video_id + "_review.mp4")
    assert not output.exists(), f"Output already exists: {output}"
    cap = cv2.VideoCapture(str(source))
    assert cap.isOpened(), f"Cannot open video: {source}"
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(meta["fps"])
    # 5x slower playback. Marks remain visible around the event; coordinates
    # are the event position, NOT the ball position on the current frame.
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-n",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps / 5), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    frame_id = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            cv2.rectangle(image, (0, 0), (width, 130), (25, 25, 25), -1)
            cv2.putText(image, f"{args.video_id} | original frame {frame_id} | 0.2x speed", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2)
            cv2.putText(image, "GREEN=GT  RED=PRED  GRAY=IGNORED | labels show event frame", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
            cv2.putText(image, "Markers stay +/-2 frames; they are NOT current ball tracking.", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, .65, (220, 220, 220), 1)
            for records, is_gt in ((gt, True), (pred, False)):
                for record in records:
                    if abs(record["frame_id"] - frame_id) > 2:
                        continue
                    valid = record.get("valid", True)
                    color = (0, 255, 0) if is_gt and valid else ((160, 160, 160) if is_gt else (0, 0, 255))
                    x, y = round(record["x"]), round(record["y"])
                    label = "GT" if is_gt and valid else ("IGNORE" if is_gt else "PRED")
                    if is_gt:
                        cv2.circle(image, (x, y), 18, color, 3)
                    else:
                        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 28, 3)
                    shift = -28 if is_gt else 38
                    cv2.putText(image, f"{label} f={record['frame_id']}", (max(0, x-60), max(20, min(height-10,y+shift))), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
            process.stdin.write(image.tobytes())
            frame_id += 1
    finally:
        cap.release()
        process.stdin.close()
    assert process.wait() == 0, "FFmpeg encoding failed"
    assert frame_id == meta["n_frames"], f"Read {frame_id} frames; expected {meta['n_frames']}"
    event_list = dict(video_id=args.video_id, source=str(source), predictions=pred, ground_truth=gt,
                      note="Overlaid event points only; this video cannot alone diagnose detector versus event-rule errors.")
    (args.output / (args.video_id + "_events.json")).write_text(json.dumps(event_list, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUCCESS: {frame_id} frames, {len(pred)} predictions, {len(gt)} GT records")
    print(f"Review video: {output}")


if __name__ == "__main__":
    main()
