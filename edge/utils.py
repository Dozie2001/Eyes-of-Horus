"""
Utility functions for StangWatch.
These are stateless operations — no classes needed, just input in, output out.
"""

import cv2
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def save_snapshot(frame, path="data/snapshot.jpg"):
    """
    Save a frame as a JPEG image file.

    Args:
        frame: numpy array from Camera.read_frame()
        path: where to save the file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, frame)
    print(f"Snapshot saved to: {path}")


def draw_boxes(frame, detections):
    """
    Draw bounding boxes on a frame for each detection.
    Returns a new frame with boxes drawn (does not modify the original).

    Args:
        frame: numpy array from Camera.read_frame()
        detections: list of dicts from Detector.detect_people()

    Returns:
        new frame with green boxes and labels drawn on it
    """
    # Make a copy so we don't modify the original frame
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        label = f"{det['label']} {conf}"

        # Green box around the person
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label above the box
        cv2.putText(
            annotated, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    return annotated


def filter_overlapping(detections, iou_threshold=0.5):
    """
    Remove duplicate detections of the same person.
    If two bounding boxes overlap more than iou_threshold,
    keep only the one with higher confidence.

    Args:
        detections: list of detection dicts (each has "bbox" and "confidence")
        iou_threshold: overlap ratio above which two boxes are considered duplicates.
            0.5 = boxes overlap by 50% or more → same person.

    Returns:
        filtered list with duplicates removed
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence (highest first) — we keep the most confident detection
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    for det in sorted_dets:
        # Check if this box overlaps too much with any box we're already keeping
        is_duplicate = False
        for kept in keep:
            if _iou(det["bbox"], kept["bbox"]) > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep.append(det)

    return keep


def _iou(box1, box2):
    """
    Calculate Intersection over Union between two [x1, y1, x2, y2] boxes.
    Returns 0.0 (no overlap) to 1.0 (identical boxes).
    """
    # Find the overlap rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Area of overlap
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Area of each box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union = both areas minus the overlap (so we don't count it twice)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def write_clip_ffmpeg(path, frames, source_fps=15, output_fps=15):
    """
    Write frames to an H.264 MP4 using ffmpeg via stdin pipe.

    Produces Telegram-compatible video with moov atom at the front
    (faststart) for inline playback and streaming.

    Args:
        path: output .mp4 file path
        frames: list of numpy BGR frames (from OpenCV)
        source_fps: FPS of the source frames
        output_fps: desired output FPS (frames are subsampled if source > output)

    Returns:
        True on success, False on failure
    """
    if not frames:
        return False

    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found, falling back to OpenCV clip writer")
        return _write_clip_opencv_fallback(path, frames, source_fps, output_fps)

    tmp_path = path.replace(".mp4", ".tmp.mp4")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        output_fps = min(source_fps, output_fps)
        step = max(1, round(source_fps / output_fps))

        h, w = frames[0].shape[:2]

        cmd = [
            "ffmpeg",
            "-y",                          # overwrite without asking
            "-f", "rawvideo",              # input format: raw pixels
            "-pix_fmt", "bgr24",           # OpenCV BGR byte order
            "-s", f"{w}x{h}",             # frame dimensions
            "-r", str(output_fps),         # input framerate
            "-i", "pipe:0",               # read from stdin
            "-c:v", "libx264",            # H.264 codec
            "-preset", "fast",            # encoding speed/quality tradeoff
            "-crf", "23",                 # quality (lower = better, 23 = default)
            "-pix_fmt", "yuv420p",        # output pixel format (max compatibility)
            "-movflags", "+faststart",    # moov atom at front for streaming
            "-an",                         # no audio track
            "-loglevel", "error",         # only show errors
            tmp_path,
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Build raw byte payload, then send all at once via communicate()
        raw_chunks = []
        for i, frame in enumerate(frames):
            if i % step == 0:
                if not frame.flags['C_CONTIGUOUS']:
                    frame = frame.copy()
                raw_chunks.append(frame.tobytes())

        raw_data = b"".join(raw_chunks)
        _, stderr = proc.communicate(input=raw_data, timeout=60)

        if proc.returncode != 0:
            logger.error(f"ffmpeg failed (rc={proc.returncode}): {stderr.decode().strip()}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

        os.rename(tmp_path, path)
        written = len([i for i in range(len(frames)) if i % step == 0])
        logger.debug(f"  CLIP: Saved {path} ({written} frames, H.264)")
        return True

    except Exception as e:
        logger.error(f"  CLIP: Failed to write {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def _write_clip_opencv_fallback(path, frames, source_fps=15, output_fps=15):
    """Fallback clip writer using OpenCV if ffmpeg is not installed."""
    tmp_path = path.replace(".mp4", ".tmp.mp4")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        output_fps = min(source_fps, output_fps)
        step = max(1, round(source_fps / output_fps))

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, output_fps, (w, h))

        for i, frame in enumerate(frames):
            if i % step == 0:
                writer.write(frame)

        writer.release()
        os.rename(tmp_path, path)
        logger.debug(f"  CLIP: Saved {path} ({len(frames)} frames, OpenCV fallback)")
        return True

    except Exception as e:
        logger.error(f"  CLIP: Fallback write failed {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
