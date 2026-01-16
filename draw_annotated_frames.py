#!/usr/bin/env python3
"""
Render images with bounding boxes drawn from annotations created by
`_bbox_video_widget.py`. This script reloads each frame directly from the
original video (important when crops were saved instead of full frames)
and draws class-colored rectangles, optionally with labels.

Usage examples:
  - 単一動画を処理（注釈フォルダは自動で video パスの拡張子除去先を使用）
      python draw_annotated_frames.py --video /path/to/video.mp4

  - 単一動画を処理（注釈フォルダを明示指定）
      python draw_annotated_frames.py --video /path/to/video.mp4 --ann_dir /path/to/video

  - ディレクトリ内の全動画を一括処理（各動画に対応する注釈フォルダが存在する場合のみ）
      python draw_annotated_frames.py --video /path/to/dir
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import yaml
import numpy as np

try:
    # Prefer napari-video's fast random access reader if available
    from napari_video.napari_video import VideoReaderNP  # type: ignore
    _HAS_VIDEOREADERNP = True
except Exception:
    _HAS_VIDEOREADERNP = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".MP4", ".MOV", ".AVI")


def _list_videos_in_dir(directory: Path) -> List[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS])


def _load_class_names(class_yaml: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if not class_yaml.exists():
        return names
    try:
        with open(class_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("names", {})
        # Keys may come as str or int in YAML; normalize to int
        for k, v in raw.items():
            try:
                names[int(k)] = str(v)
            except Exception:
                continue
    except Exception:
        pass
    return names


def _class_color(class_id: int) -> Tuple[int, int, int]:
    """Return a BGR color tuple for a class id.

    Uses a fixed palette then cycles via HSV if needed.
    """
    palette = [
        (220, 20, 60),   # Crimson
        (255, 140, 0),   # DarkOrange
        (0, 165, 255),   # Orange (BGR)
        (0, 255, 255),   # Yellow
        (34, 139, 34),   # ForestGreen
        (46, 139, 87),   # SeaGreen
        (255, 0, 0),     # Blue (BGR)
        (255, 0, 255),   # Magenta
        (128, 0, 128),   # Purple
        (128, 0, 0),     # Maroon
        (0, 128, 128),   # Teal
        (0, 69, 255),    # DarkOrange (BGR reversed)
    ]
    if class_id < len(palette):
        return palette[class_id]
    # Fallback: simple HSV cycle mapped to BGR
    if not _HAS_CV2:
        # If cv2 unavailable, just cycle palette
        return palette[class_id % len(palette)]
    hue = (class_id * 17) % 180  # OpenCV HSV hue range [0,180)
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _parse_yolo_lines(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse YOLO lines: returns list of (class_id, cx, cy, w, h) normalized."""
    anns: List[Tuple[int, float, float, float, float]] = []
    if not txt_path.exists():
        return anns
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                anns.append((cls, cx, cy, w, h))
            except Exception:
                continue
    return anns


def _extract_frame_index_from_name(name: str) -> Optional[Tuple[int, int]]:
    """Extract frame index and its zero-padding length from a filename stem.

    Strategy: find all digit groups in the stem and pick the longest group;
    if multiple groups have the same max length, choose the last one.
    Returns (frame_index, pad_len) or None if no digits.
    """
    stem = Path(name).stem
    matches = list(re.finditer(r"(\d+)", stem))
    if not matches:
        return None
    # Choose the match with max length; for ties, the later one
    best = None
    best_key = (-1, -1)
    for m in matches:
        key = (len(m.group(1)), m.start())
        if key > best_key:
            best = m
            best_key = key
    assert best is not None
    s = best.group(1)
    return int(s), len(s)


def _iter_annotation_files(ann_dir: Path) -> Iterable[Tuple[Path, int, int]]:
    """Yield (txt_path, frame_idx, pad_len) for YOLO annotation files in ann_dir.

    Skips known non-annotation files like class.yaml.
    """
    for p in sorted(ann_dir.glob("*.txt")):
        if p.name.lower() in {"class.txt"}:
            continue
        # Avoid YAML mistakenly saved as .txt
        if p.name.lower() in {"class.yaml"}:
            continue
        ext = p.suffix.lower()
        if ext != ".txt":
            continue
        info = _extract_frame_index_from_name(p.name)
        if info is None:
            continue
        frame_idx, pad_len = info
        yield p, frame_idx, pad_len


def _draw_annotations_on_frame(
    frame_bgr: np.ndarray,
    anns: List[Tuple[int, float, float, float, float]],
    class_names: Dict[int, str],
    thickness: int = 2,
    draw_label: bool = True,
) -> np.ndarray:
    """Draw rectangles (in-place) on a BGR frame given YOLO-normalized boxes."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required to draw annotations.")
    h, w = frame_bgr.shape[:2]
    for cls, cx, cy, bw, bh in anns:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = _class_color(cls)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
        if draw_label:
            label = class_names.get(cls, str(cls))
            txt = f"{cls}:{label}"
            # Text background
            (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(0, y1 - th - 4)
            tx1 = max(0, x1)
            cv2.rectangle(frame_bgr, (tx1, ty1), (tx1 + tw + 4, ty1 + th + baseline + 4), color, -1)
            cv2.putText(
                frame_bgr,
                txt,
                (tx1 + 2, ty1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return frame_bgr


def _open_video_reader(video_path: Path):
    if _HAS_VIDEOREADERNP:
        return VideoReaderNP(str(video_path))
    if not _HAS_CV2:
        raise RuntimeError("Neither napari-video nor OpenCV available to read video.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def _read_frame(reader, index: int) -> np.ndarray:
    if _HAS_VIDEOREADERNP and hasattr(reader, "__getitem__"):
        # VideoReaderNP returns RGB; convert to BGR for cv2 drawing
        frame_rgb = reader[index]
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if _HAS_CV2 else frame_rgb
    # Fallback: cv2.VideoCapture
    cap = reader
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {index}")
    # cv2 returns BGR already
    return frame


def _resolve_ann_dir_for_video(video_path: Path, ann_dir_opt: Optional[Path]) -> Path:
    """Resolve the annotation directory for a given video.

    If ann_dir_opt is provided (file mode), use it; otherwise default to
    video path without extension (same behavior as the widget).
    """
    return ann_dir_opt if ann_dir_opt is not None else video_path.with_suffix("")


def _resolve_out_dir(ann_dir: Path, out_dir_opt: Optional[Path], multi: bool, video_stem: str) -> Path:
    """Resolve output directory.

    Default is `ann_dir/<video_stem>_annotated`.
    In multi (directory) mode with a global out_dir, create `<out_dir>/<video_stem>_annotated`.
    """
    if out_dir_opt is None:
        return ann_dir / f"{video_stem}_annotated"
    if multi:
        return out_dir_opt / f"{video_stem}_annotated"
    return out_dir_opt


def _process_single_video(video_path: Path, ann_dir_opt: Optional[Path], out_dir_opt: Optional[Path], thickness: int, draw_label: bool) -> int:
    ann_dir = _resolve_ann_dir_for_video(video_path, ann_dir_opt)
    if not ann_dir.exists():
        print(f"[skip] Annotation dir not found for {video_path.name}: {ann_dir}")
        return 0

    out_dir = _resolve_out_dir(ann_dir, out_dir_opt, multi=False, video_stem=video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names(ann_dir / "class.yaml")
    ann_entries = list(_iter_annotation_files(ann_dir))
    if not ann_entries:
        print(f"[skip] No annotation files in {ann_dir}")
        return 0

    reader = _open_video_reader(video_path)

    # Sort by frame index for consistent processing
    ann_entries.sort(key=lambda x: x[1])

    count = 0
    for txt_path, frame_idx, pad_len in ann_entries:
        anns = _parse_yolo_lines(txt_path)
        if not anns:
            continue
        frame_bgr = _read_frame(reader, frame_idx)
        frame_bgr = _draw_annotations_on_frame(frame_bgr, anns, class_names, thickness=thickness, draw_label=draw_label)
        # Save using original stem for clarity; avoids assuming specific prefixes
        out_name = f"{txt_path.stem}_annotated.png"
        out_path = out_dir / out_name
        if _HAS_CV2:
            cv2.imwrite(str(out_path), frame_bgr)
        else:
            try:
                from PIL import Image
                rgb = frame_bgr if frame_bgr.ndim == 2 else frame_bgr[:, :, ::-1]
                Image.fromarray(rgb).save(out_path)
            except Exception as e:
                raise SystemExit(f"Saving image failed (need cv2 or PIL): {e}")
        count += 1

    if _HAS_CV2 and not _HAS_VIDEOREADERNP and hasattr(reader, "release"):
        reader.release()

    print(f"Saved {count} annotated images to: {out_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Draw annotations onto frames and save annotated images.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file or a directory that contains videos")
    parser.add_argument("--ann_dir", type=str, default=None, help="Annotation directory for the video (optional in file mode)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for annotated images (default: <ann_dir>/<video_name>_annotated)")
    parser.add_argument("--thickness", type=int, default=2, help="Rectangle line thickness (px)")
    parser.add_argument("--no-label", action="store_true", help="Do not draw class labels")

    args = parser.parse_args()

    video_path = Path(args.video)
    ann_dir_opt = Path(args.ann_dir) if args.ann_dir else None
    out_dir_opt = Path(args.out_dir) if args.out_dir else None

    total = 0
    if video_path.is_file():
        total += _process_single_video(
            video_path,
            ann_dir_opt=ann_dir_opt,
            out_dir_opt=out_dir_opt,
            thickness=args.thickness,
            draw_label=(not args.no_label),
        )
    elif video_path.is_dir():
        if ann_dir_opt is not None:
            print("[warn] --ann_dir is ignored in directory mode; each video uses <video_path without extension> as its annotation folder.")
        videos = _list_videos_in_dir(video_path)
        if not videos:
            raise SystemExit(f"No videos found in directory: {video_path}")
        for vp in videos:
            ann_dir = _resolve_ann_dir_for_video(vp, None)
            if not ann_dir.exists():
                print(f"[skip] Annotation dir not found for {vp.name}: {ann_dir}")
                continue
            out_dir_this = _resolve_out_dir(ann_dir, out_dir_opt, multi=True, video_stem=vp.stem)
            out_dir_this.mkdir(parents=True, exist_ok=True)
            total += _process_single_video(
                vp,
                ann_dir_opt=ann_dir,
                out_dir_opt=out_dir_this,
                thickness=args.thickness,
                draw_label=(not args.no_label),
            )
    else:
        raise SystemExit(f"--video path does not exist: {video_path}")

    if total == 0:
        print("No annotated images were generated.")


if __name__ == "__main__":
    main()
