"""Video properties and deterministic frame access."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


VideoProperties = dict[str, float | int | str | None]
VideoFrame = tuple[int, float, np.ndarray]


def read_video_properties(video_path: str | Path) -> VideoProperties:
    """Return the decoded video properties used by the project manifest."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return {
                "fps": None,
                "frame_count": None,
                "duration_sec": None,
                "width": None,
                "height": None,
                "read_status": "error",
            }

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = 0
        while True:
            success, _ = capture.read()
            if not success:
                break
            frame_count += 1
    finally:
        capture.release()

    duration_sec = frame_count / fps if fps > 0 else None
    return {
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_sec": round(duration_sec, 6) if duration_sec is not None else None,
        "width": width,
        "height": height,
        "read_status": "ok",
    }


def iter_video_frames(
    video_path: str | Path,
    *,
    expected_frame_count: int | None = None,
) -> Iterator[VideoFrame]:
    """Yield ``(frame_index, timestamp_sec, BGR frame)`` without caching frames."""
    if expected_frame_count is not None and expected_frame_count < 0:
        raise ValueError("expected_frame_count must be non-negative")

    capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError(f"Video FPS must be positive: {video_path}")

        while True:
            success, frame = capture.read()
            if not success:
                if frame_index == 0:
                    raise RuntimeError(f"Could not decode first frame: {video_path}")
                break
            yield frame_index, frame_index / fps, frame
            frame_index += 1
    finally:
        capture.release()

    if expected_frame_count is not None and frame_index < expected_frame_count:
        raise RuntimeError(
            "Video ended before expected frame count: "
            f"expected={expected_frame_count}, actual={frame_index}, "
            f"last_success_index={frame_index - 1}"
        )


def read_video_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Read one zero-based frame, favoring reliable sequential decoding."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    for current_index, _, frame in iter_video_frames(video_path):
        if current_index == frame_index:
            return frame

    raise IndexError(f"Frame index out of range: {frame_index}")
