"""원판 학습 frame sampling과 YOLO bbox 라벨 변환."""

from numbers import Integral, Real
from pathlib import Path
from tempfile import NamedTemporaryFile
import math

import cv2
import numpy as np
import pandas as pd


SAMPLE_PHASES = (
    "Set-up",
    "First Pull",
    "Second Pull",
    "Drop Under & Catch",
    "Recovery",
)

_PHASE_COLUMNS = ("video_id", "frame_index", "phase")
_SAMPLE_COLUMNS = (
    "video_id",
    "frame_index",
    "analysis_phase",
    "sampling_phase",
    "setup_reference_override",
)


def sample_phase_frames(
    phase_by_frame: pd.DataFrame,
    *,
    sample_count: int,
) -> pd.DataFrame:
    """다섯 구간에 고르게 분산된 중복 없는 frame을 선택한다."""
    missing_columns = [
        column for column in _PHASE_COLUMNS if column not in phase_by_frame.columns
    ]
    if missing_columns:
        raise ValueError(f"phase columns missing: {', '.join(missing_columns)}")
    if phase_by_frame.empty:
        raise ValueError("phase_by_frame is empty")
    if phase_by_frame["video_id"].nunique() != 1:
        raise ValueError("phase_by_frame must contain exactly one video_id")
    if phase_by_frame["frame_index"].duplicated().any():
        raise ValueError("phase_by_frame frame_index must be unique")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or sample_count < len(SAMPLE_PHASES)
        or sample_count > len(phase_by_frame)
    ):
        raise ValueError(
            f"sample_count must be {len(SAMPLE_PHASES)}..{len(phase_by_frame)}"
        )

    sampled_pool = phase_by_frame.loc[:, _PHASE_COLUMNS].copy()
    numeric_frame_indices = pd.to_numeric(
        sampled_pool["frame_index"], errors="raise"
    )
    if not np.isfinite(numeric_frame_indices).all() or not np.equal(
        numeric_frame_indices,
        np.floor(numeric_frame_indices),
    ).all():
        raise ValueError("frame_index must contain finite integers")
    sampled_pool["frame_index"] = numeric_frame_indices.astype(int)
    if sampled_pool["frame_index"].duplicated().any():
        raise ValueError("phase_by_frame frame_index must be unique")
    sampled_pool["analysis_phase"] = sampled_pool["phase"].astype(str)
    unknown_phases = sorted(
        set(sampled_pool["analysis_phase"]) - set(SAMPLE_PHASES)
    )
    if unknown_phases:
        raise ValueError("unknown phases: " + ", ".join(unknown_phases))

    sampled_pool["sampling_phase"] = sampled_pool["analysis_phase"]
    sampled_pool["setup_reference_override"] = False
    if not (sampled_pool["sampling_phase"] == "Set-up").any():
        frame_zero = sampled_pool["frame_index"] == 0
        if frame_zero.sum() != 1:
            raise ValueError("Set-up이 비어 있으면 frame 0이 필요합니다.")
        sampled_pool.loc[frame_zero, "sampling_phase"] = "Set-up"
        sampled_pool.loc[frame_zero, "setup_reference_override"] = True

    capacities = {
        phase: int((sampled_pool["sampling_phase"] == phase).sum())
        for phase in SAMPLE_PHASES
    }
    empty_phases = [phase for phase, capacity in capacities.items() if capacity == 0]
    if empty_phases:
        raise ValueError("frames missing for phases: " + ", ".join(empty_phases))

    counts = _allocate_phase_counts(sample_count, capacities)
    sampled_rows = []
    for phase in SAMPLE_PHASES:
        candidates = sampled_pool.loc[
            sampled_pool["sampling_phase"] == phase
        ].sort_values("frame_index")
        positions = np.linspace(
            0,
            len(candidates) - 1,
            num=counts[phase],
            dtype=int,
        )
        sampled_rows.append(candidates.iloc[positions])

    sampled = pd.concat(sampled_rows, ignore_index=True)
    sampled = sampled.sort_values("frame_index").reset_index(drop=True)
    return sampled.loc[:, _SAMPLE_COLUMNS]


def opencv_bbox_to_yolo(
    bbox: tuple[Real, Real, Real, Real],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """OpenCV (x, y, w, h)를 YOLO 정규화 bbox로 변환한다."""
    x, y, width, height = _validate_opencv_bbox(
        bbox,
        image_width=image_width,
        image_height=image_height,
    )
    return (
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )


def yolo_bbox_to_opencv(
    bbox: tuple[Real, Real, Real, Real],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """contact sheet 확인을 위해 YOLO bbox를 pixel 좌표로 복원한다."""
    _validate_image_size(image_width, image_height)
    if len(bbox) != 4 or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in bbox
    ):
        raise ValueError("YOLO bbox must contain four finite numbers")
    center_x, center_y, width, height = map(float, bbox)
    if not (
        0 <= center_x <= 1
        and 0 <= center_y <= 1
        and 0 < width <= 1
        and 0 < height <= 1
    ):
        raise ValueError("YOLO bbox values must be normalized to 0..1")

    pixel_width = round(width * image_width)
    pixel_height = round(height * image_height)
    x = round((center_x - width / 2) * image_width)
    y = round((center_y - height / 2) * image_height)
    return tuple(
        round(value)
        for value in _validate_opencv_bbox(
            (x, y, pixel_width, pixel_height),
            image_width=image_width,
            image_height=image_height,
        )
    )


def save_plate_image(
    image_path: str | Path,
    frame: np.ndarray,
    *,
    overwrite: bool = False,
) -> str:
    """추출 frame을 저장하되 기존 이미지는 기본적으로 보호한다."""
    image_path = Path(image_path)
    image_exists = image_path.exists()
    if image_exists and not overwrite:
        return "skipped"
    if frame is None or frame.size == 0:
        raise ValueError("frame is empty")

    action = "overwritten" if image_exists else "created"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=image_path.parent,
        suffix=image_path.suffix,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        if not cv2.imwrite(str(temporary_path), frame):
            raise OSError(f"OpenCV could not write image: {image_path}")
        temporary_path.replace(image_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return action


def save_yolo_plate_label(
    label_path: str | Path,
    bbox: tuple[Real, Real, Real, Real],
    *,
    image_width: int,
    image_height: int,
    overwrite: bool = False,
) -> str:
    """단일 plate bbox를 YOLO txt로 저장하고 기존 라벨을 보호한다."""
    yolo_bbox = opencv_bbox_to_yolo(
        bbox,
        image_width=image_width,
        image_height=image_height,
    )
    label_path = Path(label_path)
    label_exists = label_path.exists()
    if label_exists and not overwrite:
        return "skipped"

    action = "overwritten" if label_exists else "created"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    line = "0 " + " ".join(f"{value:.6f}" for value in yolo_bbox) + "\n"
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=label_path.parent,
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(line)
    try:
        temporary_path.replace(label_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return action


def _allocate_phase_counts(
    sample_count: int,
    capacities: dict[str, int],
) -> dict[str, int]:
    base_count, remainder = divmod(sample_count, len(SAMPLE_PHASES))
    counts = {
        phase: min(base_count + (index < remainder), capacities[phase])
        for index, phase in enumerate(SAMPLE_PHASES)
    }
    remaining = sample_count - sum(counts.values())
    while remaining:
        progressed = False
        for phase in SAMPLE_PHASES:
            if counts[phase] >= capacities[phase]:
                continue
            counts[phase] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("sample_count exceeds available unique frames")
    return counts


def _validate_opencv_bbox(
    bbox: tuple[Real, Real, Real, Real],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    _validate_image_size(image_width, image_height)
    if len(bbox) != 4 or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in bbox
    ):
        raise ValueError("OpenCV bbox must contain four finite numbers")

    x, y, width, height = map(float, bbox)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("bbox must have non-negative origin and positive size")
    if x + width > image_width or y + height > image_height:
        raise ValueError("bbox extends outside the image")
    return x, y, width, height


def _validate_image_size(image_width: int, image_height: int) -> None:
    if (
        isinstance(image_width, bool)
        or isinstance(image_height, bool)
        or not isinstance(image_width, Integral)
        or not isinstance(image_height, Integral)
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("image dimensions must be positive integers")
