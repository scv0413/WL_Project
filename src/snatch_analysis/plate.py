"""원판 학습 frame sampling과 YOLO bbox 라벨 변환."""

from numbers import Integral, Real
from pathlib import Path
from tempfile import NamedTemporaryFile
import math
from typing import Any

import cv2
import numpy as np
import pandas as pd

from snatch_analysis.pose import _to_numpy, _write_csv_atomically


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
_TIME_BIN_COUNT = 5
_U10_SAMPLES_PER_VIDEO = 20
_TIME_BIN_SAMPLE_COLUMNS = ("video_id", "frame_index", "sampling_bin")
U10_PLATE_SELECTION_COLUMNS = (
    "video_id",
    "person_id",
    "dataset_role",
    "selection_reason",
)
PLATE_CANDIDATE_COLUMNS = (
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "candidate_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_confidence",
    "center_x_px",
    "center_y_px",
    "center_x_normalized",
    "center_y_normalized",
)
PLATE_TRACK_COLUMNS = (*PLATE_CANDIDATE_COLUMNS, "status", "reason")
PILOT_FRAMES_PER_VIDEO = 10
PILOT_GROUND_TRUTH_COLUMNS = (
    "video_id",
    "frame_index",
    "plate_present",
    "plate_bbox_x",
    "plate_bbox_y",
    "plate_bbox_width",
    "plate_bbox_height",
    "lifter_bbox_x",
    "lifter_bbox_y",
    "lifter_bbox_width",
    "lifter_bbox_height",
    "annotator",
    "notes",
    "na_reason",
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


def sample_time_bin_frames(
    video_id: str,
    frame_count: int,
    *,
    sample_count: int = 20,
) -> pd.DataFrame:
    """정확한 phase 라벨 없이 다섯 시간 구간에서 frame을 선택한다."""
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must not be empty")
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral):
        raise ValueError("frame_count must be an integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise ValueError("sample_count must be an integer")
    if sample_count != _U10_SAMPLES_PER_VIDEO:
        raise ValueError("sample_count must be exactly 20 for U10")
    if frame_count < sample_count:
        raise ValueError("frame_count must be at least sample_count")

    rows = []
    per_bin = sample_count // _TIME_BIN_COUNT
    for sampling_bin, frame_indices in enumerate(
        np.array_split(np.arange(frame_count), _TIME_BIN_COUNT),
        start=1,
    ):
        positions = np.linspace(
            0,
            len(frame_indices) - 1,
            num=per_bin,
            dtype=int,
        )
        rows.extend(
            {
                "video_id": video_id,
                "frame_index": int(frame_indices[position]),
                "sampling_bin": sampling_bin,
            }
            for position in positions
        )
    return pd.DataFrame(rows, columns=_TIME_BIN_SAMPLE_COLUMNS).sort_values(
        "frame_index",
        ignore_index=True,
    )


def validate_u10_plate_selection(
    selection: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    baseline_video_id: str,
) -> pd.DataFrame:
    """U10 신규 영상 6개의 역할과 person 단위 분리를 검증한다."""
    if tuple(selection.columns) != U10_PLATE_SELECTION_COLUMNS:
        raise ValueError("U10 plate selection schema가 일치하지 않습니다.")
    if len(selection) != 6 or selection["video_id"].duplicated().any():
        raise ValueError("U10에는 서로 다른 신규 영상 6개가 필요합니다.")
    if selection["dataset_role"].value_counts().to_dict() != {
        "train": 4,
        "validation": 2,
    }:
        raise ValueError(
            "U10 dataset_role은 train 4개와 validation 2개여야 합니다."
        )
    if baseline_video_id in set(selection["video_id"].astype(str)):
        raise ValueError("B00 baseline은 신규 영상 6개에 포함할 수 없습니다.")
    reasons = selection["selection_reason"]
    if reasons.isna().any() or reasons.astype(str).str.strip().eq("").any():
        raise ValueError("각 U10 영상에는 selection_reason이 필요합니다.")

    required_columns = {
        "video_id",
        "person_id",
        "split_role",
        "read_status",
        "has_plate",
    }
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        raise ValueError("Manifest columns missing: " + ", ".join(missing_columns))
    if manifest["video_id"].duplicated().any():
        raise ValueError("Manifest의 video_id가 중복되었습니다.")

    selected = selection.merge(
        manifest.loc[:, sorted(required_columns)],
        on=["video_id", "person_id"],
        how="left",
        validate="one_to_one",
    )
    if selected["split_role"].isna().any():
        raise ValueError("Manifest에 없는 video_id/person_id 선택입니다.")
    if not selected["split_role"].eq("training_development").all():
        raise ValueError("U10에는 training_development 영상만 사용할 수 있습니다.")
    if not selected["read_status"].eq("ok").all():
        raise ValueError("U10 영상의 read_status는 모두 ok여야 합니다.")
    if not selected["has_plate"].eq("yes").all():
        raise ValueError("선택한 U10 영상의 has_plate를 yes로 확정하세요.")
    if selection["person_id"].nunique() < 4:
        raise ValueError("U10 신규 영상은 최소 네 사람을 포함해야 합니다.")
    roles_per_person = selection.groupby("person_id")["dataset_role"].nunique()
    if (roles_per_person > 1).any():
        raise ValueError("같은 person_id를 train과 validation에 나눌 수 없습니다.")
    return selection.copy().reset_index(drop=True)


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


def cancel_plate_label(
    label_path: str | Path,
    *,
    overwrite: bool = False,
) -> str:
    """재라벨링 취소 시 기존 bbox가 정답처럼 남지 않게 한다."""
    label_path = Path(label_path)
    if overwrite and label_path.exists():
        label_path.unlink()
        return "plate_label_removed"
    return "plate_label_cancelled"


def save_u10_plate_dataset_yaml(
    dataset_root: str | Path,
    baseline_dataset_root: str | Path,
    video_split: pd.DataFrame,
    sample_frames: pd.DataFrame,
) -> dict[str, int]:
    """현재 U10 split·sample·파일이 모두 일치할 때만 YAML을 쓴다."""
    dataset_root = Path(dataset_root)
    baseline_dataset_root = Path(baseline_dataset_root)
    yaml_path = dataset_root / "dataset.yaml"
    yaml_path.unlink(missing_ok=True)

    split_columns = {"video_id", "dataset_role"}
    sample_columns = {
        "video_id",
        "frame_index",
        "dataset_role",
        "image_filename",
        "label_filename",
    }
    if not split_columns.issubset(video_split.columns) or not sample_columns.issubset(
        sample_frames.columns
    ):
        raise ValueError("video_split and sample_frames columns are incomplete")
    if len(video_split) != 6 or video_split["video_id"].duplicated().any():
        raise ValueError("video_split and sample_frames must describe six videos")
    if video_split["dataset_role"].value_counts().to_dict() != {
        "train": 4,
        "validation": 2,
    }:
        raise ValueError("video_split and sample_frames roles do not match U10")
    if len(sample_frames) != 120 or sample_frames.duplicated(
        ["video_id", "frame_index"]
    ).any():
        raise ValueError("video_split and sample_frames must contain 120 unique frames")

    split_roles = video_split.set_index("video_id")["dataset_role"]
    if set(sample_frames["video_id"].astype(str)) != set(split_roles.index.astype(str)):
        raise ValueError("video_split and sample_frames video IDs do not match")
    expected_sample_roles = sample_frames["video_id"].map(split_roles)
    if not expected_sample_roles.eq(sample_frames["dataset_role"]).all():
        raise ValueError("video_split and sample_frames roles do not match")
    if not sample_frames.groupby("video_id").size().eq(20).all():
        raise ValueError("video_split and sample_frames need 20 frames per video")
    if sample_frames["image_filename"].duplicated().any() or sample_frames[
        "label_filename"
    ].duplicated().any():
        raise ValueError("video_split and sample_frames filenames must be unique")

    frame_indices = pd.to_numeric(sample_frames["frame_index"], errors="coerce")
    if (
        frame_indices.isna().any()
        or not np.isfinite(frame_indices).all()
        or not np.equal(frame_indices, np.floor(frame_indices)).all()
        or frame_indices.lt(0).any()
    ):
        raise ValueError("video_split and sample_frames frame indices are invalid")
    expected_image_names = [
        f"{video_id}_frame_{int(frame_index):04d}.jpg"
        for video_id, frame_index in zip(
            sample_frames["video_id"], frame_indices, strict=True
        )
    ]
    expected_label_names = [
        Path(image_name).with_suffix(".txt").name
        for image_name in expected_image_names
    ]
    if not sample_frames["image_filename"].eq(expected_image_names).all() or not sample_frames[
        "label_filename"
    ].eq(expected_label_names).all():
        raise ValueError("sample filenames must match video_id and frame_index")

    image_stems = sample_frames["image_filename"].map(lambda name: Path(name).stem)
    label_stems = sample_frames["label_filename"].map(lambda name: Path(name).stem)
    if not image_stems.eq(label_stems).all():
        raise ValueError("video_split and sample_frames image-label names do not match")

    baseline_image_stems = {
        path.stem for path in (baseline_dataset_root / "images").glob("*.jpg")
    }
    baseline_label_stems = {
        path.stem for path in (baseline_dataset_root / "labels").glob("*.txt")
    }
    if len(baseline_image_stems) != 30 or baseline_image_stems != baseline_label_stems:
        raise ValueError("baseline must contain exactly 30 image-label pairs")

    expected_stems_by_role = {
        "train": baseline_image_stems
        | set(image_stems[sample_frames["dataset_role"].eq("train")]),
        "validation": set(
            image_stems[sample_frames["dataset_role"].eq("validation")]
        ),
    }

    counts = {}
    mismatches = []
    for role, expected_stems in expected_stems_by_role.items():
        image_root = dataset_root / "images" / role
        label_root = dataset_root / "labels" / role
        actual_image_paths = {
            path.relative_to(image_root).as_posix()
            for path in image_root.rglob("*")
            if path.is_file()
        }
        actual_label_paths = {
            path.relative_to(label_root).as_posix()
            for path in label_root.rglob("*")
            if path.is_file()
        }
        expected_image_paths = {f"{stem}.jpg" for stem in expected_stems}
        expected_label_paths = {f"{stem}.txt" for stem in expected_stems}
        counts[f"{role}_images"] = len(actual_image_paths)
        counts[f"{role}_labels"] = len(actual_label_paths)
        if (
            actual_image_paths != expected_image_paths
            or actual_label_paths != expected_label_paths
        ):
            mismatches.append(role)

    if mismatches:
        raise ValueError(
            "dataset files do not match current split: " + ", ".join(mismatches)
        )

    for role, expected_stems in expected_stems_by_role.items():
        for stem in expected_stems:
            label_path = dataset_root / "labels" / role / f"{stem}.txt"
            tokens = label_path.read_text(encoding="utf-8").strip().split()
            if len(tokens) != 5 or tokens[0] != "0":
                raise ValueError(f"invalid plate label: {label_path}")
            try:
                center_x, center_y, width, height = (
                    float(value) for value in tokens[1:]
                )
            except ValueError as error:
                raise ValueError(f"invalid plate label: {label_path}") from error
            values = (center_x, center_y, width, height)
            if not all(np.isfinite(value) for value in values) or not (
                0 <= center_x <= 1
                and 0 <= center_y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
                and center_x - width / 2 >= 0
                and center_x + width / 2 <= 1
                and center_y - height / 2 >= 0
                and center_y + height / 2 <= 1
            ):
                raise ValueError(f"invalid plate label: {label_path}")

    yaml_text = (
        f"path: {dataset_root.resolve()}\n"
        "train: images/train\n"
        "val: images/validation\n"
        "names:\n"
        "  0: plate\n"
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=dataset_root,
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(yaml_text)
    try:
        temporary_path.replace(yaml_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return counts


def plate_result_to_candidates(
    result: Any,
    video_id: str,
    frame_index: int,
    timestamp_seconds: float,
    *,
    image_width: int,
    image_height: int,
) -> pd.DataFrame:
    """한 frame의 Detection bbox 전체를 원판 후보 표로 변환한다."""
    _validate_image_size(image_width, image_height)
    if result.boxes is None:
        return pd.DataFrame(columns=PLATE_CANDIDATE_COLUMNS)

    boxes = _to_numpy(result.boxes.xyxy)
    confidences = _to_numpy(result.boxes.conf)
    if len(boxes) != len(confidences):
        raise ValueError("plate bbox와 confidence 후보 수가 일치하지 않습니다.")

    rows = []
    for candidate_id, (bbox, confidence) in enumerate(
        zip(boxes, confidences, strict=True)
    ):
        if len(bbox) != 4:
            raise ValueError("plate bbox는 네 좌표여야 합니다.")
        x1, y1, x2, y2 = map(float, bbox)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        rows.append(
            {
                "video_id": str(video_id),
                "frame_index": int(frame_index),
                "timestamp_seconds": float(timestamp_seconds),
                "candidate_id": candidate_id,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_confidence": float(confidence),
                "center_x_px": center_x,
                "center_y_px": center_y,
                "center_x_normalized": center_x / image_width,
                "center_y_normalized": center_y / image_height,
            }
        )
    return pd.DataFrame(rows, columns=PLATE_CANDIDATE_COLUMNS)


def infer_plate_video_candidates(
    model: Any,
    *,
    video_path: str | Path,
    video_id: str,
    frame_count: int,
    fps: float,
    image_width: int,
    image_height: int,
    image_size: int,
    confidence: float,
    device: str,
) -> tuple[pd.DataFrame, int]:
    """Ultralytics stream 결과를 원판 후보 표로 변환한다."""
    frame_tables = []
    results = model.predict(
        source=str(video_path),
        stream=True,
        imgsz=image_size,
        conf=confidence,
        device=device,
        verbose=False,
    )
    for frame_index, result in enumerate(results):
        frame_tables.append(
            plate_result_to_candidates(
                result,
                video_id,
                frame_index,
                frame_index / fps,
                image_width=image_width,
                image_height=image_height,
            )
        )
    processed_frame_count = len(frame_tables)
    if processed_frame_count != frame_count:
        raise RuntimeError(
            "Plate processed frame count mismatch: "
            f"expected={frame_count}, actual={processed_frame_count}"
        )
    return (
        pd.concat(frame_tables, ignore_index=True)
        if frame_tables
        else pd.DataFrame(columns=PLATE_CANDIDATE_COLUMNS),
        processed_frame_count,
    )


def save_plate_candidate_cache(
    candidates: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """후보 복합키가 고유한 경우에만 전체 원판 후보를 저장한다."""
    _validate_plate_table(candidates, PLATE_CANDIDATE_COLUMNS, "후보")
    _write_csv_atomically(candidates, output_path)


def save_plate_track(
    track: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """frame당 한 행인 원판 track만 저장한다."""
    _validate_plate_table(track, PLATE_TRACK_COLUMNS, "track")
    _write_csv_atomically(track, output_path)


def build_pilot_ground_truth_template(
    sampled_frames: pd.DataFrame,
    evaluation_split: pd.DataFrame,
) -> pd.DataFrame:
    """평가 영상의 수동 plate·lifter bbox 입력용 빈 표를 만든다."""
    required_sample_columns = {"video_id", "frame_index"}
    missing_sample = sorted(required_sample_columns - set(sampled_frames.columns))
    if missing_sample:
        raise ValueError("sample columns missing: " + ", ".join(missing_sample))
    pilot = _model_selection_pilot(evaluation_split)

    samples = sampled_frames.loc[:, ["video_id", "frame_index"]].copy()
    if samples.duplicated(["video_id", "frame_index"]).any():
        raise ValueError("pilot ground truth에 duplicate video/frame이 있습니다.")
    _validate_pilot_sample_frames(samples, pilot)
    video_metadata = pilot.set_index("video_id")
    samples["plate_present"] = samples["video_id"].map(
        video_metadata["has_plate"]
    )
    for prefix in ("plate", "lifter"):
        for coordinate in ("x", "y", "width", "height"):
            samples[f"{prefix}_bbox_{coordinate}"] = math.nan
    samples["annotator"] = ""
    samples["notes"] = ""
    samples["na_reason"] = np.where(
        samples["plate_present"] == "no",
        "no_plate",
        "",
    )
    return samples.loc[:, PILOT_GROUND_TRUTH_COLUMNS]


def validate_pilot_ground_truth(
    ground_truth: pd.DataFrame,
    evaluation_split: pd.DataFrame,
) -> pd.DataFrame:
    """수동 평가 bbox, frame 범위와 N/A 사유를 검증한다."""
    if tuple(ground_truth.columns) != PILOT_GROUND_TRUTH_COLUMNS:
        raise ValueError("pilot ground truth schema가 일치하지 않습니다.")
    if ground_truth.duplicated(["video_id", "frame_index"]).any():
        raise ValueError("pilot ground truth에 duplicate video/frame이 있습니다.")

    pilot = _model_selection_pilot(evaluation_split)
    _validate_pilot_sample_frames(
        ground_truth.loc[:, ["video_id", "frame_index"]],
        pilot,
    )
    validated = ground_truth.copy()
    for column in ("plate_present", "annotator", "notes", "na_reason"):
        validated[column] = validated[column].fillna("").astype(str).str.strip()
    validated["plate_present"] = validated["plate_present"].str.lower()
    if not set(validated["plate_present"]).issubset({"yes", "no"}):
        raise ValueError("plate_present는 yes/no만 허용합니다.")

    metadata_by_video = pilot.set_index("video_id")
    expected_plate = validated["video_id"].map(metadata_by_video["has_plate"])
    if not validated["plate_present"].reset_index(drop=True).equals(
        expected_plate.reset_index(drop=True)
    ):
        raise ValueError("plate_present가 evaluation split의 has_plate와 다릅니다.")

    for row_index, row in validated.iterrows():
        metadata = metadata_by_video.loc[str(row["video_id"])]
        width = int(metadata["width"])
        height = int(metadata["height"])
        na_reason = str(row["na_reason"])
        _validate_ground_truth_bbox(
            row,
            prefix="plate",
            image_width=width,
            image_height=height,
            required=row["plate_present"] == "yes",
            na_reason=na_reason,
        )
        _validate_ground_truth_bbox(
            row,
            prefix="lifter",
            image_width=width,
            image_height=height,
            required=True,
            na_reason=na_reason,
        )
        validated.at[row_index, "frame_index"] = int(row["frame_index"])
    return validated.loc[:, PILOT_GROUND_TRUTH_COLUMNS]


def save_pilot_ground_truth(
    ground_truth: pd.DataFrame,
    evaluation_split: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """검증을 통과한 pilot 정답만 원자적으로 저장한다."""
    validated = validate_pilot_ground_truth(ground_truth, evaluation_split)
    _write_csv_atomically(validated, output_path)


def select_primary_plate_track(
    candidates: pd.DataFrame,
    *,
    frame_count: int,
    initial_frame_index: int,
    initial_candidate_id: int,
    lifter_centers: pd.DataFrame | None = None,
    fps: float | None = None,
    max_center_shift: float = 4.0,
    max_area_log_change: float = 1.0,
    continuity_ambiguity_margin: float = 0.05,
) -> pd.DataFrame:
    """확정된 원판에서 중심·크기·선수 근접도로 다음 후보를 고른다."""
    if frame_count <= 0:
        raise ValueError("frame_count는 양수여야 합니다.")
    if not 0 <= initial_frame_index < frame_count:
        raise ValueError("initial_frame_index가 frame 범위를 벗어났습니다.")
    _validate_plate_table(candidates, PLATE_CANDIDATE_COLUMNS, "후보")
    first_candidate_frame = int(candidates["frame_index"].min())
    if initial_frame_index != first_candidate_frame:
        raise ValueError("초기 원판 확인은 first candidate frame과 일치해야 합니다.")

    initial = candidates.loc[
        (candidates["frame_index"] == initial_frame_index)
        & (candidates["candidate_id"] == initial_candidate_id)
    ]
    if len(initial) != 1:
        raise ValueError("확정할 초기 원판 후보를 정확히 하나 지정해야 합니다.")
    video_id = str(initial.iloc[0]["video_id"])
    if set(candidates["video_id"].astype(str)) != {video_id}:
        raise ValueError("원판 후보는 하나의 video_id만 포함해야 합니다.")

    lifter_by_frame = _lifter_centers_by_frame(lifter_centers)
    candidates_by_frame = {
        int(frame_index): frame
        for frame_index, frame in candidates.groupby("frame_index", sort=False)
    }
    previous = initial.iloc[0]
    stopped = False
    rows = []
    for frame_index in range(frame_count):
        timestamp_seconds = frame_index / fps if fps else math.nan
        if stopped:
            rows.append(
                _empty_plate_track_row(
                    video_id, frame_index, timestamp_seconds,
                    "confirmation_required", "tracking_stopped_after_ambiguity",
                )
            )
            continue
        if frame_index < initial_frame_index:
            rows.append(
                _empty_plate_track_row(
                    video_id, frame_index, timestamp_seconds, "missing",
                    "before_initial_confirmation",
                )
            )
            continue
        if frame_index == initial_frame_index:
            chosen = previous
        else:
            frame_candidates = candidates_by_frame.get(frame_index)
            if frame_candidates is None or frame_candidates.empty:
                rows.append(
                    _empty_plate_track_row(
                        video_id, frame_index, timestamp_seconds, "missing", "no_detection"
                    )
                )
                continue
            chosen, ambiguous = _select_continuous_plate(
                frame_candidates, previous, lifter_by_frame.get(frame_index),
                max_center_shift=max_center_shift,
                max_area_log_change=max_area_log_change,
                ambiguity_margin=continuity_ambiguity_margin,
            )
            if ambiguous:
                rows.append(
                    _empty_plate_track_row(
                        video_id, frame_index, timestamp_seconds,
                        "confirmation_required", "ambiguous_candidates",
                    )
                )
                stopped = True
                continue
            if chosen is None:
                rows.append(
                    _empty_plate_track_row(
                        video_id, frame_index, timestamp_seconds, "missing",
                        "continuity_mismatch",
                    )
                )
                continue

        previous = chosen
        row = chosen.loc[list(PLATE_CANDIDATE_COLUMNS)].to_dict()
        row.update(status="ok", reason="")
        rows.append(row)
    return pd.DataFrame(rows, columns=PLATE_TRACK_COLUMNS)


def evaluate_plate_video(
    ground_truth: pd.DataFrame,
    candidates: pd.DataFrame,
    track: pd.DataFrame | None,
    *,
    frame_count: int,
    fps: float,
    iou_threshold: float = 0.5,
    missing_track_reason: str = "no_confirmed_track",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """수동 원판 bbox와 후보·track을 비교해 영상별 지표를 만든다."""
    required_ground_truth = {
        "video_id", "frame_index", "plate_present", "plate_bbox_x",
        "plate_bbox_y", "plate_bbox_width", "plate_bbox_height",
    }
    missing_columns = sorted(required_ground_truth - set(ground_truth.columns))
    if missing_columns:
        raise ValueError("plate ground truth columns missing: " + ", ".join(missing_columns))
    if ground_truth.empty or ground_truth["video_id"].astype(str).nunique() != 1:
        raise ValueError("plate ground truth must contain exactly one video_id")
    if ground_truth.duplicated(["video_id", "frame_index"]).any():
        raise ValueError("plate ground truth has duplicate video/frame keys")
    if frame_count <= 0 or fps <= 0:
        raise ValueError("frame_count와 fps는 양수여야 합니다.")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold는 0보다 크고 1 이하여야 합니다.")
    if not str(missing_track_reason).strip():
        raise ValueError("missing_track_reason은 비어 있을 수 없습니다.")

    video_id = str(ground_truth.iloc[0]["video_id"])
    if track is None:
        track = pd.DataFrame(
            [
                _empty_plate_track_row(
                    video_id,
                    frame_index,
                    frame_index / fps,
                    "not_available",
                    str(missing_track_reason).strip(),
                )
                for frame_index in range(frame_count)
            ],
            columns=PLATE_TRACK_COLUMNS,
        )
    _validate_plate_table(candidates, PLATE_CANDIDATE_COLUMNS, "후보")
    _validate_plate_table(track, PLATE_TRACK_COLUMNS, "track")
    _validate_evaluation_video_rows(candidates, video_id, frame_count, "후보")
    _validate_evaluation_video_rows(track, video_id, frame_count, "track")
    if set(track["frame_index"].astype(int)) != set(range(frame_count)):
        raise ValueError("원판 track은 전체 frame을 포함해야 합니다.")

    candidates_by_frame = {
        int(frame_index): frame
        for frame_index, frame in candidates.groupby("frame_index", sort=False)
    }
    track_by_frame = track.set_index("frame_index", drop=False)
    rows = []
    for truth in ground_truth.sort_values("frame_index").itertuples(index=False):
        frame_index = int(truth.frame_index)
        if not 0 <= frame_index < frame_count:
            raise ValueError("plate ground truth frame이 영상 범위를 벗어났습니다.")
        plate_present = str(truth.plate_present).lower()
        if plate_present not in {"yes", "no"}:
            raise ValueError("plate_present는 yes/no만 허용합니다.")
        if plate_present == "no":
            rows.append(
                {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / fps,
                    "plate_present": "no",
                    "ground_truth_bbox_x1": math.nan,
                    "ground_truth_bbox_y1": math.nan,
                    "ground_truth_bbox_x2": math.nan,
                    "ground_truth_bbox_y2": math.nan,
                    "candidate_count": 0,
                    "candidate_best_id": math.nan,
                    "candidate_best_iou": math.nan,
                    "candidate_outcome": "not_applicable",
                    "track_bbox_x1": math.nan,
                    "track_bbox_y1": math.nan,
                    "track_bbox_x2": math.nan,
                    "track_bbox_y2": math.nan,
                    "track_status": "not_available",
                    "track_reason": "no_plate",
                    "track_iou": math.nan,
                    "normalized_center_error": math.nan,
                    "track_outcome": "not_applicable",
                }
            )
            continue
        ground_truth_bbox = (
            float(truth.plate_bbox_x),
            float(truth.plate_bbox_y),
            float(truth.plate_bbox_x + truth.plate_bbox_width),
            float(truth.plate_bbox_y + truth.plate_bbox_height),
        )
        _validate_xyxy_bbox(ground_truth_bbox, "ground truth")

        frame_candidates = candidates_by_frame.get(frame_index, candidates.iloc[0:0])
        candidate_matches = [
            (
                _bbox_iou(
                    ground_truth_bbox,
                    tuple(float(candidate[column]) for column in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")),
                ),
                int(candidate["candidate_id"]),
            )
            for _, candidate in frame_candidates.iterrows()
        ]
        best_iou, best_candidate_id = max(candidate_matches, default=(math.nan, math.nan))
        candidate_outcome = (
            "missing" if not candidate_matches
            else "correct" if best_iou >= iou_threshold
            else "wrong_target"
        )

        selected = track_by_frame.loc[frame_index]
        if str(selected["status"]) == "ok":
            track_bbox = tuple(
                float(selected[column])
                for column in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
            )
            _validate_xyxy_bbox(track_bbox, "track")
            track_iou = _bbox_iou(ground_truth_bbox, track_bbox)
            normalized_center_error = _normalized_center_error(
                ground_truth_bbox, track_bbox
            )
            track_outcome = "correct" if track_iou >= iou_threshold else "wrong_target"
        else:
            track_bbox = (math.nan,) * 4
            track_iou = math.nan
            normalized_center_error = math.nan
            track_outcome = "missing"

        rows.append(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps,
                "plate_present": "yes",
                "ground_truth_bbox_x1": ground_truth_bbox[0],
                "ground_truth_bbox_y1": ground_truth_bbox[1],
                "ground_truth_bbox_x2": ground_truth_bbox[2],
                "ground_truth_bbox_y2": ground_truth_bbox[3],
                "candidate_count": len(frame_candidates),
                "candidate_best_id": best_candidate_id,
                "candidate_best_iou": best_iou,
                "candidate_outcome": candidate_outcome,
                "track_bbox_x1": track_bbox[0],
                "track_bbox_y1": track_bbox[1],
                "track_bbox_x2": track_bbox[2],
                "track_bbox_y2": track_bbox[3],
                "track_status": str(selected["status"]),
                "track_reason": str(selected["reason"]),
                "track_iou": track_iou,
                "normalized_center_error": normalized_center_error,
                "track_outcome": track_outcome,
            }
        )

    frame_evaluation = pd.DataFrame(rows)
    applicable = frame_evaluation.loc[frame_evaluation["plate_present"] == "yes"]
    if applicable.empty:
        return frame_evaluation, {
            "video_id": video_id,
            "total_frame_count": frame_count,
            "plate_applicable_frame_count": 0,
            "track_ok_frame_count": 0,
            "plate_ok_rate": math.nan,
            "plate_non_ok_rate": math.nan,
            "longest_non_ok_gap_frames": math.nan,
            "longest_non_ok_gap_seconds": math.nan,
            "ground_truth_frame_count": 0,
            "candidate_correct_count": 0,
            "candidate_recall": math.nan,
            "track_detected_ground_truth_frame_count": 0,
            "track_correct_count": 0,
            "track_correct_rate": math.nan,
            "wrong_target_count": 0,
            "wrong_target_rate": math.nan,
            "mean_normalized_center_error": math.nan,
        }
    ok_frames = track["status"].astype(str).eq("ok")
    detected_ground_truth = applicable["track_outcome"] != "missing"
    candidate_correct_count = int(applicable["candidate_outcome"].eq("correct").sum())
    track_correct_count = int(applicable["track_outcome"].eq("correct").sum())
    wrong_target_count = int(applicable["track_outcome"].eq("wrong_target").sum())
    detected_count = int(detected_ground_truth.sum())
    ground_truth_count = len(applicable)
    longest_gap = _longest_false_run(ok_frames.tolist())
    summary = {
        "video_id": video_id,
        "total_frame_count": frame_count,
        "plate_applicable_frame_count": frame_count,
        "track_ok_frame_count": int(ok_frames.sum()),
        "plate_ok_rate": float(ok_frames.mean()),
        "plate_non_ok_rate": float((~ok_frames).mean()),
        "longest_non_ok_gap_frames": longest_gap,
        "longest_non_ok_gap_seconds": longest_gap / fps,
        "ground_truth_frame_count": ground_truth_count,
        "candidate_correct_count": candidate_correct_count,
        "candidate_recall": candidate_correct_count / ground_truth_count,
        "track_detected_ground_truth_frame_count": detected_count,
        "track_correct_count": track_correct_count,
        "track_correct_rate": track_correct_count / ground_truth_count,
        "wrong_target_count": wrong_target_count,
        "wrong_target_rate": wrong_target_count / detected_count if detected_count else math.nan,
        "mean_normalized_center_error": float(
            applicable["normalized_center_error"].mean()
        ),
    }
    return frame_evaluation, summary


def aggregate_plate_evaluations(per_video: pd.DataFrame) -> dict[str, Any]:
    """영상별 count를 합산해 전체 threshold 지표를 계산한다."""
    count_columns = (
        "total_frame_count", "plate_applicable_frame_count",
        "track_ok_frame_count", "ground_truth_frame_count",
        "candidate_correct_count", "track_detected_ground_truth_frame_count",
        "track_correct_count", "wrong_target_count",
    )
    required_columns = set(count_columns) | {
        "video_id", "longest_non_ok_gap_frames", "longest_non_ok_gap_seconds",
        "mean_normalized_center_error",
    }
    missing_columns = sorted(required_columns - set(per_video.columns))
    if missing_columns:
        raise ValueError("plate evaluation columns missing: " + ", ".join(missing_columns))
    if per_video.empty:
        raise ValueError("plate evaluation summary is empty")

    counts = {
        column: int(pd.to_numeric(per_video[column], errors="raise").sum())
        for column in count_columns
    }
    applicable_frames = counts["plate_applicable_frame_count"]
    ground_truth_frames = counts["ground_truth_frame_count"]
    detected_frames = counts["track_detected_ground_truth_frame_count"]
    center_errors = pd.to_numeric(
        per_video["mean_normalized_center_error"], errors="coerce"
    )
    center_weights = pd.to_numeric(
        per_video["track_detected_ground_truth_frame_count"], errors="raise"
    )
    usable_center_errors = center_errors.notna() & center_weights.gt(0)
    weighted_center_error = (
        float(
            (center_errors[usable_center_errors] * center_weights[usable_center_errors]).sum()
            / center_weights[usable_center_errors].sum()
        )
        if usable_center_errors.any()
        else math.nan
    )
    gap_frames = pd.to_numeric(per_video["longest_non_ok_gap_frames"], errors="coerce")
    gap_seconds = pd.to_numeric(per_video["longest_non_ok_gap_seconds"], errors="coerce")
    return {
        "video_id": "ALL",
        "video_count": int(per_video["video_id"].astype(str).nunique()),
        **counts,
        "plate_ok_rate": counts["track_ok_frame_count"] / applicable_frames
        if applicable_frames else math.nan,
        "plate_non_ok_rate": 1 - counts["track_ok_frame_count"] / applicable_frames
        if applicable_frames else math.nan,
        "longest_non_ok_gap_frames": int(gap_frames.max()) if gap_frames.notna().any() else math.nan,
        "longest_non_ok_gap_seconds": float(gap_seconds.max()) if gap_seconds.notna().any() else math.nan,
        "candidate_recall": counts["candidate_correct_count"] / ground_truth_frames if ground_truth_frames else math.nan,
        "track_correct_rate": counts["track_correct_count"] / ground_truth_frames if ground_truth_frames else math.nan,
        "wrong_target_rate": counts["wrong_target_count"] / detected_frames if detected_frames else math.nan,
        "mean_normalized_center_error": weighted_center_error,
    }


def save_plate_evaluation_results(
    frame_evaluation: pd.DataFrame,
    comparison: pd.DataFrame,
    output_root: str | Path,
) -> None:
    """원판 frame 평가와 threshold 비교표를 함께 저장한다."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv_atomically(
        frame_evaluation, output_root / "plate_frame_evaluation.csv"
    )
    _write_csv_atomically(
        comparison, output_root / "plate_threshold_comparison.csv"
    )


def _validate_evaluation_video_rows(
    table: pd.DataFrame,
    video_id: str,
    frame_count: int,
    table_name: str,
) -> None:
    if table.empty:
        return
    if set(table["video_id"].astype(str)) != {video_id}:
        raise ValueError(f"원판 {table_name} video_id가 정답과 다릅니다.")
    frame_indices = pd.to_numeric(table["frame_index"], errors="coerce")
    if frame_indices.isna().any() or (frame_indices < 0).any() or (frame_indices >= frame_count).any():
        raise ValueError(f"원판 {table_name} frame이 영상 범위를 벗어났습니다.")


def _validate_xyxy_bbox(bbox: tuple[float, float, float, float], name: str) -> None:
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(value) for value in bbox) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} bbox가 유효하지 않습니다.")


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _normalized_center_error(
    ground_truth_bbox: tuple[float, float, float, float],
    predicted_bbox: tuple[float, float, float, float],
) -> float:
    ground_truth_center = (
        (ground_truth_bbox[0] + ground_truth_bbox[2]) / 2.0,
        (ground_truth_bbox[1] + ground_truth_bbox[3]) / 2.0,
    )
    predicted_center = (
        (predicted_bbox[0] + predicted_bbox[2]) / 2.0,
        (predicted_bbox[1] + predicted_bbox[3]) / 2.0,
    )
    diagonal = math.hypot(
        ground_truth_bbox[2] - ground_truth_bbox[0],
        ground_truth_bbox[3] - ground_truth_bbox[1],
    )
    return math.dist(ground_truth_center, predicted_center) / diagonal


def _longest_false_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _validate_plate_table(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    table_name: str,
) -> None:
    if tuple(table.columns) != columns:
        raise ValueError(f"원판 {table_name} schema가 일치하지 않습니다.")
    key_columns = ["video_id", "frame_index"]
    if columns == PLATE_CANDIDATE_COLUMNS:
        key_columns.append("candidate_id")
    if table.duplicated(key_columns).any():
        raise ValueError(f"원판 {table_name}에 duplicate 복합키가 있습니다.")


def _lifter_centers_by_frame(
    lifter_centers: pd.DataFrame | None,
) -> dict[int, tuple[float, float]]:
    if lifter_centers is None:
        return {}
    required_columns = {"frame_index", "center_x_px", "center_y_px"}
    if not required_columns.issubset(lifter_centers.columns):
        raise ValueError("lifter_centers에 frame_index와 중심 좌표가 필요합니다.")
    if lifter_centers.duplicated("frame_index").any():
        raise ValueError("lifter_centers frame_index는 고유해야 합니다.")
    return {
        int(row.frame_index): (float(row.center_x_px), float(row.center_y_px))
        for row in lifter_centers.loc[:, sorted(required_columns)].itertuples(index=False)
    }


def _select_continuous_plate(
    frame_candidates: pd.DataFrame,
    previous: pd.Series,
    lifter_center: tuple[float, float] | None,
    *,
    max_center_shift: float,
    max_area_log_change: float,
    ambiguity_margin: float,
) -> tuple[pd.Series | None, bool]:
    previous_width = max(float(previous["bbox_x2"] - previous["bbox_x1"]), 1.0)
    previous_height = max(float(previous["bbox_y2"] - previous["bbox_y1"]), 1.0)
    scale = max(math.hypot(previous_width, previous_height), 1.0)
    previous_area = max(previous_width * previous_height, 1.0)
    ranked = frame_candidates.copy()
    for column in (
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_x_px",
        "center_y_px",
    ):
        ranked[column] = pd.to_numeric(ranked[column], errors="raise").astype(float)
    ranked["center_shift"] = np.hypot(
        ranked["center_x_px"] - float(previous["center_x_px"]),
        ranked["center_y_px"] - float(previous["center_y_px"]),
    ) / scale
    areas = (ranked["bbox_x2"] - ranked["bbox_x1"]) * (
        ranked["bbox_y2"] - ranked["bbox_y1"]
    )
    ranked["area_log_change"] = np.abs(np.log(areas.clip(lower=1.0) / previous_area))
    viable = ranked.loc[
        (ranked["center_shift"] <= max_center_shift)
        & (ranked["area_log_change"] <= max_area_log_change)
    ].copy()
    if viable.empty:
        return None, False
    viable["continuity_cost"] = viable["center_shift"] + viable["area_log_change"]
    if lifter_center is not None:
        viable["continuity_cost"] += np.hypot(
            viable["center_x_px"] - lifter_center[0],
            viable["center_y_px"] - lifter_center[1],
        ) / scale
    ranked = viable.sort_values(["continuity_cost", "candidate_id"]).reset_index(drop=True)
    ambiguous = len(ranked) > 1 and (
        float(ranked.loc[1, "continuity_cost"] - ranked.loc[0, "continuity_cost"])
        <= ambiguity_margin
    )
    return (None, True) if ambiguous else (ranked.iloc[0], False)


def _empty_plate_track_row(
    video_id: str,
    frame_index: int,
    timestamp_seconds: float,
    status: str,
    reason: str,
) -> dict[str, Any]:
    row = {column: math.nan for column in PLATE_TRACK_COLUMNS}
    row.update(
        video_id=video_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        status=status,
        reason=reason,
    )
    return row


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


def _model_selection_pilot(evaluation_split: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "video_id",
        "group",
        "split_role",
        "frame_count",
        "width",
        "height",
        "has_plate",
    }
    missing = sorted(required_columns - set(evaluation_split.columns))
    if missing:
        raise ValueError("evaluation split columns missing: " + ", ".join(missing))
    pilot = evaluation_split.loc[
        evaluation_split["split_role"] == "model_selection_pilot"
    ].copy()
    if pilot.empty:
        raise ValueError("model_selection_pilot 영상이 없습니다.")
    if pilot["video_id"].duplicated().any():
        raise ValueError("evaluation split video_id가 중복되었습니다.")
    if len(pilot) != 3 or pilot["group"].value_counts().to_dict() != {
        "A": 1,
        "B": 1,
        "C": 1,
    }:
        raise ValueError("model_selection_pilot은 A/B/C 영상이 각각 하나씩 필요합니다.")
    return pilot


def _validate_pilot_sample_frames(
    samples: pd.DataFrame,
    pilot: pd.DataFrame,
) -> None:
    expected_video_ids = set(pilot["video_id"].astype(str))
    actual_video_ids = set(samples["video_id"].astype(str))
    if actual_video_ids != expected_video_ids:
        raise ValueError("pilot ground truth video_id가 evaluation split과 다릅니다.")

    numeric_indices = pd.to_numeric(samples["frame_index"], errors="raise")
    if not np.isfinite(numeric_indices).all() or not np.equal(
        numeric_indices,
        np.floor(numeric_indices),
    ).all():
        raise ValueError("frame_index must contain finite integers")
    counts = samples.groupby("video_id").size()
    if not (counts == PILOT_FRAMES_PER_VIDEO).all():
        raise ValueError(
            f"각 pilot 영상에는 {PILOT_FRAMES_PER_VIDEO}개 frame이 필요합니다."
        )
    frame_counts = samples["video_id"].map(
        pilot.set_index("video_id")["frame_count"]
    )
    if ((numeric_indices < 0) | (numeric_indices >= frame_counts)).any():
        raise ValueError("pilot frame_index가 영상 범위를 벗어났습니다.")


def _validate_ground_truth_bbox(
    row: pd.Series,
    *,
    prefix: str,
    image_width: int,
    image_height: int,
    required: bool,
    na_reason: str,
) -> None:
    columns = [
        f"{prefix}_bbox_x",
        f"{prefix}_bbox_y",
        f"{prefix}_bbox_width",
        f"{prefix}_bbox_height",
    ]
    values = row.loc[columns]
    missing_count = int(values.isna().sum())
    if missing_count == len(columns):
        reasons = set(na_reason.split(";"))
        if not required and prefix == "plate" and "no_plate" not in reasons:
            raise ValueError("원판 없는 frame은 na_reason에 no_plate가 필요합니다.")
        if required and not any(
            reason.startswith(f"{prefix}_") for reason in reasons
        ):
            raise ValueError(
                f"{prefix} bbox가 없으면 {prefix}_ 사유를 na_reason에 기록하세요."
            )
        return
    if missing_count:
        raise ValueError(f"{prefix} bbox는 네 좌표를 모두 입력해야 합니다.")
    if not required:
        raise ValueError(f"{prefix}가 없는 frame에는 bbox를 입력할 수 없습니다.")
    _validate_opencv_bbox(
        tuple(values.tolist()),
        image_width=image_width,
        image_height=image_height,
    )


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
