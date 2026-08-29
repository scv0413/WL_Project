"""YOLO Pose 후보를 저장하고 한 선수의 고정 신체면 측정값을 만든다."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from snatch_analysis.geometry import (
    calculate_internal_angle,
    calculate_trunk_inclination,
)
COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

POSE_CANDIDATE_COLUMNS = (
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "candidate_id",
    "joint_index",
    "joint_name",
    "x",
    "y",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_confidence",
    "bbox_area",
    "bbox_center_x",
    "bbox_center_y",
)

SELECTED_KEYPOINT_COLUMNS = (*POSE_CANDIDATE_COLUMNS, "status", "reason")
ANGLE_JOINTS = {
    "knee": ("hip", "knee", "ankle"),
    "hip": ("shoulder", "hip", "knee"),
    "elbow": ("shoulder", "elbow", "wrist"),
    "trunk": ("shoulder", "hip"),
}


def verify_ultralytics_safe_load() -> dict[str, Any]:
    """Restricted checkpoint loading을 켜고 실제 활성 상태를 검증한다."""
    already_imported = "ultralytics" in sys.modules
    os.environ["ULTRALYTICS_SAFE_LOAD"] = "True"

    from ultralytics.nn.tasks import _SafeLoad
    from ultralytics.utils import SAFE_LOAD

    active = bool(SAFE_LOAD and _SafeLoad.SUPPORTED and _SafeLoad.restricted())
    if not active:
        hint = (
            " 커널을 재시작하고 Ultralytics보다 먼저 이 함수를 실행하세요."
            if already_imported and not SAFE_LOAD
            else " 현재 PyTorch/Ultralytics 조합은 restricted loading을 지원하지 않습니다."
        )
        raise RuntimeError(
            "ULTRALYTICS_SAFE_LOAD가 활성화되지 않아 .pt 로드를 중단합니다." + hint
        )

    return {
        "environment_value": os.environ["ULTRALYTICS_SAFE_LOAD"],
        "supported": bool(_SafeLoad.SUPPORTED),
        "active": active,
    }


def resolve_inference_device(requested_device: str = "mps") -> dict[str, str]:
    """요청 장치와 실제 사용 장치를 기록 가능한 형태로 반환한다."""
    import torch

    if requested_device == "mps" and not torch.backends.mps.is_available():
        return {
            "requested_device": requested_device,
            "actual_device": "cpu",
            "fallback_reason": "mps_unavailable",
        }
    return {
        "requested_device": requested_device,
        "actual_device": requested_device,
        "fallback_reason": "",
    }


def save_pose_candidate_cache(
    candidates: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """완성된 Pose 후보 표만 원자적으로 교체한다."""
    if tuple(candidates.columns) != POSE_CANDIDATE_COLUMNS:
        raise ValueError("Pose 후보 cache schema가 일치하지 않습니다.")

    _write_csv_atomically(candidates, output_path)


def _write_csv_atomically(table: pd.DataFrame, output_path: str | Path) -> None:
    """같은 디렉터리의 임시 파일을 완성한 뒤 CSV를 교체한다."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output_path.parent,
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        table.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_selected_pose_keypoints(
    selected_keypoints: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """완성된 주동작자 Pose 표만 원자적으로 교체한다."""
    if tuple(selected_keypoints.columns) != SELECTED_KEYPOINT_COLUMNS:
        raise ValueError("선택 Pose schema가 일치하지 않습니다.")

    _write_csv_atomically(selected_keypoints, output_path)


def validate_pose_candidate_cache(
    candidates: pd.DataFrame,
    *,
    expected_video_id: str,
    expected_frame_count: int,
    run_metadata: dict[str, Any],
    expected_run_metadata: dict[str, Any],
) -> None:
    """저장 후보가 현재 실험의 완료된 추론 결과인지 확인한다."""
    if tuple(candidates.columns) != POSE_CANDIDATE_COLUMNS:
        raise ValueError("Pose 후보 cache schema가 일치하지 않습니다.")
    mismatched = [
        key
        for key, expected in expected_run_metadata.items()
        if run_metadata.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            "Pose 후보 cache 설정이 현재 실험과 다릅니다: "
            + ", ".join(mismatched)
        )
    if (
        run_metadata.get("status") != "complete"
        or run_metadata.get("processed_frame_count") != expected_frame_count
    ):
        raise ValueError("Pose 후보 cache가 전체 frame 추론 완료 상태가 아닙니다.")
    if candidates.empty:
        return
    if set(candidates["video_id"].astype(str)) != {expected_video_id}:
        raise ValueError("Pose 후보 cache의 video_id가 현재 영상과 다릅니다.")
    frame_indices = pd.to_numeric(candidates["frame_index"], errors="coerce")
    if (
        frame_indices.isna().any()
        or (frame_indices < 0).any()
        or (frame_indices >= expected_frame_count).any()
    ):
        raise ValueError("Pose 후보 cache에 영상 범위 밖 frame이 있습니다.")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def pose_result_to_candidates(
    result: Any,
    video_id: str,
    frame_index: int,
    timestamp_seconds: float,
) -> pd.DataFrame:
    """한 frame의 Ultralytics Pose 결과를 후보×17관절 long table로 변환한다."""
    if result.boxes is None or result.keypoints is None:
        return pd.DataFrame(columns=POSE_CANDIDATE_COLUMNS)

    boxes = _to_numpy(result.boxes.xyxy)
    box_confidences = _to_numpy(result.boxes.conf)
    keypoints = _to_numpy(result.keypoints.data)
    if len(boxes) != len(box_confidences) or len(boxes) != len(keypoints):
        raise ValueError("Pose bbox, confidence, keypoint 후보 수가 일치하지 않습니다.")
    if len(keypoints) and keypoints.shape[1] != len(COCO_KEYPOINT_NAMES):
        raise ValueError(
            f"COCO keypoint는 {len(COCO_KEYPOINT_NAMES)}개여야 합니다: "
            f"{keypoints.shape[1]}개"
        )

    rows: list[dict[str, Any]] = []
    for candidate_id, (bbox, bbox_confidence, candidate_keypoints) in enumerate(
        zip(boxes, box_confidences, keypoints, strict=True)
    ):
        x1, y1, x2, y2 = map(float, bbox)
        bbox_values = {
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_confidence": float(bbox_confidence),
            "bbox_area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
            "bbox_center_x": (x1 + x2) / 2.0,
            "bbox_center_y": (y1 + y2) / 2.0,
        }
        for joint_index, joint_name in enumerate(COCO_KEYPOINT_NAMES):
            joint = candidate_keypoints[joint_index]
            confidence = float(joint[2]) if len(joint) > 2 else math.nan
            rows.append(
                {
                    "video_id": video_id,
                    "frame_index": int(frame_index),
                    "timestamp_seconds": float(timestamp_seconds),
                    "candidate_id": candidate_id,
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "x": float(joint[0]),
                    "y": float(joint[1]),
                    "confidence": confidence,
                    **bbox_values,
                }
            )
    return pd.DataFrame(rows, columns=POSE_CANDIDATE_COLUMNS)


def infer_pose_video_candidates(
    model: Any,
    *,
    video_path: str | Path,
    video_id: str,
    frame_count: int,
    fps: float,
    image_size: int,
    confidence: float,
    device: str,
) -> tuple[pd.DataFrame, int]:
    """Ultralytics stream 결과를 Pose 후보 표로 변환한다."""
    if fps <= 0:
        raise ValueError("fps는 양수여야 합니다.")
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
            pose_result_to_candidates(
                result,
                video_id,
                frame_index,
                frame_index / fps,
            )
        )
    processed_frame_count = len(frame_tables)
    if processed_frame_count != frame_count:
        raise RuntimeError(
            "Pose processed frame count mismatch: "
            f"expected={frame_count}, actual={processed_frame_count}"
        )
    return (
        pd.concat(frame_tables, ignore_index=True)
        if frame_tables
        else pd.DataFrame(columns=POSE_CANDIDATE_COLUMNS),
        processed_frame_count,
    )


def _candidate_boxes(candidates: pd.DataFrame) -> pd.DataFrame:
    joint_columns = {"joint_index", "joint_name", "x", "y", "confidence"}
    columns = [column for column in POSE_CANDIDATE_COLUMNS if column not in joint_columns]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    return (
        candidates.loc[:, columns]
        .drop_duplicates(["frame_index", "candidate_id"])
        .sort_values(["frame_index", "candidate_id"])
        .reset_index(drop=True)
    )


def select_initial_primary_lifter(
    candidates: pd.DataFrame,
    frame_width: int,
) -> dict[str, Any]:
    """가장 크면서 가로 중앙에 가까운 첫 후보만 자동 선택한다."""
    boxes = _candidate_boxes(candidates)
    if boxes.empty:
        return {
            "frame_index": None,
            "candidate_id": None,
            "bbox_confidence": None,
            "status": "not_available",
            "reason": "no_pose_candidates",
        }

    first_frame_index = int(boxes["frame_index"].min())
    first_frame = boxes.loc[boxes["frame_index"] == first_frame_index].copy()
    largest = first_frame.loc[
        first_frame["bbox_area"] == first_frame["bbox_area"].max()
    ]
    horizontal_distance = (first_frame["bbox_center_x"] - frame_width / 2.0).abs()
    centered = first_frame.loc[horizontal_distance == horizontal_distance.min()]
    agrees = (
        len(largest) == 1
        and len(centered) == 1
        and int(largest.iloc[0]["candidate_id"])
        == int(centered.iloc[0]["candidate_id"])
    )
    selected = largest.iloc[0] if agrees else None
    reason = (
        "largest_bbox_and_horizontal_center"
        if agrees
        else "selection_criteria_disagree"
        if len(largest) == 1 and len(centered) == 1
        else "ambiguous_candidates"
    )
    return {
        "frame_index": first_frame_index,
        "candidate_id": int(selected["candidate_id"]) if agrees else None,
        "bbox_confidence": float(selected["bbox_confidence"]) if agrees else None,
        "status": "auto_selected" if agrees else "confirmation_required",
        "reason": reason,
    }


def resolve_initial_primary_lifter(
    primary_selection: dict[str, Any],
    manual_override_candidate_id: int | None = None,
) -> dict[str, Any]:
    """자동 선택을 기본으로 사용하고 수동 override가 있으면 우선한다."""
    if manual_override_candidate_id is not None:
        return {
            "candidate_id": int(manual_override_candidate_id),
            "selection_method": "manual_override",
        }

    candidate_id = primary_selection.get("candidate_id")
    return {
        "candidate_id": int(candidate_id) if candidate_id is not None else None,
        "selection_method": "automatic" if candidate_id is not None else None,
    }


def _empty_selected_row(
    video_id: str,
    frame_index: int,
    timestamp_seconds: float,
    joint_index: int,
    joint_name: str,
    reason: str,
) -> dict[str, Any]:
    row = {column: math.nan for column in SELECTED_KEYPOINT_COLUMNS}
    row.update(
        {
            "video_id": video_id,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "joint_index": joint_index,
            "joint_name": joint_name,
            "status": "not_available",
            "reason": reason,
        }
    )
    return row


def _empty_selected_rows(
    video_id: str,
    frame_index: int,
    timestamp_seconds: float,
    reason: str,
) -> list[dict[str, Any]]:
    return [
        _empty_selected_row(
            video_id,
            frame_index,
            timestamp_seconds,
            joint_index,
            joint_name,
            reason,
        )
        for joint_index, joint_name in enumerate(COCO_KEYPOINT_NAMES)
    ]


def _selected_candidate_rows(
    candidate_rows: pd.DataFrame,
    keypoint_confidence_threshold: float,
) -> list[dict[str, Any]]:
    first = candidate_rows.iloc[0]
    by_joint = candidate_rows.set_index("joint_name", drop=False)
    rows = []
    for joint_index, joint_name in enumerate(COCO_KEYPOINT_NAMES):
        if joint_name not in by_joint.index:
            missing = _empty_selected_row(
                video_id=str(first["video_id"]),
                frame_index=int(first["frame_index"]),
                timestamp_seconds=float(first["timestamp_seconds"]),
                joint_index=joint_index,
                joint_name=joint_name,
                reason="missing_keypoint",
            )
            for column in POSE_CANDIDATE_COLUMNS:
                if column.startswith("bbox_") or column == "candidate_id":
                    missing[column] = first[column]
            rows.append(missing)
            continue

        source = by_joint.loc[joint_name]
        if isinstance(source, pd.DataFrame):
            raise ValueError(f"선택 후보에 중복 관절이 있습니다: {joint_name}")
        row = source.loc[list(POSE_CANDIDATE_COLUMNS)].to_dict()
        confidence = pd.to_numeric(row["confidence"], errors="coerce")
        coordinate_ok = pd.notna(row["x"]) and pd.notna(row["y"])
        if not coordinate_ok or pd.isna(confidence):
            row.update(x=math.nan, y=math.nan, status="not_available", reason="missing_keypoint")
        elif float(confidence) < keypoint_confidence_threshold:
            row.update(
                x=math.nan,
                y=math.nan,
                status="not_available",
                reason="low_keypoint_confidence",
            )
        else:
            row.update(status="ok", reason="")
        rows.append(row)
    return rows


def select_primary_lifter_track(
    candidates: pd.DataFrame,
    frame_count: int,
    initial_frame_index: int,
    initial_candidate_id: int,
    keypoint_confidence_threshold: float,
    fps: float | None = None,
    max_center_shift: float = 1.5,
    max_area_log_change: float = 1.0,
    continuity_ambiguity_margin: float = 0.05,
) -> pd.DataFrame:
    """확정된 첫 후보에서 중심·면적 연속성으로 한 선수의 17관절을 선택한다."""
    if frame_count <= 0:
        raise ValueError("frame_count는 양수여야 합니다.")
    if not 0 <= initial_frame_index < frame_count:
        raise ValueError("initial_frame_index가 frame 범위를 벗어났습니다.")
    boxes = _candidate_boxes(candidates)
    initial = boxes.loc[
        (boxes["frame_index"] == initial_frame_index)
        & (boxes["candidate_id"] == initial_candidate_id)
    ]
    if len(initial) != 1:
        raise ValueError("확정할 초기 Pose 후보를 정확히 하나 지정해야 합니다.")

    video_id = str(initial.iloc[0]["video_id"])
    previous = initial.iloc[0]
    selected_rows: list[dict[str, Any]] = []
    boxes_by_frame = {
        int(frame_index): frame
        for frame_index, frame in boxes.groupby("frame_index", sort=False)
    }
    candidates_by_frame_and_id = {
        (int(frame_index), int(candidate_id)): group
        for (frame_index, candidate_id), group in candidates.groupby(
            ["frame_index", "candidate_id"], sort=False
        )
    }
    empty_boxes = boxes.iloc[0:0]
    for frame_index in range(frame_count):
        timestamp_seconds = frame_index / fps if fps else math.nan
        frame_boxes = boxes_by_frame.get(frame_index, empty_boxes).copy()
        reason = "no_detection"
        chosen: pd.Series | None = None

        if frame_index == initial_frame_index:
            chosen = previous
        elif frame_index > initial_frame_index and not frame_boxes.empty:
            area_scale = max(math.sqrt(float(previous["bbox_area"])), 1.0)
            frame_boxes["center_shift"] = np.hypot(
                frame_boxes["bbox_center_x"] - float(previous["bbox_center_x"]),
                frame_boxes["bbox_center_y"] - float(previous["bbox_center_y"]),
            ) / area_scale
            frame_boxes["area_log_change"] = np.abs(
                np.log(
                    frame_boxes["bbox_area"].clip(lower=1.0)
                    / max(float(previous["bbox_area"]), 1.0)
                )
            )
            viable = frame_boxes.loc[
                (frame_boxes["center_shift"] <= max_center_shift)
                & (frame_boxes["area_log_change"] <= max_area_log_change)
            ].copy()
            if viable.empty:
                reason = "continuity_mismatch"
            else:
                viable["continuity_cost"] = (
                    viable["center_shift"] + viable["area_log_change"]
                )
                ranked = viable.sort_values(
                    ["continuity_cost", "candidate_id"]
                ).reset_index(drop=True)
                if len(ranked) > 1 and (
                    float(ranked.loc[1, "continuity_cost"] - ranked.loc[0, "continuity_cost"])
                    <= continuity_ambiguity_margin
                ):
                    reason = "ambiguous_candidates"
                else:
                    chosen = ranked.iloc[0]

        if chosen is None:
            selected_rows.extend(
                _empty_selected_rows(video_id, frame_index, timestamp_seconds, reason)
            )
            continue

        previous = chosen
        source_rows = candidates_by_frame_and_id[
            (frame_index, int(chosen["candidate_id"]))
        ]
        selected_rows.extend(
            _selected_candidate_rows(source_rows, keypoint_confidence_threshold)
        )

    return pd.DataFrame(selected_rows, columns=SELECTED_KEYPOINT_COLUMNS)


def summarize_body_side_confidence(selected_keypoints: pd.DataFrame) -> pd.DataFrame:
    """각도에 쓰는 여섯 관절의 좌우 confidence와 사용 가능 비율을 요약한다."""
    rows = []
    for side in ("left", "right"):
        joint_names = {f"{side}_{name}" for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")}
        joints = selected_keypoints.loc[selected_keypoints["joint_name"].isin(joint_names)]
        confidences = pd.to_numeric(joints["confidence"], errors="coerce")
        rows.append(
            {
                "side": side,
                "mean_confidence": float(confidences.mean()) if confidences.notna().any() else math.nan,
                "valid_fraction": float((joints["status"] == "ok").mean()) if len(joints) else 0.0,
                "valid_joint_count": int((joints["status"] == "ok").sum()),
                "total_joint_count": int(len(joints)),
            }
        )
    return pd.DataFrame(rows)


def _angle_measurement(
    indexed_frame: pd.DataFrame,
    side: str,
    measurement_name: str,
) -> dict[str, Any]:
    joint_names = [f"{side}_{name}" for name in ANGLE_JOINTS[measurement_name]]
    joints = [indexed_frame.loc[name] for name in joint_names]
    failed = [joint for joint in joints if joint["status"] != "ok"]
    confidence_values = pd.to_numeric(
        pd.Series([joint["confidence"] for joint in joints]), errors="coerce"
    )
    confidence = (
        float(confidence_values.min()) if confidence_values.notna().all() else math.nan
    )
    if failed:
        return {
            "angle_deg": math.nan,
            "confidence": confidence,
            "status": "not_available",
            "reason": str(failed[0]["reason"]),
        }

    points = [(float(joint["x"]), float(joint["y"])) for joint in joints]
    if measurement_name == "trunk":
        result = calculate_trunk_inclination(points[0], points[1])
    else:
        result = calculate_internal_angle(points[0], points[1], points[2])
    return {
        "angle_deg": result["angle_deg"] if result["angle_deg"] is not None else math.nan,
        "confidence": confidence,
        "status": result["status"],
        "reason": result["reason"],
    }


def calculate_joint_angles(
    selected_keypoints: pd.DataFrame,
    body_side: str,
) -> pd.DataFrame:
    """선택한 좌/우 한쪽을 모든 frame에 고정해 네 가지 각도를 계산한다."""
    if body_side not in {"left", "right"}:
        raise ValueError("body_side는 'left' 또는 'right'여야 합니다.")

    rows = []
    for frame_index, frame in selected_keypoints.groupby("frame_index", sort=True):
        indexed_frame = frame.set_index("joint_name", drop=False)
        row = {
            "video_id": str(frame.iloc[0]["video_id"]),
            "frame_index": int(frame_index),
            "timestamp_seconds": float(frame.iloc[0]["timestamp_seconds"]),
            "body_side": body_side,
        }
        for measurement_name in ANGLE_JOINTS:
            measurement = _angle_measurement(
                indexed_frame, body_side, measurement_name
            )
            for field, value in measurement.items():
                row[f"{measurement_name}_{field}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
