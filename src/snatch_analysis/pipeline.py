"""Join saved B00 measurements into one replayable frame table."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from snatch_analysis.pose import COCO_KEYPOINT_NAMES, _write_csv_atomically


_KEY_COLUMNS = ["video_id", "frame_index"]


def build_frame_basis(video_id: str, frame_count: int, fps: float) -> pd.DataFrame:
    """Build the authoritative ordered frame basis for one decoded video."""
    if not video_id:
        raise ValueError("video_id must not be empty")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return pd.DataFrame(
        {
            "video_id": [str(video_id)] * frame_count,
            "frame_index": range(frame_count),
            "timestamp_seconds": [frame_index / fps for frame_index in range(frame_count)],
        }
    )


def summarize_pose_keypoints(selected_keypoints: pd.DataFrame) -> pd.DataFrame:
    """Reduce selected long-form Pose rows to one non-interpolated row per frame."""
    required = {*_KEY_COLUMNS, "joint_name", "confidence", "status", "reason"}
    missing = sorted(required - set(selected_keypoints.columns))
    if missing:
        raise ValueError("pose_keypoints columns missing: " + ", ".join(missing))
    if selected_keypoints.empty:
        return pd.DataFrame(
            columns=[*_KEY_COLUMNS, "pose_status", "pose_reason", "pose_mean_confidence", "pose_available_keypoint_count", "pose_total_keypoint_count"]
        )
    if selected_keypoints.duplicated([*_KEY_COLUMNS, "joint_name"]).any():
        raise ValueError("duplicate long-form pose keys")

    rows = []
    expected_joint_names = set(COCO_KEYPOINT_NAMES)
    for (video_id, frame_index), frame in selected_keypoints.groupby(_KEY_COLUMNS, sort=False):
        joint_names = set(frame["joint_name"])
        missing_joints = sorted(expected_joint_names - joint_names)
        unexpected_joints = sorted(joint_names - expected_joint_names)
        if missing_joints or unexpected_joints:
            details = []
            if missing_joints:
                details.append("missing joints: " + ", ".join(missing_joints))
            if unexpected_joints:
                details.append("unexpected joints: " + ", ".join(unexpected_joints))
            raise ValueError(
                "pose_keypoints frame has incomplete COCO joint set "
                f"(video_id={video_id}, frame_index={frame_index}; "
                + "; ".join(details)
                + ")"
            )
        available = frame.loc[frame["status"] == "ok"]
        reasons = frame.loc[frame["status"] != "ok", "reason"].dropna().astype(str)
        available_count = len(available)
        total_count = len(frame)
        status = "ok" if available_count == total_count else "partial" if available_count else "not_available"
        rows.append(
            {
                "video_id": str(video_id),
                "frame_index": int(frame_index),
                "pose_status": status,
                "pose_reason": "" if status == "ok" else (reasons.iloc[0] if not reasons.empty else "no_pose_keypoints"),
                "pose_mean_confidence": pd.to_numeric(available["confidence"], errors="coerce").mean(),
                "pose_available_keypoint_count": available_count,
                "pose_total_keypoint_count": total_count,
            }
        )
    return pd.DataFrame(rows)


def fuse_frame_analysis(
    frame_basis: pd.DataFrame,
    phase_by_frame: pd.DataFrame,
    pose_keypoints: pd.DataFrame,
    joint_angles: pd.DataFrame,
    plate_track: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join saved phase, Pose summary, angles, and plate track onto frames."""
    _validate_basis(frame_basis)
    basis = frame_basis
    video_id = str(basis.loc[0, "video_id"])
    _validate_direct_table(phase_by_frame, "phase", video_id, basis)
    if "phase" not in phase_by_frame.columns:
        raise ValueError("phase columns missing: phase")
    if (
        set(phase_by_frame["frame_index"]) != set(basis["frame_index"])
        or phase_by_frame["phase"].isna().any()
    ):
        raise ValueError("phase frame coverage must match the frame basis with no null phase")
    _validate_direct_table(joint_angles, "angles", video_id, basis)
    _validate_direct_table(plate_track, "plate", video_id, basis)
    pose = summarize_pose_keypoints(pose_keypoints)
    if not pose.empty:
        _validate_direct_table(pose, "pose summary", video_id, basis)

    result = basis.copy()
    for table in (phase_by_frame, pose, _without_timestamp(joint_angles), _rename_plate_columns(plate_track)):
        result = result.merge(table, on=_KEY_COLUMNS, how="left", sort=False, validate="one_to_one")
    result["pose_status"] = result["pose_status"].fillna("not_available")
    result["pose_reason"] = result["pose_reason"].fillna("no_pose_keypoints")
    result["pose_available_keypoint_count"] = result["pose_available_keypoint_count"].fillna(0).astype(int)
    result["pose_total_keypoint_count"] = result["pose_total_keypoint_count"].fillna(0).astype(int)
    result["plate_status"] = result["plate_status"].fillna("missing")
    result["plate_reason"] = result["plate_reason"].fillna("no_plate_track")
    return result


def save_frame_analysis(frame_analysis: pd.DataFrame, output_path: str | Path) -> None:
    """Atomically save a completed measurement table before visualization starts."""
    _validate_basis(frame_analysis)
    _write_csv_atomically(frame_analysis, output_path)


def save_json_atomically(data: object, output_path: str | Path) -> None:
    """Write a complete JSON document before replacing the prior summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_fusion(
    frame_basis: pd.DataFrame,
    phase_by_frame: pd.DataFrame,
    pose_keypoints: pd.DataFrame,
    joint_angles: pd.DataFrame,
    plate_track: pd.DataFrame,
    *,
    output_path: str | Path,
    visualization_writer: Callable[[Path], None] | None = None,
) -> dict[str, str]:
    """Save measurements first; record visualization failure without hiding success."""
    frame_analysis = fuse_frame_analysis(
        frame_basis, phase_by_frame, pose_keypoints, joint_angles, plate_track
    )
    saved_path = Path(output_path)
    save_frame_analysis(frame_analysis, saved_path)
    result = {"measurement_status": "ok", "visualization_status": "not_run", "visualization_reason": ""}
    if visualization_writer is None:
        return result
    try:
        visualization_writer(saved_path)
    except Exception as error:  # visualization must not invalidate saved measurements
        result.update(visualization_status="failed", visualization_reason=str(error))
    else:
        result["visualization_status"] = "ok"
    return result


def _validate_basis(frame_basis: pd.DataFrame) -> None:
    basis = frame_basis
    missing = sorted(set(_KEY_COLUMNS) - set(basis.columns))
    if missing:
        raise ValueError("frame basis columns missing: " + ", ".join(missing))
    if basis.empty:
        raise ValueError("frame basis must not be empty")
    if basis.duplicated(_KEY_COLUMNS).any():
        raise ValueError("frame basis has duplicate frame keys")
    if basis["video_id"].astype(str).nunique() != 1:
        raise ValueError("frame basis must contain exactly one video_id")
    if not basis["frame_index"].is_monotonic_increasing:
        raise ValueError("frame basis frame_index must be ordered")
    if basis["frame_index"].tolist() != list(range(len(basis))):
        raise ValueError("frame basis frame_index must be zero-based contiguous")


def _validate_direct_table(
    table: pd.DataFrame,
    table_name: str,
    video_id: str,
    basis: pd.DataFrame,
) -> None:
    missing = sorted(set(_KEY_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"{table_name} columns missing: " + ", ".join(missing))
    if table.duplicated(_KEY_COLUMNS).any():
        raise ValueError(f"duplicate frame keys in {table_name}")
    if table.empty:
        return
    if set(table["video_id"].astype(str)) != {video_id}:
        raise ValueError(f"{table_name} video_id does not match frame basis")
    valid_frames = set(basis["frame_index"])
    if not table["frame_index"].isin(valid_frames).all():
        raise ValueError(f"{table_name} frame_index is outside frame basis")


def _without_timestamp(table: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns=["timestamp_seconds"], errors="ignore")


def _rename_plate_columns(table: pd.DataFrame) -> pd.DataFrame:
    renamed = _without_timestamp(table)
    return renamed.rename(columns={"status": "plate_status", "reason": "plate_reason"})
