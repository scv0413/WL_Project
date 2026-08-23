"""Replay saved B00 analysis tables without model inference."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from snatch_analysis.video import iter_video_frames, read_video_frame, read_video_properties


_PHASES = ("Set-up", "First Pull", "Second Pull", "Drop Under & Catch", "Recovery")
_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def save_angle_plot(frame_analysis_path: str | Path, output_path: str | Path) -> None:
    """Plot saved angles; NaNs remain visible as gaps."""
    analysis = _read_analysis(frame_analysis_path)
    figure, axis = plt.subplots(figsize=(10, 5))
    for joint in ("knee", "hip", "elbow", "trunk"):
        column = f"{joint}_angle_deg"
        if column in analysis:
            axis.plot(analysis["frame_index"], pd.to_numeric(analysis[column], errors="coerce"), label=joint)
    axis.set(xlabel="Frame", ylabel="Angle (degrees)", title="Joint angles (saved measurements)")
    axis.legend()
    axis.grid(alpha=0.3)
    _save_figure(figure, output_path)


def save_plate_trajectory_plot(
    frame_analysis_path: str | Path,
    output_path: str | Path,
) -> tuple[np.ndarray, ...]:
    """Plot the selected plate center without connecting missing coordinates."""
    analysis = _read_analysis(frame_analysis_path)
    figure, axis = plt.subplots(figsize=(6, 8))
    segments = _plate_trajectory_segments(analysis)
    for segment in segments:
        axis.plot(segment[:, 0], segment[:, 1], marker="o", markersize=2, linewidth=1)
    axis.invert_yaxis()
    axis.set(xlabel="Plate center x (px)", ylabel="Plate center y (px)", title="Plate trajectory (gaps are missing detections)")
    axis.grid(alpha=0.3)
    _save_figure(figure, output_path)
    return segments


def save_phase_storyboard(
    video_path: str | Path,
    frame_analysis_path: str | Path,
    output_path: str | Path,
) -> None:
    """Save one representative decoded frame for each manual phase."""
    analysis = _read_analysis(frame_analysis_path)
    figure, axes = plt.subplots(1, len(_PHASES), figsize=(20, 8), squeeze=False)
    for axis, phase in zip(axes.flat, _PHASES, strict=True):
        matches = analysis.loc[analysis["phase"] == phase]
        if matches.empty:
            axis.text(0.5, 0.5, f"{phase}\nN/A: no phase frame", ha="center", va="center")
        else:
            row = matches.iloc[len(matches) // 2]
            frame = read_video_frame(video_path, int(row["frame_index"]))
            _draw_measurements(frame, row, pd.DataFrame())
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axis.set_title(f"{phase}\nrepresentative frame {int(row['frame_index'])}")
        axis.axis("off")
    _save_figure(figure, output_path)


def write_overlay(
    video_path: str | Path,
    frame_analysis_path: str | Path,
    pose_keypoints_path: str | Path,
    output_path: str | Path,
) -> None:
    """Write a same-size/fps overlay from cached Pose, plate, and phase tables."""
    analysis = _read_analysis(frame_analysis_path)
    pose = pd.read_csv(pose_keypoints_path)
    properties = read_video_properties(video_path)
    if properties["read_status"] != "ok":
        raise RuntimeError(f"Could not read overlay input video: {video_path}")
    width, height, fps = int(properties["width"]), int(properties["height"]), float(properties["fps"])
    _validate_overlay_tables(analysis, pose, int(properties["frame_count"]))
    analysis = analysis.set_index("frame_index", drop=False)
    pose = pose.set_index("frame_index", drop=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, suffix=".mp4", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not open overlay writer: {output_path}")
    trajectory_segments: list[list[tuple[int, int]]] = []
    current_trajectory: list[tuple[int, int]] | None = None
    try:
        for frame_index, _, frame in iter_video_frames(video_path, expected_frame_count=len(analysis)):
            if frame_index not in analysis.index:
                raise ValueError(f"frame_analysis is missing frame {frame_index}")
            frame_pose = pose.loc[[frame_index]]
            row = analysis.loc[frame_index]
            _draw_measurements(frame, row, frame_pose)
            plate_center = _plate_center(row)
            if plate_center is None:
                current_trajectory = None
            else:
                if current_trajectory is None:
                    current_trajectory = [plate_center]
                    trajectory_segments.append(current_trajectory)
                else:
                    current_trajectory.append(plate_center)
                drawable_segments = [
                    np.asarray(segment, dtype=np.int32)
                    for segment in trajectory_segments
                    if len(segment) > 1
                ]
                if drawable_segments:
                    cv2.polylines(frame, drawable_segments, False, (255, 180, 0), 2)
            if frame.shape[:2] != (height, width):
                raise RuntimeError("decoded overlay frame shape changed")
            writer.write(frame)
    except BaseException:
        try:
            writer.release()
        finally:
            temporary_path.unlink(missing_ok=True)
        raise
    else:
        try:
            writer.release()
            _validate_written_overlay(
                temporary_path,
                expected_frame_count=len(analysis),
                expected_width=width,
                expected_height=height,
                expected_fps=fps,
            )
            temporary_path.replace(output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def _read_analysis(path: str | Path) -> pd.DataFrame:
    analysis = pd.read_csv(path)
    missing = {"frame_index", "phase"} - set(analysis.columns)
    if missing:
        raise ValueError("frame_analysis columns missing: " + ", ".join(sorted(missing)))
    return analysis.sort_values("frame_index").reset_index(drop=True)


def _plate_trajectory_segments(analysis: pd.DataFrame) -> tuple[np.ndarray, ...]:
    if not {"center_x_px", "center_y_px"}.issubset(analysis.columns):
        return ()
    segments: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    for row in analysis.itertuples(index=False):
        x, y = getattr(row, "center_x_px"), getattr(row, "center_y_px")
        if pd.isna(x) or pd.isna(y):
            if current:
                segments.append(np.asarray(current))
                current = []
        else:
            current.append((float(x), float(y)))
    if current:
        segments.append(np.asarray(current))
    return tuple(segments)


def _validate_overlay_tables(
    analysis: pd.DataFrame,
    pose: pd.DataFrame,
    frame_count: int,
) -> None:
    analysis_required = {"video_id", "frame_index", "phase"}
    pose_required = {"video_id", "frame_index", "joint_name"}
    missing_analysis = sorted(analysis_required - set(analysis.columns))
    missing_pose = sorted(pose_required - set(pose.columns))
    if missing_analysis:
        raise ValueError("frame_analysis columns missing: " + ", ".join(missing_analysis))
    if missing_pose:
        raise ValueError("pose_keypoints columns missing: " + ", ".join(missing_pose))
    if analysis.duplicated("frame_index").any():
        raise ValueError("duplicate frame keys in frame_analysis")
    if pose.duplicated(["video_id", "frame_index", "joint_name"]).any():
        raise ValueError("duplicate long-form pose keys")
    if analysis["video_id"].astype(str).nunique() != 1:
        raise ValueError("frame_analysis must contain exactly one video_id")
    video_id = str(analysis.iloc[0]["video_id"])
    if set(pose["video_id"].astype(str)) != {video_id}:
        raise ValueError("pose_keypoints video_id does not match frame_analysis")
    expected_frames = set(range(frame_count))
    if set(analysis["frame_index"]) != expected_frames:
        raise ValueError("frame_analysis frame coverage does not match decoded video")
    if set(pose["frame_index"]) != expected_frames:
        raise ValueError("pose_keypoints frame coverage does not match decoded video")


def _validate_written_overlay(
    video_path: Path,
    *,
    expected_frame_count: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
) -> None:
    properties = read_video_properties(video_path)
    fps_tolerance = max(0.01, expected_fps * 0.001)
    if (
        properties["read_status"] != "ok"
        or properties["frame_count"] != expected_frame_count
        or properties["width"] != expected_width
        or properties["height"] != expected_height
        or properties["fps"] is None
        or abs(float(properties["fps"]) - expected_fps) > fps_tolerance
    ):
        raise RuntimeError("overlay output does not match the decoded input contract")
    decoded_count = 0
    for _, _, frame in iter_video_frames(video_path, expected_frame_count=expected_frame_count):
        if frame.shape != (expected_height, expected_width, 3):
            raise RuntimeError("overlay output frame shape does not match the decoded input")
        decoded_count += 1
    if decoded_count != expected_frame_count:
        raise RuntimeError("overlay output frame count does not match the decoded input")


def _save_figure(figure: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
    finally:
        plt.close(figure)


def _draw_measurements(frame: np.ndarray, row: pd.Series, pose: pd.DataFrame) -> None:
    previous = {}
    if not pose.empty and {"joint_index", "x", "y", "status"}.issubset(pose.columns):
        valid = pose.loc[pose["status"] == "ok"]
        for joint in valid.itertuples(index=False):
            if pd.notna(joint.x) and pd.notna(joint.y):
                point = (round(float(joint.x)), round(float(joint.y)))
                previous[int(joint.joint_index)] = point
                cv2.circle(frame, point, 3, (0, 255, 0), -1)
        for start, end in _SKELETON:
            if start in previous and end in previous:
                cv2.line(frame, previous[start], previous[end], (0, 200, 0), 2)
    bbox_columns = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    if all(column in row.index and pd.notna(row[column]) for column in bbox_columns):
        cv2.rectangle(frame, (round(row["bbox_x1"]), round(row["bbox_y1"])), (round(row["bbox_x2"]), round(row["bbox_y2"])), (255, 180, 0), 2)
    plate_center = _plate_center(row)
    if plate_center is not None:
        cv2.circle(frame, plate_center, 4, (255, 180, 0), -1)
    phase = str(row.get("phase", "N/A"))
    plate_status = str(row.get("plate_status", "missing"))
    cv2.putText(frame, phase, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Pose: {row.get('pose_status', 'not_available')}  Plate: {plate_status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def _plate_center(row: pd.Series) -> tuple[int, int] | None:
    if all(column in row.index and pd.notna(row[column]) for column in ("center_x_px", "center_y_px")):
        return (round(float(row["center_x_px"])), round(float(row["center_y_px"])))
    return None
