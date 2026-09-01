"""Manual snatch events and deterministic phase assignment."""

from collections.abc import Mapping
import math
from numbers import Integral
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


EVENT_NAMES = (
    "bar_off_floor",
    "knee_pass",
    "max_extension",
    "catch_stable",
    "recovery_stable",
)

_EVENT_COLUMNS = [
    "video_id",
    "event_name",
    "frame_index",
    "annotator",
    "notes",
    "na_reason",
]


def propose_bar_off_floor_event(
    plate_track: pd.DataFrame,
    *,
    upward_speed_threshold: float = 0.1,
    persistence_frames: int = 2,
    max_missing_gap_frames: int = 2,
) -> dict[str, object]:
    """Propose the first frame of sustained upward plate motion."""
    required_columns = {
        "frame_index",
        "timestamp_seconds",
        "center_y_normalized",
        "status",
    }
    missing_columns = sorted(required_columns - set(plate_track.columns))
    if missing_columns:
        raise ValueError(
            "plate track columns missing: " + ", ".join(missing_columns)
        )
    if isinstance(upward_speed_threshold, bool):
        raise ValueError("upward_speed_threshold must be positive")
    try:
        normalized_speed_threshold = float(upward_speed_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("upward_speed_threshold must be positive") from error
    if (
        not math.isfinite(normalized_speed_threshold)
        or normalized_speed_threshold <= 0
    ):
        raise ValueError("upward_speed_threshold must be positive")
    if (
        isinstance(persistence_frames, bool)
        or not isinstance(persistence_frames, Integral)
        or persistence_frames <= 0
    ):
        raise ValueError("persistence_frames must be a positive integer")
    if (
        isinstance(max_missing_gap_frames, bool)
        or not isinstance(max_missing_gap_frames, Integral)
        or max_missing_gap_frames <= 0
    ):
        raise ValueError("max_missing_gap_frames must be a positive integer")

    track = plate_track[
        [
            "frame_index",
            "timestamp_seconds",
            "center_y_normalized",
            "status",
        ]
    ].copy()
    for column in ("frame_index", "timestamp_seconds", "center_y_normalized"):
        track[column] = pd.to_numeric(track[column], errors="coerce")

    result = {
        "event_name": "bar_off_floor",
        "frame_index": None,
        "timestamp_seconds": None,
        "status": "not_available",
        "reason": "no_sustained_upward_motion",
        "upward_speed_threshold": normalized_speed_threshold,
        "persistence_frames": int(persistence_frames),
        "max_missing_gap_frames": int(max_missing_gap_frames),
    }
    qualifying_steps = 0
    unavailable_gap = 0
    run_start = None
    previous = None
    previous_frame_index = None
    for current in track.itertuples(index=False):
        frame_index_is_valid = (
            pd.notna(current.frame_index)
            and math.isfinite(float(current.frame_index))
            and float(current.frame_index).is_integer()
        )
        current_frame_index = (
            int(current.frame_index) if frame_index_is_valid else None
        )
        current_is_valid = (
            not pd.isna(current.status)
            and current.status == "ok"
            and frame_index_is_valid
            and all(
                pd.notna(value) and math.isfinite(float(value))
                for value in (
                    current.timestamp_seconds,
                    current.center_y_normalized,
                )
            )
        )
        if current_frame_index is not None and previous_frame_index is not None:
            if current_frame_index > previous_frame_index:
                unavailable_gap += max(
                    0, current_frame_index - previous_frame_index - 1
                )
            else:
                unavailable_gap += 1

        if not current_is_valid:
            unavailable_gap += 1
            qualifying_steps = 0
            run_start = None
            if unavailable_gap > max_missing_gap_frames:
                result["reason"] = "plate_gap_too_long"
                return result
            previous = current
            previous_frame_index = current_frame_index
            continue

        if unavailable_gap:
            qualifying_steps = 0
            run_start = None
            if unavailable_gap > max_missing_gap_frames:
                result["reason"] = "plate_gap_too_long"
                return result
            unavailable_gap = 0
            previous = current
            previous_frame_index = current_frame_index
            continue

        if previous is None:
            previous = current
            previous_frame_index = current_frame_index
            continue

        if current.timestamp_seconds <= previous.timestamp_seconds:
            unavailable_gap = 1
            qualifying_steps = 0
            run_start = None
            previous = current
            previous_frame_index = current_frame_index
            continue

        upward_speed = (
            previous.center_y_normalized - current.center_y_normalized
        ) / (current.timestamp_seconds - previous.timestamp_seconds)
        if upward_speed < normalized_speed_threshold:
            qualifying_steps = 0
            run_start = None
            previous = current
            previous_frame_index = current_frame_index
            continue

        if qualifying_steps == 0:
            run_start = current
        qualifying_steps += 1
        if qualifying_steps >= persistence_frames:
            result.update(
                {
                    "frame_index": int(run_start.frame_index),
                    "timestamp_seconds": float(run_start.timestamp_seconds),
                    "status": "ok",
                    "reason": "",
                }
            )
            return result
        previous = current
        previous_frame_index = current_frame_index

    return result


def summarize_bar_off_floor_timing_experiment(
    trials: pd.DataFrame,
) -> dict[str, object]:
    """Summarize the fixed A/B/C manual-versus-assisted timing experiment."""
    required_columns = (
        "video_id",
        "group",
        "method",
        "review_seconds",
        "status",
    )
    missing_columns = sorted(set(required_columns) - set(trials.columns))
    if missing_columns:
        raise ValueError(
            "timing experiment columns missing: " + ", ".join(missing_columns)
        )

    experiment = trials[list(required_columns)].copy()
    if experiment["video_id"].duplicated().any():
        raise ValueError("timing experiment video_id must be unique")
    experiment["group"] = experiment["group"].astype(str).str.upper()
    experiment["method"] = experiment["method"].astype(str).str.lower()
    experiment["status"] = experiment["status"].astype(str).str.lower()

    invalid_groups = sorted(set(experiment["group"]) - {"A", "B", "C"})
    invalid_methods = sorted(
        set(experiment["method"]) - {"manual", "assisted"}
    )
    invalid_statuses = sorted(
        set(experiment["status"]) - {"pending", "complete", "failed"}
    )
    if invalid_groups:
        raise ValueError(
            "unknown timing experiment group: " + ", ".join(invalid_groups)
        )
    if invalid_methods:
        raise ValueError(
            "unknown timing experiment method: " + ", ".join(invalid_methods)
        )
    if invalid_statuses:
        raise ValueError(
            "unknown timing experiment status: " + ", ".join(invalid_statuses)
        )
    if experiment.duplicated(["group", "method"]).any():
        raise ValueError("timing experiment allows one trial per group and method")

    complete = experiment.loc[experiment["status"].eq("complete")].copy()
    complete["review_seconds"] = pd.to_numeric(
        complete["review_seconds"], errors="coerce"
    )
    invalid_seconds = complete["review_seconds"].map(
        lambda value: pd.isna(value)
        or not math.isfinite(float(value))
        or float(value) <= 0
    )
    if invalid_seconds.any():
        raise ValueError("complete timing trials require positive review_seconds")
    manual = complete.loc[complete["method"].eq("manual"), "review_seconds"]
    assisted = complete.loc[
        complete["method"].eq("assisted"), "review_seconds"
    ]
    manual_median = float(manual.median()) if not manual.empty else None
    assisted_median = float(assisted.median()) if not assisted.empty else None
    reduction_ratio = (
        (manual_median - assisted_median) / manual_median
        if manual_median is not None and assisted_median is not None
        else None
    )

    faster_group_count = 0
    for group in ("A", "B", "C"):
        group_trials = complete.loc[complete["group"].eq(group)].set_index(
            "method"
        )
        if {"manual", "assisted"}.issubset(group_trials.index):
            faster_group_count += int(
                group_trials.loc["assisted", "review_seconds"]
                < group_trials.loc["manual", "review_seconds"]
            )

    expected_pairs = {
        (group, method)
        for group in "ABC"
        for method in ("manual", "assisted")
    }
    completed_pairs = set(zip(complete["group"], complete["method"]))
    is_complete = completed_pairs == expected_pairs
    gate_pass = bool(
        is_complete
        and reduction_ratio is not None
        and reduction_ratio >= 0.25
        and faster_group_count >= 2
    )
    return {
        "status": "complete" if is_complete else "incomplete",
        "complete_count": int(len(complete)),
        "expected_count": 6,
        "manual_median_seconds": manual_median,
        "assisted_median_seconds": assisted_median,
        "median_reduction_ratio": reduction_ratio,
        "faster_group_count": faster_group_count,
        "time_gate_pass": gate_pass,
        "reason": ""
        if gate_pass
        else "time_gate_not_met"
        if is_complete
        else "incomplete_trials",
    }


def validate_phase_events(
    event_frames: Mapping[str, object],
    frame_count: int,
    completion: str = "f",
) -> dict[str, int | None]:
    """Validate and return the five manual event frames in boundary order."""
    completion = _validate_completion(completion)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    missing = [name for name in EVENT_NAMES if name not in event_frames]
    if missing:
        raise ValueError(f"missing events: {', '.join(missing)}")

    extra = [name for name in event_frames if name not in EVENT_NAMES]
    if extra:
        raise ValueError(f"unknown events: {', '.join(extra)}")

    validated: dict[str, int | None] = {}
    for name in EVENT_NAMES[:-1]:
        value = event_frames[name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"event frame must be an integer: {name}={value!r}")
        frame_index = int(value)
        if not 0 <= frame_index < frame_count:
            raise ValueError(
                f"event outside frame range 0..{frame_count - 1}: "
                f"{name}={frame_index}"
            )
        validated[name] = frame_index

    recovery_frame = event_frames["recovery_stable"]
    if completion == "h":
        if recovery_frame is not None:
            raise ValueError("recovery_stable must be None for completion h")
        validated["recovery_stable"] = None
    else:
        if isinstance(recovery_frame, bool) or not isinstance(
            recovery_frame, Integral
        ):
            raise ValueError(
                "event frame must be an integer: "
                f"recovery_stable={recovery_frame!r}"
            )
        recovery_index = int(recovery_frame)
        if not 0 <= recovery_index < frame_count:
            raise ValueError(
                f"event outside frame range 0..{frame_count - 1}: "
                f"recovery_stable={recovery_index}"
            )
        validated["recovery_stable"] = recovery_index

    ordered_events = EVENT_NAMES if completion == "f" else EVENT_NAMES[:-1]
    ordered_frames = [validated[name] for name in ordered_events]
    if any(left >= right for left, right in zip(ordered_frames, ordered_frames[1:])):
        raise ValueError("event frames must be strictly increasing")

    return validated


def build_phase_by_frame(
    video_id: str,
    event_frames: Mapping[str, object],
    frame_count: int,
    completion: str = "f",
) -> pd.DataFrame:
    """Expand five validated event frames into one phase row per video frame."""
    events = validate_phase_events(event_frames, frame_count, completion)
    rows = []
    for frame_index in range(frame_count):
        if frame_index < events["bar_off_floor"]:
            phase = "Set-up"
        elif frame_index < events["knee_pass"]:
            phase = "First Pull"
        elif frame_index < events["max_extension"]:
            phase = "Second Pull"
        elif frame_index <= events["catch_stable"]:
            phase = "Drop Under & Catch"
        elif completion == "f":
            phase = "Recovery"
        else:
            phase = "N/A"

        phase_status = "ok" if phase != "N/A" else "not_applicable"
        na_reason = "" if phase != "N/A" else "video_ended"

        rows.append(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "phase": phase,
                "recovery_complete": (
                    frame_index >= events["recovery_stable"]
                    if completion == "f"
                    else pd.NA
                ),
                "phase_status": phase_status,
                "na_reason": na_reason,
            }
        )

    return pd.DataFrame(rows)


def prepare_phase_annotations(
    *,
    video_id: str,
    event_frames: Mapping[str, object],
    frame_count: int,
    completion: str = "f",
    annotation_path: str | Path,
    artifact_directory: str | Path,
    annotator: str = "",
    notes: str = "",
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Protect the editable source and write validated per-run snapshots."""
    completion = _validate_completion(completion)
    annotation_path = Path(annotation_path)
    artifact_directory = Path(artifact_directory)
    existing = _read_annotation_source(annotation_path)
    existing_video = existing.loc[existing["video_id"] == video_id]

    if not existing_video.empty and not overwrite:
        active_events = _canonical_event_table(
            existing_video, video_id, frame_count, completion
        )
        action = "reused"
    else:
        required_events = EVENT_NAMES if completion == "f" else EVENT_NAMES[:-1]
        missing_events = [
            name for name in required_events if event_frames.get(name) is None
        ]
        if missing_events:
            event_count = "다섯" if completion == "f" else "네"
            raise ValueError(
                f"{event_count} 이벤트 프레임을 직접 입력하세요. 비어 있는 이벤트: "
                + ", ".join(missing_events)
            )
        validated = validate_phase_events(event_frames, frame_count, completion)
        active_events = _build_event_table(
            video_id,
            validated,
            completion=completion,
            annotator=annotator,
            notes=notes,
        )
        remaining = existing.loc[existing["video_id"] != video_id]
        combined = pd.concat([remaining, active_events], ignore_index=True)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=annotation_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            combined.to_csv(temporary_path, index=False)
            temporary_path.replace(annotation_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        if existing_video.empty:
            action = "created" if existing.empty else "appended"
        else:
            action = "overwritten"

    active_mapping = _event_mapping_from_table(active_events)
    phase_by_frame = build_phase_by_frame(
        video_id, active_mapping, frame_count, completion
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    active_events.to_csv(artifact_directory / "phase_events.csv", index=False)
    phase_by_frame.to_csv(artifact_directory / "phase_by_frame.csv", index=False)
    return active_events, phase_by_frame, action


def _read_annotation_source(annotation_path: Path) -> pd.DataFrame:
    if not annotation_path.exists():
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    events = pd.read_csv(annotation_path)
    required_columns = _EVENT_COLUMNS[:-1]
    missing_columns = [name for name in required_columns if name not in events.columns]
    if missing_columns:
        raise ValueError(
            f"phase annotation columns missing: {', '.join(missing_columns)}"
        )
    if "na_reason" not in events.columns:
        events["na_reason"] = ""
    return events[_EVENT_COLUMNS]


def _canonical_event_table(
    event_table: pd.DataFrame,
    video_id: str,
    frame_count: int,
    completion: str,
) -> pd.DataFrame:
    if event_table["event_name"].duplicated().any():
        raise ValueError(f"duplicate event names for video_id={video_id}")

    indexed = event_table.set_index("event_name")
    validated = validate_phase_events(
        _event_mapping_from_table(event_table), frame_count, completion
    )
    canonical = indexed.loc[list(EVENT_NAMES)].reset_index()
    canonical["video_id"] = video_id
    canonical["frame_index"] = [validated[name] for name in EVENT_NAMES]
    canonical["annotator"] = canonical["annotator"].fillna("").astype(str)
    canonical["notes"] = canonical["notes"].fillna("").astype(str)
    canonical["na_reason"] = canonical["na_reason"].fillna("").astype(str)
    return canonical[_EVENT_COLUMNS]


def _event_mapping_from_table(event_table: pd.DataFrame) -> dict[str, object]:
    """Restore nullable integer event frames after a CSV round trip."""
    frames = {}
    for row in event_table[["event_name", "frame_index"]].itertuples(index=False):
        value = row.frame_index
        if pd.isna(value):
            frames[row.event_name] = None
        elif isinstance(value, float) and value.is_integer():
            frames[row.event_name] = int(value)
        else:
            frames[row.event_name] = value
    return frames


def _build_event_table(
    video_id: str,
    event_frames: Mapping[str, int | None],
    *,
    completion: str,
    annotator: str,
    notes: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": video_id,
                "event_name": name,
                "frame_index": event_frames[name],
                "annotator": annotator,
                "notes": notes,
                "na_reason": "video_ended"
                if completion == "h" and name == "recovery_stable"
                else "",
            }
            for name in EVENT_NAMES
        ],
        columns=_EVENT_COLUMNS,
    )


def _validate_completion(completion: str) -> str:
    if completion not in {"f", "h"}:
        raise ValueError(f"unknown completion: {completion!r}")
    return completion
