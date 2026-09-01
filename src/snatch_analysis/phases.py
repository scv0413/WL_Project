"""Manual snatch events and deterministic phase assignment."""

from collections.abc import Mapping
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
