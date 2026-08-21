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

_EVENT_COLUMNS = ["video_id", "event_name", "frame_index", "annotator", "notes"]


def validate_phase_events(
    event_frames: Mapping[str, object],
    frame_count: int,
) -> dict[str, int]:
    """Validate and return the five manual event frames in boundary order."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    missing = [name for name in EVENT_NAMES if name not in event_frames]
    if missing:
        raise ValueError(f"missing events: {', '.join(missing)}")

    extra = [name for name in event_frames if name not in EVENT_NAMES]
    if extra:
        raise ValueError(f"unknown events: {', '.join(extra)}")

    validated = {}
    for name in EVENT_NAMES:
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

    ordered_frames = [validated[name] for name in EVENT_NAMES]
    if any(left >= right for left, right in zip(ordered_frames, ordered_frames[1:])):
        raise ValueError("event frames must be strictly increasing")

    return validated


def build_phase_by_frame(
    video_id: str,
    event_frames: Mapping[str, object],
    frame_count: int,
) -> pd.DataFrame:
    """Expand five validated event frames into one phase row per video frame."""
    events = validate_phase_events(event_frames, frame_count)
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
        else:
            phase = "Recovery"

        rows.append(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "phase": phase,
                "recovery_complete": frame_index >= events["recovery_stable"],
            }
        )

    return pd.DataFrame(rows)


def prepare_phase_annotations(
    *,
    video_id: str,
    event_frames: Mapping[str, object],
    frame_count: int,
    annotation_path: str | Path,
    artifact_directory: str | Path,
    annotator: str = "",
    notes: str = "",
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Protect the editable source and write validated per-run snapshots."""
    annotation_path = Path(annotation_path)
    artifact_directory = Path(artifact_directory)
    existing = _read_annotation_source(annotation_path)
    existing_video = existing.loc[existing["video_id"] == video_id]

    if not existing_video.empty and not overwrite:
        active_events = _canonical_event_table(existing_video, video_id, frame_count)
        action = "reused"
    else:
        missing_events = [
            name for name in EVENT_NAMES if event_frames.get(name) is None
        ]
        if missing_events:
            raise ValueError(
                "다섯 이벤트 프레임을 직접 입력하세요. 비어 있는 이벤트: "
                + ", ".join(missing_events)
            )
        validated = validate_phase_events(event_frames, frame_count)
        active_events = _build_event_table(
            video_id,
            validated,
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

    active_mapping = dict(
        zip(active_events["event_name"], active_events["frame_index"], strict=True)
    )
    phase_by_frame = build_phase_by_frame(video_id, active_mapping, frame_count)

    artifact_directory.mkdir(parents=True, exist_ok=True)
    active_events.to_csv(artifact_directory / "phase_events.csv", index=False)
    phase_by_frame.to_csv(artifact_directory / "phase_by_frame.csv", index=False)
    return active_events, phase_by_frame, action


def _read_annotation_source(annotation_path: Path) -> pd.DataFrame:
    if not annotation_path.exists():
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    events = pd.read_csv(annotation_path)
    missing_columns = [name for name in _EVENT_COLUMNS if name not in events.columns]
    if missing_columns:
        raise ValueError(
            f"phase annotation columns missing: {', '.join(missing_columns)}"
        )
    return events[_EVENT_COLUMNS]


def _canonical_event_table(
    event_table: pd.DataFrame,
    video_id: str,
    frame_count: int,
) -> pd.DataFrame:
    if event_table["event_name"].duplicated().any():
        raise ValueError(f"duplicate event names for video_id={video_id}")

    indexed = event_table.set_index("event_name")
    event_frames = indexed["frame_index"].to_dict()
    validated = validate_phase_events(event_frames, frame_count)
    canonical = indexed.loc[list(EVENT_NAMES)].reset_index()
    canonical["video_id"] = video_id
    canonical["frame_index"] = [validated[name] for name in EVENT_NAMES]
    canonical["annotator"] = canonical["annotator"].fillna("").astype(str)
    canonical["notes"] = canonical["notes"].fillna("").astype(str)
    return canonical[_EVENT_COLUMNS]


def _build_event_table(
    video_id: str,
    event_frames: Mapping[str, int],
    *,
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
            }
            for name in EVENT_NAMES
        ],
        columns=_EVENT_COLUMNS,
    )
