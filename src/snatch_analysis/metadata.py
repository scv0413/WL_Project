"""Video filename parsing and manifest generation."""

from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import pandas as pd

from snatch_analysis.video import read_video_properties


_FILENAME_PATTERN = re.compile(
    r"^(?P<group>[abc])(?P<person_num>\d{2})_"
    r"(?P<lift>sn|psn)_"
    r"(?P<completion>[fh])_"
    r"(?P<rep_index>\d{2})\.mp4$",
    re.IGNORECASE,
)

_MANIFEST_COLUMNS = [
    "video_id",
    "filename",
    "path",
    "group",
    "person_id",
    "lift",
    "completion",
    "rep_index",
    "file_size_bytes",
    "fps",
    "frame_count",
    "duration_sec",
    "width",
    "height",
    "read_status",
    "has_plate",
    "camera_motion",
    "visible_body_side",
]

_MANUAL_METADATA_DEFAULTS = {
    "has_plate": "unknown",
    "camera_motion": "unknown",
    "visible_body_side": "unknown",
}
VISIBLE_BODY_SIDE_VALUES = frozenset({"left", "right", "uncertain", "unknown"})


def parse_video_filename(filename: str | Path) -> dict[str, object]:
    """Parse one project filename into stable metadata fields."""
    name = Path(filename).name
    match = _FILENAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid video filename: {name}")

    values = match.groupdict()
    group = values["group"].upper()
    person_id = f"{values['group'].lower()}{values['person_num']}"
    video_id = Path(name).stem.lower()

    return {
        "video_id": video_id,
        "group": group,
        "person_id": person_id,
        "lift": values["lift"].lower(),
        "completion": values["completion"].lower(),
        "rep_index": int(values["rep_index"]),
    }


def write_video_manifest(video_directory: Path, output_path: Path) -> pd.DataFrame:
    """Inspect every MP4 in a directory and save a CSV manifest."""
    video_directory = Path(video_directory)
    output_path = Path(output_path)
    if not video_directory.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_directory}")

    video_paths = sorted(video_directory.glob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No MP4 files found in: {video_directory}")

    rows = [_build_manifest_row(path) for path in video_paths]
    manifest = pd.DataFrame(rows, columns=_MANIFEST_COLUMNS)
    manifest = _preserve_manual_metadata(manifest, output_path)
    save_video_manifest(manifest, output_path)
    return manifest


def save_video_manifest(manifest: pd.DataFrame, output_path: Path) -> None:
    """완성된 manifest만 기존 CSV와 원자적으로 교체한다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output_path.parent,
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        manifest.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def set_visible_body_side_labels(
    manifest: pd.DataFrame,
    labels_by_video_id: dict[str, str],
) -> pd.DataFrame:
    """video_id별 수동 촬영 신체면 라벨을 검증해 반영한다."""
    if "video_id" not in manifest.columns:
        raise ValueError("Manifest에 video_id 열이 없습니다.")
    if manifest["video_id"].duplicated().any():
        raise ValueError("Manifest의 video_id가 중복되었습니다.")

    normalized_labels = {
        str(video_id): str(label).strip().lower()
        for video_id, label in labels_by_video_id.items()
    }
    invalid_labels = sorted(
        set(normalized_labels.values()) - VISIBLE_BODY_SIDE_VALUES
    )
    if invalid_labels:
        raise ValueError(
            "visible_body_side는 left/right/uncertain/unknown만 허용합니다: "
            + ", ".join(invalid_labels)
        )

    unknown_video_ids = sorted(
        set(normalized_labels) - set(manifest["video_id"].astype(str))
    )
    if unknown_video_ids:
        raise ValueError(
            "Manifest에 없는 video_id입니다: " + ", ".join(unknown_video_ids)
        )

    labeled = manifest.copy()
    if "visible_body_side" not in labeled.columns:
        labeled["visible_body_side"] = _MANUAL_METADATA_DEFAULTS[
            "visible_body_side"
        ]
    label_updates = labeled["video_id"].astype(str).map(normalized_labels)
    labeled["visible_body_side"] = label_updates.combine_first(
        labeled["visible_body_side"]
    )
    labeled["visible_body_side"] = (
        labeled["visible_body_side"]
        .fillna(_MANUAL_METADATA_DEFAULTS["visible_body_side"])
        .astype(str)
        .str.strip()
        .str.lower()
    )
    invalid_labels = sorted(
        set(labeled["visible_body_side"]) - VISIBLE_BODY_SIDE_VALUES
    )
    if invalid_labels:
        raise ValueError(
            "visible_body_side는 left/right/uncertain/unknown만 허용합니다: "
            + ", ".join(invalid_labels)
        )
    return labeled


def _preserve_manual_metadata(
    manifest: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    try:
        previous = pd.read_csv(output_path)
    except FileNotFoundError:
        return manifest
    if "video_id" not in previous.columns or previous["video_id"].duplicated().any():
        raise ValueError("기존 Manifest의 video_id가 없거나 중복되었습니다.")
    previous = previous.set_index("video_id")
    preserved = manifest.copy()
    for column, default in _MANUAL_METADATA_DEFAULTS.items():
        if column not in previous.columns:
            continue
        values = preserved["video_id"].map(previous[column])
        preserved[column] = values.fillna(default)

    visible_labels = dict(
        zip(
            preserved["video_id"].astype(str),
            preserved["visible_body_side"].astype(str),
            strict=True,
        )
    )
    return set_visible_body_side_labels(preserved, visible_labels)


def _build_manifest_row(video_path: Path) -> dict[str, object]:
    row = {
        **parse_video_filename(video_path.name),
        "filename": video_path.name,
        "path": video_path.as_posix(),
        "file_size_bytes": video_path.stat().st_size,
        **_MANUAL_METADATA_DEFAULTS,
    }
    return {**row, **read_video_properties(video_path)}
