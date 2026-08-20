"""Video filename parsing and manifest generation."""

from pathlib import Path
import re

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
]


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def _build_manifest_row(video_path: Path) -> dict[str, object]:
    row = {
        **parse_video_filename(video_path.name),
        "filename": video_path.name,
        "path": video_path.as_posix(),
        "file_size_bytes": video_path.stat().st_size,
        "has_plate": "unknown",
        "camera_motion": "unknown",
    }
    return {**row, **read_video_properties(video_path)}
