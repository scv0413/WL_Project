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
    "split_role",
]

_MANUAL_METADATA_DEFAULTS = {
    "has_plate": "unknown",
    "camera_motion": "unknown",
    "visible_body_side": "unknown",
    "split_role": "unassigned",
}
VISIBLE_BODY_SIDE_VALUES = frozenset({"left", "right", "uncertain", "unknown"})
HAS_PLATE_VALUES = frozenset({"yes", "no", "unknown"})
SPLIT_ROLE_VALUES = frozenset(
    {
        "unassigned",
        "training_development",
        "model_selection_pilot",
        "rule_development",
        "rule_blind",
    }
)
EVALUATION_ROLE_VALUES = frozenset(
    {"model_selection_pilot", "rule_development", "rule_blind"}
)
EVALUATION_SPLIT_COLUMNS = (
    "video_id",
    "person_id",
    "group",
    "split_role",
    "completion",
    "path",
    "fps",
    "frame_count",
    "width",
    "height",
    "has_plate",
    "visible_body_side",
    "read_status",
)


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
    return _set_video_category_labels(
        manifest,
        labels_by_video_id,
        column="visible_body_side",
        allowed_values=VISIBLE_BODY_SIDE_VALUES,
    )


def set_has_plate_labels(
    manifest: pd.DataFrame,
    labels_by_video_id: dict[str, str],
) -> pd.DataFrame:
    """video_id별 원판 유무 라벨을 검증해 반영한다."""
    return _set_video_category_labels(
        manifest,
        labels_by_video_id,
        column="has_plate",
        allowed_values=HAS_PLATE_VALUES,
    )


def set_split_roles_by_person(
    manifest: pd.DataFrame,
    roles_by_person_id: dict[str, str],
) -> pd.DataFrame:
    """같은 사람의 모든 영상을 하나의 평가 역할로 묶는다."""
    required_columns = {"video_id", "person_id"}
    missing = sorted(required_columns - set(manifest.columns))
    if missing:
        raise ValueError("Manifest columns missing: " + ", ".join(missing))
    if manifest["video_id"].duplicated().any():
        raise ValueError("Manifest의 video_id가 중복되었습니다.")

    normalized_roles = {
        str(person_id): str(role).strip().lower()
        for person_id, role in roles_by_person_id.items()
    }
    invalid_roles = sorted(set(normalized_roles.values()) - SPLIT_ROLE_VALUES)
    if invalid_roles:
        raise ValueError(
            "split_role 값이 올바르지 않습니다: " + ", ".join(invalid_roles)
        )
    known_people = set(manifest["person_id"].astype(str))
    unknown_people = sorted(set(normalized_roles) - known_people)
    if unknown_people:
        raise ValueError(
            "Manifest에 없는 person_id입니다: " + ", ".join(unknown_people)
        )

    labeled = manifest.copy()
    if "split_role" not in labeled.columns:
        labeled["split_role"] = "unassigned"
    updates = labeled["person_id"].astype(str).map(normalized_roles)
    labeled["split_role"] = updates.combine_first(labeled["split_role"])
    labeled["split_role"] = (
        labeled["split_role"]
        .fillna("unassigned")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    invalid_existing = sorted(set(labeled["split_role"]) - SPLIT_ROLE_VALUES)
    if invalid_existing:
        raise ValueError(
            "split_role 값이 올바르지 않습니다: " + ", ".join(invalid_existing)
        )
    roles_per_person = labeled.groupby("person_id")["split_role"].nunique()
    if (roles_per_person > 1).any():
        raise ValueError("같은 person_id에는 하나의 split_role만 허용합니다.")
    return labeled


def build_evaluation_video_split(
    manifest: pd.DataFrame,
    selected_video_ids: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """수동 확인된 영상만 사람 누수 없는 평가 split으로 고정한다."""
    missing_columns = [
        column for column in EVALUATION_SPLIT_COLUMNS if column not in manifest.columns
    ]
    if missing_columns:
        raise ValueError("Manifest columns missing: " + ", ".join(missing_columns))
    if manifest["video_id"].duplicated().any():
        raise ValueError("Manifest의 video_id가 중복되었습니다.")

    roles_per_person = manifest.groupby("person_id")["split_role"].nunique()
    leaked_people = roles_per_person.loc[roles_per_person > 1].index.astype(str).tolist()
    if leaked_people:
        raise ValueError(
            "person_id가 여러 split_role에 포함되었습니다: "
            + ", ".join(leaked_people)
        )

    selected_ids = [str(video_id) for video_id in selected_video_ids]
    if not selected_ids:
        raise ValueError("selected_video_ids가 비어 있습니다.")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected_video_ids가 중복되었습니다.")
    manifest_by_video = manifest.set_index("video_id", drop=False)
    unknown_ids = sorted(set(selected_ids) - set(manifest_by_video.index.astype(str)))
    if unknown_ids:
        raise ValueError("Manifest에 없는 video_id입니다: " + ", ".join(unknown_ids))

    selected = manifest_by_video.loc[selected_ids, list(EVALUATION_SPLIT_COLUMNS)].copy()
    selected = selected.reset_index(drop=True)
    invalid_selected_roles = sorted(
        set(selected["split_role"]) - EVALUATION_ROLE_VALUES
    )
    if invalid_selected_roles:
        raise ValueError(
            "evaluation split에는 training_development/unassigned 영상을 넣을 수 "
            "없습니다: " + ", ".join(invalid_selected_roles)
        )
    if selected["person_id"].duplicated().any():
        raise ValueError("evaluation split은 person_id당 영상 하나만 허용합니다.")
    expected_people = set(
        manifest.loc[
            manifest["split_role"].isin(EVALUATION_ROLE_VALUES),
            "person_id",
        ].astype(str)
    )
    selected_people = set(selected["person_id"].astype(str))
    if selected_people != expected_people:
        missing_people = sorted(expected_people - selected_people)
        raise ValueError(
            "평가 person_id마다 영상 하나를 선택하세요: "
            + ", ".join(missing_people)
        )
    if (selected["read_status"] != "ok").any():
        raise ValueError("평가 영상의 read_status는 모두 ok여야 합니다.")
    if not selected["has_plate"].isin({"yes", "no"}).all():
        raise ValueError("평가 영상의 has_plate는 yes/no여야 합니다.")
    if not selected["visible_body_side"].isin(
        {"left", "right", "uncertain"}
    ).all():
        raise ValueError(
            "평가 영상의 visible_body_side를 left/right/uncertain으로 확정하세요."
        )

    pilot = selected.loc[selected["split_role"] == "model_selection_pilot"]
    if len(pilot) != 3 or pilot["group"].value_counts().to_dict() != {
        "A": 1,
        "B": 1,
        "C": 1,
    }:
        raise ValueError(
            "model_selection_pilot은 A/B/C 영상이 각각 하나씩 필요합니다."
        )
    if (pilot["completion"] != "f").any():
        raise ValueError(
            "U12 전 model_selection_pilot은 completion=f 영상을 선택하세요."
        )
    return selected


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
        if column == "split_role" and "person_id" in previous.columns:
            roles_per_person = previous.groupby("person_id")[column].nunique()
            if (roles_per_person > 1).any():
                raise ValueError(
                    "기존 Manifest의 같은 person_id에 여러 split_role이 있습니다."
                )
            role_by_person = (
                previous.reset_index()
                .drop_duplicates("person_id")
                .set_index("person_id")[column]
            )
            values = preserved["person_id"].map(role_by_person)
        else:
            values = preserved["video_id"].map(previous[column])
        preserved[column] = values.fillna(default)

    visible_labels = dict(
        zip(
            preserved["video_id"].astype(str),
            preserved["visible_body_side"].astype(str),
            strict=True,
        )
    )
    preserved = set_visible_body_side_labels(preserved, visible_labels)
    plate_labels = dict(
        zip(
            preserved["video_id"].astype(str),
            preserved["has_plate"].astype(str),
            strict=True,
        )
    )
    preserved = set_has_plate_labels(preserved, plate_labels)
    return set_split_roles_by_person(preserved, {})


def _set_video_category_labels(
    manifest: pd.DataFrame,
    labels_by_video_id: dict[str, str],
    *,
    column: str,
    allowed_values: frozenset[str],
) -> pd.DataFrame:
    if "video_id" not in manifest.columns:
        raise ValueError("Manifest에 video_id 열이 없습니다.")
    if manifest["video_id"].duplicated().any():
        raise ValueError("Manifest의 video_id가 중복되었습니다.")

    normalized_labels = {
        str(video_id): str(label).strip().lower()
        for video_id, label in labels_by_video_id.items()
    }
    invalid_labels = sorted(set(normalized_labels.values()) - allowed_values)
    if invalid_labels:
        raise ValueError(
            f"{column} 값이 올바르지 않습니다: " + ", ".join(invalid_labels)
        )
    unknown_video_ids = sorted(
        set(normalized_labels) - set(manifest["video_id"].astype(str))
    )
    if unknown_video_ids:
        raise ValueError(
            "Manifest에 없는 video_id입니다: " + ", ".join(unknown_video_ids)
        )

    labeled = manifest.copy()
    default = _MANUAL_METADATA_DEFAULTS[column]
    if column not in labeled.columns:
        labeled[column] = default
    updates = labeled["video_id"].astype(str).map(normalized_labels)
    labeled[column] = updates.combine_first(labeled[column])
    labeled[column] = (
        labeled[column].fillna(default).astype(str).str.strip().str.lower()
    )
    invalid_existing = sorted(set(labeled[column]) - allowed_values)
    if invalid_existing:
        raise ValueError(
            f"{column} 값이 올바르지 않습니다: " + ", ".join(invalid_existing)
        )
    return labeled


def _build_manifest_row(video_path: Path) -> dict[str, object]:
    row = {
        **parse_video_filename(video_path.name),
        "filename": video_path.name,
        "path": video_path.as_posix(),
        "file_size_bytes": video_path.stat().st_size,
        **_MANUAL_METADATA_DEFAULTS,
    }
    return {**row, **read_video_properties(video_path)}
