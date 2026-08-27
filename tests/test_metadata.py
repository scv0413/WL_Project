from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from snatch_analysis.metadata import (
    EVALUATION_SPLIT_COLUMNS,
    build_evaluation_video_split,
    parse_video_filename,
    save_video_manifest,
    set_has_plate_labels,
    set_split_roles_by_person,
    set_visible_body_side_labels,
    write_video_manifest,
)


class ParseVideoFilenameTests(unittest.TestCase):
    def test_parses_project_filename_fields(self) -> None:
        metadata = parse_video_filename("a04_sn_h_02.mp4")

        self.assertEqual(
            metadata,
            {
                "video_id": "a04_sn_h_02",
                "group": "A",
                "person_id": "a04",
                "lift": "sn",
                "completion": "h",
                "rep_index": 2,
            },
        )

    def test_rejects_filename_outside_project_rule(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid video filename"):
            parse_video_filename("athlete_a04_snatch.mp4")


class WriteVideoManifestTests(unittest.TestCase):
    def test_writes_one_row_with_real_video_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_directory = root / "videos"
            output_path = root / "metadata" / "videos.csv"
            video_directory.mkdir()
            video_path = video_directory / "c01_sn_f_01.mp4"
            self._write_test_video(video_path)

            manifest = write_video_manifest(video_directory, output_path)
            saved_manifest = pd.read_csv(output_path)

            self.assertEqual(len(manifest), 1)
            self.assertEqual(len(saved_manifest), 1)
            self.assertEqual(saved_manifest.loc[0, "video_id"], "c01_sn_f_01")
            self.assertEqual(saved_manifest.loc[0, "group"], "C")
            self.assertEqual(saved_manifest.loc[0, "person_id"], "c01")
            self.assertEqual(saved_manifest.loc[0, "read_status"], "ok")
            self.assertEqual(saved_manifest.loc[0, "frame_count"], 5)
            self.assertEqual(saved_manifest.loc[0, "width"], 64)
            self.assertEqual(saved_manifest.loc[0, "height"], 48)
            self.assertAlmostEqual(saved_manifest.loc[0, "fps"], 25.0, places=1)
            self.assertAlmostEqual(saved_manifest.loc[0, "duration_sec"], 0.2, places=2)
            self.assertEqual(saved_manifest.loc[0, "has_plate"], "unknown")
            self.assertEqual(saved_manifest.loc[0, "camera_motion"], "unknown")
            self.assertEqual(saved_manifest.loc[0, "visible_body_side"], "unknown")
            self.assertEqual(saved_manifest.loc[0, "split_role"], "unassigned")

    def test_rebuild_preserves_manual_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_directory = root / "videos"
            output_path = root / "metadata" / "videos.csv"
            video_directory.mkdir()
            self._write_test_video(video_directory / "c01_sn_f_01.mp4")
            manifest = write_video_manifest(video_directory, output_path)
            manifest.loc[0, ["has_plate", "camera_motion", "visible_body_side"]] = [
                "yes",
                "fixed",
                "left",
            ]
            manifest.to_csv(output_path, index=False)

            rebuilt = write_video_manifest(video_directory, output_path)

            self.assertEqual(rebuilt.loc[0, "has_plate"], "yes")
            self.assertEqual(rebuilt.loc[0, "camera_motion"], "fixed")
            self.assertEqual(rebuilt.loc[0, "visible_body_side"], "left")

    def test_rebuild_applies_person_split_role_to_a_new_repeat(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_directory = root / "videos"
            output_path = root / "metadata" / "videos.csv"
            video_directory.mkdir()
            self._write_test_video(video_directory / "c01_sn_f_01.mp4")
            manifest = write_video_manifest(video_directory, output_path)
            manifest.loc[0, "split_role"] = "rule_blind"
            manifest.to_csv(output_path, index=False)
            self._write_test_video(video_directory / "c01_sn_f_02.mp4")

            rebuilt = write_video_manifest(video_directory, output_path)

            self.assertEqual(set(rebuilt["split_role"]), {"rule_blind"})

    def test_sets_visible_body_side_labels_by_video_id(self) -> None:
        manifest = pd.DataFrame(
            {"video_id": ["a01_sn_f_01", "b00_sn_f_01"]}
        )

        labeled = set_visible_body_side_labels(
            manifest,
            {"a01_sn_f_01": "left", "b00_sn_f_01": "uncertain"},
        )

        self.assertEqual(
            labeled["visible_body_side"].tolist(),
            ["left", "uncertain"],
        )

    def test_rejects_invalid_visible_body_side_label(self) -> None:
        manifest = pd.DataFrame({"video_id": ["b00_sn_f_01"]})

        with self.assertRaisesRegex(ValueError, "visible_body_side"):
            set_visible_body_side_labels(manifest, {"b00_sn_f_01": "front"})

    def test_rejects_label_for_unknown_video_id(self) -> None:
        manifest = pd.DataFrame({"video_id": ["b00_sn_f_01"]})

        with self.assertRaisesRegex(ValueError, "Manifest에 없는 video_id"):
            set_visible_body_side_labels(manifest, {"c99_sn_f_01": "right"})

    def test_rejects_existing_invalid_visible_body_side_label(self) -> None:
        manifest = pd.DataFrame(
            {
                "video_id": ["a01_sn_f_01", "b00_sn_f_01"],
                "visible_body_side": ["front", "unknown"],
            }
        )

        with self.assertRaisesRegex(ValueError, "visible_body_side"):
            set_visible_body_side_labels(manifest, {"b00_sn_f_01": "left"})

    def test_atomic_save_keeps_existing_manifest_when_csv_write_fails(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "videos.csv"
            output_path.write_text("previous manifest\n", encoding="utf-8")
            manifest = pd.DataFrame({"video_id": ["b00_sn_f_01"]})

            with patch.object(
                pd.DataFrame,
                "to_csv",
                side_effect=OSError("disk write failed"),
            ):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    save_video_manifest(manifest, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "previous manifest\n",
            )
            self.assertEqual(list(output_path.parent.glob("*.tmp")), [])

    @staticmethod
    def _write_test_video(path: Path) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (64, 48),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the test MP4")
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        for _ in range(5):
            writer.write(frame)
        writer.release()


class EvaluationSplitTests(unittest.TestCase):
    def test_assigns_a_person_role_to_all_of_their_videos(self) -> None:
        manifest = pd.DataFrame(
            {
                "video_id": ["a01_sn_f_01", "a01_sn_h_02", "b01_sn_f_01"],
                "person_id": ["a01", "a01", "b01"],
            }
        )

        labeled = set_split_roles_by_person(
            manifest,
            {
                "a01": "model_selection_pilot",
                "b01": "rule_blind",
            },
        )

        self.assertEqual(
            labeled["split_role"].tolist(),
            ["model_selection_pilot", "model_selection_pilot", "rule_blind"],
        )

    def test_rejects_an_invalid_split_role(self) -> None:
        manifest = pd.DataFrame(
            {"video_id": ["a01_sn_f_01"], "person_id": ["a01"]}
        )

        with self.assertRaisesRegex(ValueError, "split_role"):
            set_split_roles_by_person(manifest, {"a01": "validation"})

    def test_builds_one_confirmed_abc_model_selection_video_each(self) -> None:
        manifest = self._evaluation_manifest()

        split = build_evaluation_video_split(
            manifest,
            ["a01_sn_f_01", "b01_sn_f_01", "c01_sn_f_01"],
        )

        self.assertEqual(tuple(split.columns), EVALUATION_SPLIT_COLUMNS)
        self.assertEqual(split["group"].tolist(), ["A", "B", "C"])
        self.assertEqual(
            set(split["split_role"]),
            {"model_selection_pilot"},
        )

    def test_requires_f_videos_for_the_pre_u12_model_pilot(self) -> None:
        manifest = self._evaluation_manifest()
        manifest.loc[manifest["group"] == "A", "completion"] = "h"

        with self.assertRaisesRegex(ValueError, "completion=f"):
            build_evaluation_video_split(
                manifest,
                ["a01_sn_f_01", "b01_sn_f_01", "c01_sn_f_01"],
            )

    def test_rejects_person_leakage_between_split_roles(self) -> None:
        manifest = self._evaluation_manifest()
        leaked = pd.concat(
            [
                manifest,
                manifest.iloc[[0]].assign(
                    video_id="a01_sn_f_02",
                    split_role="rule_blind",
                ),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "person_id.*split_role"):
            build_evaluation_video_split(
                leaked,
                ["a01_sn_f_01", "b01_sn_f_01", "c01_sn_f_01"],
            )

    def test_rejects_unconfirmed_manual_metadata(self) -> None:
        manifest = self._evaluation_manifest()
        manifest.loc[manifest["video_id"] == "b01_sn_f_01", "has_plate"] = (
            "unknown"
        )

        with self.assertRaisesRegex(ValueError, "has_plate"):
            build_evaluation_video_split(
                manifest,
                ["a01_sn_f_01", "b01_sn_f_01", "c01_sn_f_01"],
            )

    def test_rejects_invalid_manual_metadata_values(self) -> None:
        for column, invalid_value in (
            ("has_plate", "maybe"),
            ("visible_body_side", "front"),
        ):
            manifest = self._evaluation_manifest()
            manifest.loc[manifest["video_id"] == "b01_sn_f_01", column] = (
                invalid_value
            )

            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, column):
                    build_evaluation_video_split(
                        manifest,
                        ["a01_sn_f_01", "b01_sn_f_01", "c01_sn_f_01"],
                    )

    def test_rejects_training_video_from_evaluation_split(self) -> None:
        manifest = self._evaluation_manifest()
        training = manifest.iloc[[0]].assign(
            video_id="a02_sn_f_01",
            person_id="a02",
            split_role="training_development",
        )
        manifest = pd.concat([manifest, training], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "training_development"):
            build_evaluation_video_split(
                manifest,
                [
                    "a01_sn_f_01",
                    "b01_sn_f_01",
                    "c01_sn_f_01",
                    "a02_sn_f_01",
                ],
            )

    def test_requires_one_video_for_each_evaluation_person(self) -> None:
        manifest = self._evaluation_manifest()
        rule_blind = manifest.iloc[[0]].assign(
            video_id="a02_sn_f_01",
            person_id="a02",
            split_role="rule_blind",
        )
        manifest = pd.concat([manifest, rule_blind], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "평가 person_id"):
            build_evaluation_video_split(
                manifest,
                ["a01_sn_f_01", "b01_sn_f_01", "c01_sn_f_01"],
            )

    def test_sets_has_plate_labels_by_video_id(self) -> None:
        manifest = pd.DataFrame(
            {"video_id": ["a01_sn_f_01", "b01_sn_f_01"]}
        )

        labeled = set_has_plate_labels(
            manifest,
            {"a01_sn_f_01": "yes", "b01_sn_f_01": "no"},
        )

        self.assertEqual(labeled["has_plate"].tolist(), ["yes", "no"])

    @staticmethod
    def _evaluation_manifest() -> pd.DataFrame:
        rows = []
        for group in ("A", "B", "C"):
            prefix = group.lower()
            rows.append(
                {
                    "video_id": f"{prefix}01_sn_f_01",
                    "person_id": f"{prefix}01",
                    "group": group,
                    "split_role": "model_selection_pilot",
                    "completion": "f",
                    "path": f"data/raw/videos/{prefix}01_sn_f_01.mp4",
                    "fps": 30.0,
                    "frame_count": 100,
                    "width": 1920,
                    "height": 1080,
                    "has_plate": "yes",
                    "visible_body_side": "left",
                    "read_status": "ok",
                }
            )
        return pd.DataFrame(rows)

if __name__ == "__main__":
    unittest.main()
