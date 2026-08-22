from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from snatch_analysis.metadata import (
    parse_video_filename,
    save_video_manifest,
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


if __name__ == "__main__":
    unittest.main()
