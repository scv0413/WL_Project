from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import pandas as pd

from snatch_analysis.plate import (
    SAMPLE_PHASES,
    opencv_bbox_to_yolo,
    sample_phase_frames,
    save_plate_image,
    save_yolo_plate_label,
    yolo_bbox_to_opencv,
)


class PlateSamplingTests(unittest.TestCase):
    def test_samples_unique_frames_across_all_five_phases(self) -> None:
        phase_by_frame = pd.DataFrame(
            {
                "video_id": ["b00_sn_f_01"] * 20,
                "frame_index": list(range(20)),
                "phase": [phase for phase in SAMPLE_PHASES for _ in range(4)],
            }
        )

        sampled = sample_phase_frames(phase_by_frame, sample_count=10)

        self.assertEqual(len(sampled), 10)
        self.assertFalse(sampled["frame_index"].duplicated().any())
        self.assertEqual(
            sampled["sampling_phase"].value_counts().to_dict(),
            {phase: 2 for phase in SAMPLE_PHASES},
        )
        self.assertFalse(sampled["setup_reference_override"].any())

    def test_uses_frame_zero_as_setup_reference_when_setup_is_empty(self) -> None:
        phases = (
            ["First Pull"] * 5
            + ["Second Pull"] * 5
            + ["Drop Under & Catch"] * 5
            + ["Recovery"] * 5
        )
        phase_by_frame = pd.DataFrame(
            {
                "video_id": ["b00_sn_f_01"] * 20,
                "frame_index": list(range(20)),
                "phase": phases,
            }
        )

        sampled = sample_phase_frames(phase_by_frame, sample_count=10)
        setup_row = sampled.loc[sampled["sampling_phase"] == "Set-up"].iloc[0]

        self.assertEqual(set(sampled["sampling_phase"]), set(SAMPLE_PHASES))
        self.assertEqual(int(setup_row["frame_index"]), 0)
        self.assertEqual(setup_row["analysis_phase"], "First Pull")
        self.assertTrue(bool(setup_row["setup_reference_override"]))
        self.assertFalse(sampled["frame_index"].duplicated().any())

    def test_rejects_duplicate_frame_rows(self) -> None:
        phase_by_frame = pd.DataFrame(
            {
                "video_id": ["b00_sn_f_01", "b00_sn_f_01"],
                "frame_index": [0, 0],
                "phase": ["Set-up", "First Pull"],
            }
        )

        with self.assertRaisesRegex(ValueError, "frame_index"):
            sample_phase_frames(phase_by_frame, sample_count=1)

    def test_rejects_fractional_frame_indices_before_integer_conversion(self) -> None:
        phase_by_frame = pd.DataFrame(
            {
                "video_id": ["b00_sn_f_01"] * 5,
                "frame_index": [0, 1.5, 2, 3, 4],
                "phase": list(SAMPLE_PHASES),
            }
        )

        with self.assertRaisesRegex(ValueError, "finite integers"):
            sample_phase_frames(phase_by_frame, sample_count=5)


class PlateBoundingBoxTests(unittest.TestCase):
    def test_converts_opencv_bbox_to_normalized_yolo_and_back(self) -> None:
        yolo_bbox = opencv_bbox_to_yolo(
            (20, 10, 40, 20),
            image_width=100,
            image_height=50,
        )

        self.assertEqual(yolo_bbox, (0.4, 0.4, 0.4, 0.4))
        self.assertEqual(
            yolo_bbox_to_opencv(
                yolo_bbox,
                image_width=100,
                image_height=50,
            ),
            (20, 10, 40, 20),
        )

    def test_rejects_cancelled_negative_and_out_of_frame_bboxes(self) -> None:
        invalid_bboxes = (
            (0, 0, 0, 0),
            (-1, 0, 10, 10),
            (95, 0, 10, 10),
            (0, 45, 10, 10),
            (0, 0, float("nan"), 10),
        )

        with TemporaryDirectory() as temporary_directory:
            for index, bbox in enumerate(invalid_bboxes):
                label_path = Path(temporary_directory) / f"invalid_{index}.txt"
                with self.subTest(bbox=bbox):
                    with self.assertRaises(ValueError):
                        save_yolo_plate_label(
                            label_path,
                            bbox,
                            image_width=100,
                            image_height=50,
                        )
                    self.assertFalse(label_path.exists())

    def test_existing_image_and_label_are_not_overwritten_by_default(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "images" / "frame_000.jpg"
            label_path = root / "labels" / "frame_000.txt"
            black_frame = np.zeros((20, 30, 3), dtype=np.uint8)
            white_frame = np.full((20, 30, 3), 255, dtype=np.uint8)

            self.assertEqual(
                save_plate_image(image_path, black_frame),
                "created",
            )
            self.assertEqual(
                save_yolo_plate_label(
                    label_path,
                    (5, 5, 10, 10),
                    image_width=30,
                    image_height=20,
                ),
                "created",
            )
            original_label = label_path.read_text(encoding="utf-8")

            self.assertEqual(
                save_plate_image(image_path, white_frame),
                "skipped",
            )
            self.assertEqual(
                save_yolo_plate_label(
                    label_path,
                    (0, 0, 30, 20),
                    image_width=30,
                    image_height=20,
                ),
                "skipped",
            )

            saved_frame = cv2.imread(str(image_path))
            self.assertLess(float(saved_frame.mean()), 1.0)
            self.assertEqual(
                label_path.read_text(encoding="utf-8"),
                original_label,
            )


if __name__ == "__main__":
    unittest.main()
