from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import cv2
import numpy as np
import pandas as pd

import snatch_analysis.plate as plate_module
from snatch_analysis.plate import (
    PILOT_GROUND_TRUTH_COLUMNS,
    PLATE_CANDIDATE_COLUMNS,
    PLATE_TRACK_COLUMNS,
    SAMPLE_PHASES,
    build_pilot_ground_truth_template,
    opencv_bbox_to_yolo,
    plate_result_to_candidates,
    sample_phase_frames,
    save_pilot_ground_truth,
    save_plate_candidate_cache,
    save_plate_image,
    save_plate_track,
    save_yolo_plate_label,
    select_primary_plate_track,
    validate_pilot_ground_truth,
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


class PilotGroundTruthTests(unittest.TestCase):
    def test_builds_and_saves_thirty_validated_frame_labels(self) -> None:
        evaluation_split = self._evaluation_split()
        ground_truth = self._completed_ground_truth(evaluation_split)

        validated = validate_pilot_ground_truth(
            ground_truth,
            evaluation_split,
        )

        self.assertEqual(tuple(validated.columns), PILOT_GROUND_TRUTH_COLUMNS)
        self.assertEqual(len(validated), 30)
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "pilot_ground_truth.csv"
            save_pilot_ground_truth(validated, evaluation_split, output_path)
            saved = pd.read_csv(output_path)
        self.assertEqual(len(saved), 30)

    def test_rejects_partial_model_selection_pilot(self) -> None:
        evaluation_split = self._evaluation_split().iloc[[0]].copy()
        samples = self._sample_frames(evaluation_split)

        with self.assertRaisesRegex(ValueError, "A/B/C"):
            build_pilot_ground_truth_template(samples, evaluation_split)

    def test_rejects_duplicate_video_frame_keys(self) -> None:
        evaluation_split = self._evaluation_split()
        ground_truth = self._completed_ground_truth(evaluation_split)
        ground_truth.loc[1, "frame_index"] = ground_truth.loc[0, "frame_index"]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_pilot_ground_truth(ground_truth, evaluation_split)

    def test_rejects_bbox_outside_the_frame(self) -> None:
        evaluation_split = self._evaluation_split()
        ground_truth = self._completed_ground_truth(evaluation_split)
        ground_truth.loc[0, "plate_bbox_x"] = 95

        with self.assertRaisesRegex(ValueError, "outside"):
            validate_pilot_ground_truth(ground_truth, evaluation_split)

    def test_rejects_missing_bbox_without_na_reason(self) -> None:
        evaluation_split = self._evaluation_split()
        samples = self._sample_frames(evaluation_split)
        template = build_pilot_ground_truth_template(samples, evaluation_split)

        with self.assertRaisesRegex(ValueError, "bbox.*na_reason"):
            validate_pilot_ground_truth(template, evaluation_split)

    def test_no_plate_video_keeps_lifter_bbox_and_plate_na(self) -> None:
        evaluation_split = self._evaluation_split(has_plate="no")
        ground_truth = self._completed_ground_truth(evaluation_split)
        plate_columns = [
            "plate_bbox_x",
            "plate_bbox_y",
            "plate_bbox_width",
            "plate_bbox_height",
        ]
        ground_truth.loc[:, plate_columns] = np.nan
        ground_truth.loc[:, "na_reason"] = "no_plate"

        validated = validate_pilot_ground_truth(
            ground_truth,
            evaluation_split,
        )

        self.assertTrue(validated[plate_columns].isna().all().all())
        self.assertFalse(validated["lifter_bbox_x"].isna().any())

    def test_no_plate_bbox_requires_no_plate_reason(self) -> None:
        evaluation_split = self._evaluation_split(has_plate="no")
        ground_truth = self._completed_ground_truth(evaluation_split)
        plate_columns = [
            "plate_bbox_x",
            "plate_bbox_y",
            "plate_bbox_width",
            "plate_bbox_height",
        ]
        ground_truth.loc[:, plate_columns] = np.nan
        ground_truth.loc[:, "na_reason"] = ""

        with self.assertRaisesRegex(ValueError, "no_plate"):
            validate_pilot_ground_truth(ground_truth, evaluation_split)

    def test_no_plate_reason_cannot_hide_a_missing_lifter_bbox(self) -> None:
        evaluation_split = self._evaluation_split(has_plate="no")
        ground_truth = self._completed_ground_truth(evaluation_split)
        ground_truth.loc[:, [
            "plate_bbox_x",
            "plate_bbox_y",
            "plate_bbox_width",
            "plate_bbox_height",
            "lifter_bbox_x",
            "lifter_bbox_y",
            "lifter_bbox_width",
            "lifter_bbox_height",
        ]] = np.nan
        ground_truth.loc[:, "na_reason"] = "no_plate"

        with self.assertRaisesRegex(ValueError, "lifter bbox.*na_reason"):
            validate_pilot_ground_truth(ground_truth, evaluation_split)

    def test_validation_does_not_depend_on_dataframe_index(self) -> None:
        evaluation_split = self._evaluation_split()
        ground_truth = self._completed_ground_truth(evaluation_split)
        ground_truth.index = range(10, 40)

        validated = validate_pilot_ground_truth(ground_truth, evaluation_split)

        self.assertEqual(len(validated), 30)

    def test_keeps_plate_and_lifter_cancellation_reasons(self) -> None:
        evaluation_split = self._evaluation_split()
        ground_truth = self._completed_ground_truth(evaluation_split)
        ground_truth.loc[0, [
            "plate_bbox_x",
            "plate_bbox_y",
            "plate_bbox_width",
            "plate_bbox_height",
        ]] = np.nan
        ground_truth.loc[0, "na_reason"] = "plate_label_cancelled"
        ground_truth.loc[1, [
            "lifter_bbox_x",
            "lifter_bbox_y",
            "lifter_bbox_width",
            "lifter_bbox_height",
        ]] = np.nan
        ground_truth.loc[1, "na_reason"] = "lifter_label_cancelled"

        validated = validate_pilot_ground_truth(ground_truth, evaluation_split)

        self.assertEqual(validated.loc[0, "na_reason"], "plate_label_cancelled")
        self.assertEqual(validated.loc[1, "na_reason"], "lifter_label_cancelled")

    @staticmethod
    def _evaluation_split(has_plate: str = "yes") -> pd.DataFrame:
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
                    "frame_count": 20,
                    "width": 100,
                    "height": 50,
                    "has_plate": has_plate,
                    "visible_body_side": "left",
                    "read_status": "ok",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _sample_frames(evaluation_split: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"video_id": video_id, "frame_index": frame_index}
                for video_id in evaluation_split["video_id"]
                for frame_index in range(10)
            ]
        )

    @classmethod
    def _completed_ground_truth(
        cls,
        evaluation_split: pd.DataFrame,
    ) -> pd.DataFrame:
        samples = cls._sample_frames(evaluation_split)
        ground_truth = build_pilot_ground_truth_template(
            samples,
            evaluation_split,
        )
        ground_truth.loc[:, [
            "plate_bbox_x",
            "plate_bbox_y",
            "plate_bbox_width",
            "plate_bbox_height",
        ]] = [10, 10, 20, 20]
        ground_truth.loc[:, [
            "lifter_bbox_x",
            "lifter_bbox_y",
            "lifter_bbox_width",
            "lifter_bbox_height",
        ]] = [30, 0, 50, 50]
        ground_truth.loc[:, "annotator"] = "coach"
        return ground_truth

class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = confidence


class _FakeResult:
    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray) -> None:
        self.boxes = _FakeBoxes(xyxy, confidence)


def _plate_candidates(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=PLATE_CANDIDATE_COLUMNS)


class PlateTrackingTests(unittest.TestCase):
    def test_infers_plate_candidates_with_streaming_results(self) -> None:
        runner = getattr(plate_module, "infer_plate_video_candidates", None)
        self.assertTrue(callable(runner), "infer_plate_video_candidates is missing")

        class FakePlateModel:
            def __init__(self) -> None:
                self.call = None

            def predict(self, **kwargs):
                self.call = kwargs
                return iter(
                    [
                        _FakeResult(
                            np.array([[10.0, 10.0, 30.0, 30.0]]),
                            np.array([0.9]),
                        ),
                        _FakeResult(
                            np.array([[12.0, 10.0, 32.0, 30.0]]),
                            np.array([0.8]),
                        ),
                    ]
                )

        model = FakePlateModel()
        candidates, processed_frame_count = runner(
            model,
            video_path=Path("video.mp4"),
            video_id="a01_sn_f_01",
            frame_count=2,
            fps=30.0,
            image_width=100,
            image_height=50,
            image_size=320,
            confidence=0.25,
            device="cpu",
        )

        self.assertEqual(processed_frame_count, 2)
        self.assertTrue(model.call["stream"])
        self.assertEqual(model.call["imgsz"], 320)
        self.assertEqual(candidates["frame_index"].tolist(), [0, 1])
        self.assertEqual(candidates["center_x_px"].tolist(), [20.0, 22.0])

    def test_rejects_plate_stream_frame_count_mismatch(self) -> None:
        class ShortPlateModel:
            def predict(self, **kwargs):
                return iter([
                    _FakeResult(
                        np.array([[10.0, 10.0, 30.0, 30.0]]),
                        np.array([0.9]),
                    )
                ])

        with self.assertRaisesRegex(
            RuntimeError, "expected=2, actual=1"
        ):
            plate_module.infer_plate_video_candidates(
                ShortPlateModel(),
                video_path=Path("short.mp4"),
                video_id="a01_sn_f_01",
                frame_count=2,
                fps=30.0,
                image_width=100,
                image_height=50,
                image_size=320,
                confidence=0.25,
                device="cpu",
            )

    def test_evaluates_candidate_track_and_gap_against_manual_bboxes(self) -> None:
        evaluator = getattr(plate_module, "evaluate_plate_video", None)
        self.assertTrue(callable(evaluator), "evaluate_plate_video is missing")
        ground_truth = pd.DataFrame(
            {
                "video_id": ["a01_sn_f_01", "a01_sn_f_01"],
                "frame_index": [0, 1],
                "plate_present": ["yes", "yes"],
                "plate_bbox_x": [0.0, 0.0],
                "plate_bbox_y": [0.0, 0.0],
                "plate_bbox_width": [10.0, 10.0],
                "plate_bbox_height": [10.0, 10.0],
            }
        )
        candidates = _plate_candidates(
            [
                {
                    "video_id": "a01_sn_f_01", "frame_index": 0,
                    "timestamp_seconds": 0.0, "candidate_id": 0,
                    "bbox_x1": 20.0, "bbox_y1": 20.0, "bbox_x2": 30.0,
                    "bbox_y2": 30.0, "bbox_confidence": 0.9,
                    "center_x_px": 25.0, "center_y_px": 25.0,
                    "center_x_normalized": 0.25, "center_y_normalized": 0.25,
                },
                {
                    "video_id": "a01_sn_f_01", "frame_index": 0,
                    "timestamp_seconds": 0.0, "candidate_id": 1,
                    "bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 10.0,
                    "bbox_y2": 10.0, "bbox_confidence": 0.8,
                    "center_x_px": 5.0, "center_y_px": 5.0,
                    "center_x_normalized": 0.05, "center_y_normalized": 0.05,
                },
                {
                    "video_id": "a01_sn_f_01", "frame_index": 1,
                    "timestamp_seconds": 0.1, "candidate_id": 0,
                    "bbox_x1": 20.0, "bbox_y1": 20.0, "bbox_x2": 30.0,
                    "bbox_y2": 30.0, "bbox_confidence": 0.7,
                    "center_x_px": 25.0, "center_y_px": 25.0,
                    "center_x_normalized": 0.25, "center_y_normalized": 0.25,
                },
            ]
        )
        track = pd.DataFrame(
            [
                {**candidates.iloc[1].to_dict(), "status": "ok", "reason": ""},
                {**candidates.iloc[2].to_dict(), "status": "ok", "reason": ""},
                {
                    **{column: np.nan for column in PLATE_CANDIDATE_COLUMNS},
                    "video_id": "a01_sn_f_01", "frame_index": 2,
                    "timestamp_seconds": 0.2, "status": "missing",
                    "reason": "no_detection",
                },
            ],
            columns=PLATE_TRACK_COLUMNS,
        )

        frame_evaluation, summary = evaluator(
            ground_truth,
            candidates,
            track,
            frame_count=3,
            fps=10.0,
            iou_threshold=0.5,
        )

        self.assertEqual(frame_evaluation["candidate_outcome"].tolist(), ["correct", "wrong_target"])
        self.assertEqual(frame_evaluation["track_outcome"].tolist(), ["correct", "wrong_target"])
        self.assertEqual(frame_evaluation["candidate_best_id"].tolist(), [1, 0])
        self.assertAlmostEqual(frame_evaluation.loc[1, "normalized_center_error"], 2.0)
        self.assertEqual(summary["track_ok_frame_count"], 2)
        self.assertAlmostEqual(summary["plate_ok_rate"], 2 / 3)
        self.assertEqual(summary["longest_non_ok_gap_frames"], 1)
        self.assertAlmostEqual(summary["longest_non_ok_gap_seconds"], 0.1)
        self.assertAlmostEqual(summary["candidate_recall"], 0.5)
        self.assertAlmostEqual(summary["track_correct_rate"], 0.5)
        self.assertAlmostEqual(summary["wrong_target_rate"], 0.5)

    def test_evaluates_missing_candidates_and_track_without_fabricating_values(self) -> None:
        evaluator = getattr(plate_module, "evaluate_plate_video", None)
        self.assertTrue(callable(evaluator), "evaluate_plate_video is missing")
        ground_truth = pd.DataFrame(
            {
                "video_id": ["a01_sn_f_01"],
                "frame_index": [1],
                "plate_present": ["yes"],
                "plate_bbox_x": [0.0],
                "plate_bbox_y": [0.0],
                "plate_bbox_width": [10.0],
                "plate_bbox_height": [10.0],
            }
        )

        try:
            frame_evaluation, summary = evaluator(
                ground_truth,
                pd.DataFrame(columns=PLATE_CANDIDATE_COLUMNS),
                None,
                frame_count=3,
                fps=10.0,
                iou_threshold=0.5,
                missing_track_reason="primary_plate_not_detected",
            )
        except Exception as error:
            self.fail(f"missing track must be measurable, got {type(error).__name__}: {error}")

        self.assertEqual(frame_evaluation.loc[0, "candidate_outcome"], "missing")
        self.assertEqual(frame_evaluation.loc[0, "track_outcome"], "missing")
        self.assertEqual(
            frame_evaluation.loc[0, "track_reason"],
            "primary_plate_not_detected",
        )
        self.assertTrue(pd.isna(frame_evaluation.loc[0, "track_iou"]))
        self.assertEqual(summary["track_ok_frame_count"], 0)
        self.assertEqual(summary["longest_non_ok_gap_frames"], 3)
        self.assertEqual(summary["candidate_recall"], 0.0)
        self.assertTrue(pd.isna(summary["wrong_target_rate"]))

    def test_marks_plate_metrics_not_applicable_for_no_plate_video(self) -> None:
        evaluator = getattr(plate_module, "evaluate_plate_video", None)
        self.assertTrue(callable(evaluator), "evaluate_plate_video is missing")
        ground_truth = pd.DataFrame(
            {
                "video_id": ["a01_sn_f_01"],
                "frame_index": [0],
                "plate_present": ["no"],
                "plate_bbox_x": [np.nan],
                "plate_bbox_y": [np.nan],
                "plate_bbox_width": [np.nan],
                "plate_bbox_height": [np.nan],
            }
        )

        try:
            frame_evaluation, summary = evaluator(
                ground_truth,
                pd.DataFrame(columns=PLATE_CANDIDATE_COLUMNS),
                None,
                frame_count=2,
                fps=10.0,
            )
        except Exception as error:
            self.fail(f"no-plate video must be N/A, got {type(error).__name__}: {error}")

        self.assertEqual(frame_evaluation.loc[0, "track_outcome"], "not_applicable")
        self.assertTrue(pd.isna(summary["plate_ok_rate"]))
        self.assertTrue(pd.isna(summary["candidate_recall"]))
        self.assertTrue(pd.isna(summary["track_correct_rate"]))
        self.assertEqual(summary["plate_applicable_frame_count"], 0)

    def test_aggregate_excludes_no_plate_video_from_track_coverage(self) -> None:
        per_video = pd.DataFrame(
            [
                {
                    "video_id": "a01_sn_f_01", "total_frame_count": 3,
                    "plate_applicable_frame_count": 3,
                    "track_ok_frame_count": 2, "longest_non_ok_gap_frames": 1,
                    "longest_non_ok_gap_seconds": 0.1, "ground_truth_frame_count": 1,
                    "candidate_correct_count": 1,
                    "track_detected_ground_truth_frame_count": 1,
                    "track_correct_count": 1, "wrong_target_count": 0,
                    "mean_normalized_center_error": 0.0,
                },
                {
                    "video_id": "c01_sn_f_01", "total_frame_count": 2,
                    "plate_applicable_frame_count": 0,
                    "track_ok_frame_count": 0, "longest_non_ok_gap_frames": np.nan,
                    "longest_non_ok_gap_seconds": np.nan, "ground_truth_frame_count": 0,
                    "candidate_correct_count": 0,
                    "track_detected_ground_truth_frame_count": 0,
                    "track_correct_count": 0, "wrong_target_count": 0,
                    "mean_normalized_center_error": np.nan,
                },
            ]
        )

        overall = plate_module.aggregate_plate_evaluations(per_video)

        self.assertEqual(overall["total_frame_count"], 5)
        self.assertEqual(overall["plate_applicable_frame_count"], 3)
        self.assertAlmostEqual(overall["plate_ok_rate"], 2 / 3)

    def test_aggregates_threshold_metrics_from_counts_not_rate_averages(self) -> None:
        aggregator = getattr(plate_module, "aggregate_plate_evaluations", None)
        self.assertTrue(callable(aggregator), "aggregate_plate_evaluations is missing")
        per_video = pd.DataFrame(
            [
                {
                    "video_id": "a01_sn_f_01", "total_frame_count": 3,
                    "plate_applicable_frame_count": 3,
                    "track_ok_frame_count": 2, "longest_non_ok_gap_frames": 1,
                    "longest_non_ok_gap_seconds": 0.1, "ground_truth_frame_count": 2,
                    "candidate_correct_count": 1,
                    "track_detected_ground_truth_frame_count": 2,
                    "track_correct_count": 1, "wrong_target_count": 1,
                    "mean_normalized_center_error": 1.0,
                },
                {
                    "video_id": "b01_sn_f_01", "total_frame_count": 2,
                    "plate_applicable_frame_count": 2,
                    "track_ok_frame_count": 0, "longest_non_ok_gap_frames": 2,
                    "longest_non_ok_gap_seconds": 0.2, "ground_truth_frame_count": 1,
                    "candidate_correct_count": 0,
                    "track_detected_ground_truth_frame_count": 0,
                    "track_correct_count": 0, "wrong_target_count": 0,
                    "mean_normalized_center_error": np.nan,
                },
            ]
        )

        overall = aggregator(per_video)

        self.assertEqual(overall["video_id"], "ALL")
        self.assertEqual(overall["video_count"], 2)
        self.assertAlmostEqual(overall["plate_ok_rate"], 0.4)
        self.assertEqual(overall["longest_non_ok_gap_frames"], 2)
        self.assertAlmostEqual(overall["candidate_recall"], 1 / 3)
        self.assertAlmostEqual(overall["track_correct_rate"], 1 / 3)
        self.assertAlmostEqual(overall["wrong_target_rate"], 0.5)
        self.assertAlmostEqual(overall["mean_normalized_center_error"], 1.0)

    def test_saves_plate_evaluation_tables_through_public_api(self) -> None:
        saver = getattr(plate_module, "save_plate_evaluation_results", None)
        self.assertTrue(callable(saver), "save_plate_evaluation_results is missing")
        frames = pd.DataFrame({"video_id": ["a01_sn_f_01"], "frame_index": [0]})
        comparison = pd.DataFrame({"experiment_id": ["conf010"]})

        with TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            saver(frames, comparison, output_root)

            saved_frames = pd.read_csv(output_root / "plate_frame_evaluation.csv")
            saved_comparison = pd.read_csv(output_root / "plate_threshold_comparison.csv")

        pd.testing.assert_frame_equal(saved_frames, frames)
        pd.testing.assert_frame_equal(saved_comparison, comparison)

    def test_tracking_accepts_object_typed_numeric_candidates_from_empty_frame_concat(self) -> None:
        candidates = _plate_candidates(
            [
                {
                    "video_id": "b00_sn_f_01", "frame_index": 0,
                    "timestamp_seconds": 0.0, "candidate_id": 0,
                    "bbox_x1": 40.0, "bbox_y1": 50.0, "bbox_x2": 60.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 50.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.5, "center_y_normalized": 0.6,
                },
                {
                    "video_id": "b00_sn_f_01", "frame_index": 1,
                    "timestamp_seconds": 0.1, "candidate_id": 0,
                    "bbox_x1": 42.0, "bbox_y1": 50.0, "bbox_x2": 62.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.8,
                    "center_x_px": 52.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.52, "center_y_normalized": 0.6,
                },
            ]
        ).astype(object)

        track = select_primary_plate_track(
            candidates,
            frame_count=2,
            initial_frame_index=0,
            initial_candidate_id=0,
        )

        self.assertEqual(track["status"].tolist(), ["ok", "ok"])

    def test_rejects_primary_plate_confirmation_outside_first_candidate_frame(self) -> None:
        first_frame = plate_result_to_candidates(
            _FakeResult(np.array([[10.0, 20.0, 30.0, 40.0]]), np.array([0.8])),
            "b00_sn_f_01", frame_index=0, timestamp_seconds=0.0,
            image_width=100, image_height=50,
        )
        later_frame = plate_result_to_candidates(
            _FakeResult(np.array([[11.0, 20.0, 31.0, 40.0]]), np.array([0.8])),
            "b00_sn_f_01", frame_index=1, timestamp_seconds=0.1,
            image_width=100, image_height=50,
        )

        with self.assertRaisesRegex(ValueError, "first candidate frame"):
            select_primary_plate_track(
                pd.concat([first_frame, later_frame], ignore_index=True),
                frame_count=2, initial_frame_index=1, initial_candidate_id=0,
            )

    def test_keeps_lifting_plate_when_candidate_id_changes(self) -> None:
        candidates = _plate_candidates(
            [
                {
                    "video_id": "b00_sn_f_01", "frame_index": 0,
                    "timestamp_seconds": 0.0, "candidate_id": 4,
                    "bbox_x1": 40.0, "bbox_y1": 50.0, "bbox_x2": 60.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 50.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.5, "center_y_normalized": 0.6,
                },
                {
                    "video_id": "b00_sn_f_01", "frame_index": 1,
                    "timestamp_seconds": 0.1, "candidate_id": 0,
                    "bbox_x1": 43.0, "bbox_y1": 52.0, "bbox_x2": 63.0,
                    "bbox_y2": 72.0, "bbox_confidence": 0.8,
                    "center_x_px": 53.0, "center_y_px": 62.0,
                    "center_x_normalized": 0.53, "center_y_normalized": 0.62,
                },
                {
                    "video_id": "b00_sn_f_01", "frame_index": 1,
                    "timestamp_seconds": 0.1, "candidate_id": 8,
                    "bbox_x1": 80.0, "bbox_y1": 50.0, "bbox_x2": 100.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.95,
                    "center_x_px": 90.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.9, "center_y_normalized": 0.6,
                },
            ]
        )
        lifter_centers = pd.DataFrame(
            {"frame_index": [0, 1], "center_x_px": [50.0, 54.0], "center_y_px": [60.0, 62.0]}
        )

        track = select_primary_plate_track(
            candidates, frame_count=2, initial_frame_index=0,
            initial_candidate_id=4, lifter_centers=lifter_centers,
        )

        self.assertEqual(track["candidate_id"].tolist(), [4, 0])
        self.assertEqual(track["status"].tolist(), ["ok", "ok"])

    def test_records_missing_frame_without_interpolating_coordinates(self) -> None:
        candidates = _plate_candidates(
            [
                {
                    "video_id": "b00_sn_f_01", "frame_index": 0,
                    "timestamp_seconds": 0.0, "candidate_id": 0,
                    "bbox_x1": 40.0, "bbox_y1": 50.0, "bbox_x2": 60.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 50.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.5, "center_y_normalized": 0.6,
                },
                {
                    "video_id": "b00_sn_f_01", "frame_index": 2,
                    "timestamp_seconds": 0.2, "candidate_id": 3,
                    "bbox_x1": 44.0, "bbox_y1": 50.0, "bbox_x2": 64.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 54.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.54, "center_y_normalized": 0.6,
                },
            ]
        )

        track = select_primary_plate_track(
            candidates, frame_count=3, initial_frame_index=0, initial_candidate_id=0,
        )

        missing = track.iloc[1]
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["reason"], "no_detection")
        self.assertTrue(pd.isna(missing["center_x_px"]))
        self.assertEqual(float(track.iloc[2]["center_x_px"]), 54.0)

    def test_stops_track_when_continuity_costs_are_ambiguous(self) -> None:
        candidates = _plate_candidates(
            [
                {
                    "video_id": "b00_sn_f_01", "frame_index": 0,
                    "timestamp_seconds": 0.0, "candidate_id": 0,
                    "bbox_x1": 40.0, "bbox_y1": 50.0, "bbox_x2": 60.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 50.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.5, "center_y_normalized": 0.6,
                },
                {
                    "video_id": "b00_sn_f_01", "frame_index": 1,
                    "timestamp_seconds": 0.1, "candidate_id": 1,
                    "bbox_x1": 42.0, "bbox_y1": 50.0, "bbox_x2": 62.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 52.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.52, "center_y_normalized": 0.6,
                },
                {
                    "video_id": "b00_sn_f_01", "frame_index": 1,
                    "timestamp_seconds": 0.1, "candidate_id": 2,
                    "bbox_x1": 42.4, "bbox_y1": 50.0, "bbox_x2": 62.4,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 52.4, "center_y_px": 60.0,
                    "center_x_normalized": 0.524, "center_y_normalized": 0.6,
                },
                {
                    "video_id": "b00_sn_f_01", "frame_index": 2,
                    "timestamp_seconds": 0.2, "candidate_id": 3,
                    "bbox_x1": 44.0, "bbox_y1": 50.0, "bbox_x2": 64.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 54.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.54, "center_y_normalized": 0.6,
                },
            ]
        )

        track = select_primary_plate_track(
            candidates, frame_count=3, initial_frame_index=0, initial_candidate_id=0,
            continuity_ambiguity_margin=0.05,
        )

        self.assertEqual(track.loc[1, "status"], "confirmation_required")
        self.assertEqual(track.loc[1, "reason"], "ambiguous_candidates")
        self.assertEqual(track.loc[2, "reason"], "tracking_stopped_after_ambiguity")

    def test_rejects_duplicate_candidate_and_track_keys_before_saving(self) -> None:
        duplicate_candidates = _plate_candidates(
            [
                {
                    "video_id": "b00_sn_f_01", "frame_index": 0,
                    "timestamp_seconds": 0.0, "candidate_id": 0,
                    "bbox_x1": 40.0, "bbox_y1": 50.0, "bbox_x2": 60.0,
                    "bbox_y2": 70.0, "bbox_confidence": 0.9,
                    "center_x_px": 50.0, "center_y_px": 60.0,
                    "center_x_normalized": 0.5, "center_y_normalized": 0.6,
                }
            ] * 2
        )
        duplicate_track = pd.DataFrame(
            [
                {
                    column: "b00_sn_f_01" if column == "video_id" else 0
                    if column == "frame_index" else np.nan
                    for column in PLATE_TRACK_COLUMNS
                },
                {
                    column: "b00_sn_f_01" if column == "video_id" else 0
                    if column == "frame_index" else np.nan
                    for column in PLATE_TRACK_COLUMNS
                },
            ],
            columns=PLATE_TRACK_COLUMNS,
        )

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                save_plate_candidate_cache(
                    duplicate_candidates, Path(temporary_directory) / "candidates.csv"
                )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                save_plate_track(
                    duplicate_track, Path(temporary_directory) / "track.csv"
                )

    def test_converts_every_detection_bbox_to_pixel_and_normalized_centers(self) -> None:
        result = _FakeResult(
            np.array([[10.0, 20.0, 30.0, 40.0], [40.0, 10.0, 80.0, 30.0]]),
            np.array([0.8, 0.7]),
        )

        candidates = plate_result_to_candidates(
            result, "b00_sn_f_01", frame_index=3, timestamp_seconds=0.1,
            image_width=100, image_height=50,
        )

        self.assertEqual(candidates["candidate_id"].tolist(), [0, 1])
        self.assertEqual(candidates["center_x_px"].tolist(), [20.0, 60.0])
        self.assertEqual(candidates["center_y_px"].tolist(), [30.0, 20.0])
        self.assertEqual(candidates["center_x_normalized"].tolist(), [0.2, 0.6])
        self.assertEqual(candidates["center_y_normalized"].tolist(), [0.6, 0.4])


class PlateNotebookSourceContractTests(unittest.TestCase):
    def test_training_uses_absolute_project_directory_to_reuse_saved_weight(self) -> None:
        notebook = json.loads(
            Path("notebooks/03_plate_detection.ipynb").read_text(encoding="utf-8")
        )
        source_by_id = {
            cell.get("id"): "".join(cell.get("source", []))
            for cell in notebook["cells"]
        }

        self.assertIn(
            "project=str(PLATE_RUN_DIRECTORY.parent.resolve())",
            source_by_id.get("u7-train", ""),
        )

    def test_reuse_preserves_recorded_training_time(self) -> None:
        notebook = json.loads(
            Path("notebooks/03_plate_detection.ipynb").read_text(encoding="utf-8")
        )
        source_by_id = {
            cell.get("id"): "".join(cell.get("source", []))
            for cell in notebook["cells"]
        }

        inference = source_by_id.get("u7-candidates", "")
        self.assertIn("recorded_training_seconds", inference)
        self.assertIn('run_summary.get("plate", {}).get("training_seconds"', inference)

    def test_phase_review_prefers_frames_with_candidates(self) -> None:
        notebook = json.loads(
            Path("notebooks/03_plate_detection.ipynb").read_text(encoding="utf-8")
        )
        source_by_id = {
            cell.get("id"): "".join(cell.get("source", []))
            for cell in notebook["cells"]
        }

        preview = source_by_id.get("u7-preview", "")
        self.assertIn("detected_phase_frames", preview)
        self.assertIn("phase_by_frame", preview)
        self.assertIn("len(detected_phase_frames) // 2", preview)

    def test_safe_load_precedes_ultralytics_import_and_uses_pose_artifact(self) -> None:
        notebook = json.loads(
            Path("notebooks/03_plate_detection.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertLess(
            source.index("verify_ultralytics_safe_load()"),
            source.index("from ultralytics import YOLO"),
        )
        self.assertIn('"pose_keypoints.csv"', source)
        self.assertNotIn("selected_pose_keypoints.csv", source)

    def test_notebook_has_post_confirmation_selected_track_review(self) -> None:
        notebook = json.loads(
            Path("notebooks/03_plate_detection.ipynb").read_text(encoding="utf-8")
        )
        source_by_id = {
            cell.get("id"): "".join(cell.get("source", []))
            for cell in notebook["cells"]
        }

        review = source_by_id.get("u7-selected-track-review", "")
        self.assertIn("plate_track", review)
        self.assertIn("BACKGROUND_PLATE_MISSELECTION_FRAMES", review)
        self.assertIn("candidate.status == 'ok'", review)
        self.assertIn("candidate.reason", review)

    def test_notebook_decodes_review_frames_once_into_bounded_cache(self) -> None:
        notebook = json.loads(
            Path("notebooks/03_plate_detection.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn("review_frame_indices", source)
        self.assertIn("len(review_frame_indices) > 6", source)
        self.assertIn("review_frames[review_frame_index].copy()", source)
        self.assertNotIn("read_video_frame(", source)


if __name__ == "__main__":
    unittest.main()
