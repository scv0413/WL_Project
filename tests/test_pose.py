import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import cv2
import numpy as np
import pandas as pd

import snatch_analysis.pose as pose_module
from snatch_analysis.pose import (
    COCO_KEYPOINT_NAMES,
    POSE_CANDIDATE_COLUMNS,
    calculate_joint_angles,
    pose_result_to_candidates,
    resolve_initial_primary_lifter,
    save_pose_candidate_cache,
    save_selected_pose_keypoints,
    select_initial_primary_lifter,
    select_primary_lifter_track,
    summarize_body_side_confidence,
    validate_pose_candidate_cache,
    verify_ultralytics_safe_load,
)


JOINT_COUNT = len(COCO_KEYPOINT_NAMES)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_pose_baseline.ipynb"


def pose_result(
    boxes: list[list[float]],
    box_confidences: list[float],
    keypoints: list[list[list[float]]],
) -> SimpleNamespace:
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray(boxes, dtype=float),
            conf=np.asarray(box_confidences, dtype=float),
        ),
        keypoints=SimpleNamespace(data=np.asarray(keypoints, dtype=float)),
    )


def person_keypoints(confidence: float = 0.9) -> list[list[float]]:
    points = [[float(index), float(index + 1), confidence] for index in range(JOINT_COUNT)]
    coordinates = {
        "left_shoulder": [0.0, 0.0],
        "left_elbow": [1.0, 0.0],
        "left_wrist": [2.0, 0.0],
        "left_hip": [0.0, 1.0],
        "left_knee": [0.0, 2.0],
        "left_ankle": [0.0, 3.0],
        "right_shoulder": [10.0, 0.0],
        "right_elbow": [10.0, 1.0],
        "right_wrist": [11.0, 1.0],
        "right_hip": [10.0, 2.0],
        "right_knee": [11.0, 2.0],
        "right_ankle": [11.0, 3.0],
    }
    for joint_name, coordinate in coordinates.items():
        points[COCO_KEYPOINT_NAMES.index(joint_name)][:2] = coordinate
    return points


def candidates_for_frame(
    frame_index: int,
    boxes: list[list[float]],
    confidences: list[float] | None = None,
):
    confidences = confidences or [0.9] * len(boxes)
    keypoints = [person_keypoints() for _ in boxes]
    return pose_result_to_candidates(
        result=pose_result(boxes, confidences, keypoints),
        video_id="b00_sn_f_01",
        frame_index=frame_index,
        timestamp_seconds=frame_index / 60.0,
    )


class PoseTests(unittest.TestCase):
    def test_infers_pose_candidates_from_one_streaming_prediction(self) -> None:
        runner = getattr(pose_module, "infer_pose_video_candidates", None)
        self.assertTrue(callable(runner), "infer_pose_video_candidates is missing")

        class FakePoseModel:
            def __init__(self) -> None:
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                return iter(
                    [
                        pose_result(
                            [[10.0, 10.0, 30.0, 40.0]],
                            [0.9],
                            [person_keypoints()],
                        ),
                        pose_result(
                            [[11.0, 10.0, 31.0, 40.0]],
                            [0.8],
                            [person_keypoints()],
                        ),
                    ]
                )

        with TemporaryDirectory() as directory:
            video_path = Path(directory) / "two_frames.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            try:
                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
                writer.write(np.ones((24, 32, 3), dtype=np.uint8))
            finally:
                writer.release()

            model = FakePoseModel()
            candidates, processed_frame_count = runner(
                model,
                video_path=video_path,
                video_id="a01_sn_f_01",
                frame_count=2,
                fps=30.0,
                image_size=640,
                confidence=0.25,
                device="cpu",
            )

        self.assertEqual(processed_frame_count, 2)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["source"], str(video_path))
        self.assertTrue(model.calls[0]["stream"])
        self.assertFalse(model.calls[0]["verbose"])
        self.assertEqual(candidates["frame_index"].unique().tolist(), [0, 1])
        self.assertEqual(set(candidates["video_id"]), {"a01_sn_f_01"})

    def test_rejects_pose_stream_frame_count_mismatch(self) -> None:
        class ShortPoseModel:
            def predict(self, **kwargs):
                return iter([
                    pose_result(
                        [[10.0, 10.0, 30.0, 40.0]],
                        [0.9],
                        [person_keypoints()],
                    )
                ])

        with self.assertRaisesRegex(
            RuntimeError, "expected=2, actual=1"
        ):
            pose_module.infer_pose_video_candidates(
                ShortPoseModel(),
                video_path=Path("short.mp4"),
                video_id="a01_sn_f_01",
                frame_count=2,
                fps=30.0,
                image_size=640,
                confidence=0.25,
                device="cpu",
            )

        with self.assertRaisesRegex(ValueError, "fps"):
            pose_module.infer_pose_video_candidates(
                ShortPoseModel(),
                video_path=Path("short.mp4"),
                video_id="a01_sn_f_01",
                frame_count=1,
                fps=0.0,
                image_size=640,
                confidence=0.25,
                device="cpu",
            )

    def test_pose_notebook_is_source_only_and_uses_automatic_selection(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        source_by_id = {
            cell["id"]: "".join(cell["source"])
            for cell in code_cells
        }

        self.assertIn("REINFER_POSE = False", source_by_id["pose-settings"])
        self.assertIn(
            "MANUAL_OVERRIDE_CANDIDATE_ID = None",
            source_by_id["primary-settings"],
        )
        self.assertIn("visible_body_side", source_by_id["body-side-setting"])
        self.assertIn(
            "BODY_SIDE = labeled_body_side",
            source_by_id["body-side-setting"],
        )
        self.assertIn("phase_events", source_by_id["primary-track-review"])
        self.assertIn("motion_score", source_by_id["primary-track-review"])
        self.assertIn(
            "POSE_KEYPOINT_PATH.unlink(missing_ok=True)",
            source_by_id["primary-track"],
        )
        self.assertIn(
            "resolve_initial_primary_lifter(",
            source_by_id["primary-track"],
        )
        self.assertIn('"status": "failed"', source_by_id["primary-track"])
        self.assertLess(
            source_by_id["primary-track"].index("save_run_summary(run_summary)"),
            source_by_id["primary-track"].index(
                "JOINT_ANGLE_PATH.unlink(missing_ok=True)"
            ),
        )
        self.assertIn(
            "if pose_keypoints_preview.empty:",
            source_by_id["joint-angle-save"],
        )
        self.assertLess(
            source_by_id["joint-angle-save"].index("if pose_keypoints_preview.empty:"),
            source_by_id["joint-angle-save"].index("calculate_joint_angles"),
        )
        self.assertTrue(
            all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
        )

    def test_body_side_cell_reloads_current_manifest_label(self) -> None:
        from tempfile import TemporaryDirectory

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        body_side_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell.get("id") == "body-side-setting"
        )

        with TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "videos.csv"
            pd.DataFrame(
                {
                    "video_id": ["b00_sn_f_01"],
                    "visible_body_side": ["right"],
                }
            ).to_csv(manifest_path, index=False)
            namespace = {
                "pd": pd,
                "MANIFEST_PATH": manifest_path,
                "VIDEO_ID": "b00_sn_f_01",
                "selected_video": pd.Series({"visible_body_side": "left"}),
            }

            exec(compile(body_side_source, "body-side-setting", "exec"), namespace)

            self.assertEqual(namespace["BODY_SIDE"], "right")

    def test_unlabelled_body_side_removes_stale_angle_output(self) -> None:
        from tempfile import TemporaryDirectory

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        angle_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell.get("id") == "joint-angle-save"
        )
        run_summary = {"pose_selection": {"body_side": "left"}}

        def load_run_summary():
            return dict(run_summary)

        def save_run_summary(summary):
            run_summary.clear()
            run_summary.update(summary)

        with TemporaryDirectory() as temporary_directory:
            angle_path = Path(temporary_directory) / "joint_angles.csv"
            pose_path = Path(temporary_directory) / "pose_keypoints.csv"
            angle_path.write_text("stale angles", encoding="utf-8")
            namespace = {
                "pd": pd,
                "pose_keypoints_preview": pd.DataFrame({"frame_index": [0]}),
                "BODY_SIDE": None,
                "POSE_KEYPOINT_PATH": pose_path,
                "JOINT_ANGLE_PATH": angle_path,
                "load_run_summary": load_run_summary,
                "save_run_summary": save_run_summary,
                "calculate_joint_angles": lambda *_: self.fail(
                    "Unlabelled side must not calculate angles"
                ),
            }

            exec(compile(angle_source, "joint-angle-save", "exec"), namespace)

            self.assertFalse(angle_path.exists())
            self.assertNotIn("body_side", run_summary["pose_selection"])
            self.assertEqual(
                run_summary["pose_selection"]["angle_status"],
                "not_run",
            )

    def test_converts_every_pose_candidate_and_joint_to_long_rows(self) -> None:
        candidates = candidates_for_frame(
            frame_index=4,
            boxes=[[10.0, 20.0, 110.0, 220.0], [300.0, 30.0, 350.0, 130.0]],
            confidences=[0.95, 0.7],
        )

        self.assertEqual(len(candidates), 2 * JOINT_COUNT)
        self.assertEqual(candidates["candidate_id"].nunique(), 2)
        self.assertEqual(candidates["joint_name"].nunique(), JOINT_COUNT)
        first = candidates.iloc[0]
        self.assertEqual(first["frame_index"], 4)
        self.assertEqual(first["joint_name"], "nose")
        self.assertEqual(first["bbox_area"], 20_000.0)
        self.assertAlmostEqual(first["bbox_confidence"], 0.95)

    def test_auto_selects_candidate_that_is_largest_and_horizontally_centered(self) -> None:
        candidates = candidates_for_frame(
            frame_index=0,
            boxes=[[40.0, 10.0, 160.0, 190.0], [0.0, 0.0, 30.0, 60.0]],
        )

        proposal = select_initial_primary_lifter(candidates, frame_width=200)

        self.assertEqual(proposal["candidate_id"], 0)
        self.assertEqual(proposal["status"], "auto_selected")
        self.assertEqual(
            proposal["reason"], "largest_bbox_and_horizontal_center"
        )

    def test_requires_override_when_size_and_horizontal_center_disagree(self) -> None:
        candidates = candidates_for_frame(
            frame_index=0,
            boxes=[[0.0, 20.0, 80.0, 180.0], [80.0, 60.0, 120.0, 140.0]],
        )

        proposal = select_initial_primary_lifter(candidates, frame_width=200)

        self.assertIsNone(proposal["candidate_id"])
        self.assertEqual(proposal["status"], "confirmation_required")
        self.assertEqual(proposal["reason"], "selection_criteria_disagree")

    def test_resolves_automatic_conflict_and_manual_override(self) -> None:
        cases = (
            ({"candidate_id": 0}, None, 0, "automatic"),
            ({"candidate_id": None}, None, None, None),
            ({"candidate_id": 0}, 1, 1, "manual_override"),
        )

        for selection, override, expected_id, expected_method in cases:
            with self.subTest(selection=selection, override=override):
                resolved = resolve_initial_primary_lifter(selection, override)
                self.assertEqual(resolved["candidate_id"], expected_id)
                self.assertEqual(resolved["selection_method"], expected_method)

    def test_notebook_marks_invalid_manual_override_as_failed(self) -> None:
        from tempfile import TemporaryDirectory

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        track_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell.get("id") == "primary-track"
        )
        candidates = candidates_for_frame(
            frame_index=0,
            boxes=[[10.0, 10.0, 100.0, 190.0]],
        )
        run_summary = {"pose_selection": {"status": "complete"}}

        def load_run_summary():
            return dict(run_summary)

        def save_run_summary(summary):
            run_summary.clear()
            run_summary.update(summary)

        with TemporaryDirectory() as directory:
            pose_path = Path(directory) / "pose_keypoints.csv"
            angle_path = Path(directory) / "joint_angles.csv"
            pose_path.write_text("previous pose", encoding="utf-8")
            angle_path.write_text("previous angles", encoding="utf-8")
            namespace = {
                "pd": pd,
                "pose_candidates": candidates,
                "primary_selection": {
                    "frame_index": 0,
                    "candidate_id": 0,
                    "status": "auto_selected",
                    "reason": "largest_bbox_and_horizontal_center",
                },
                "MANUAL_OVERRIDE_CANDIDATE_ID": 99,
                "KEYPOINT_CONFIDENCE_THRESHOLD": 0.5,
                "frame_count": 1,
                "video_properties": {"fps": 60.0},
                "POSE_KEYPOINT_PATH": pose_path,
                "JOINT_ANGLE_PATH": angle_path,
                "resolve_initial_primary_lifter": resolve_initial_primary_lifter,
                "select_primary_lifter_track": select_primary_lifter_track,
                "save_selected_pose_keypoints": save_selected_pose_keypoints,
                "load_run_summary": load_run_summary,
                "save_run_summary": save_run_summary,
            }

            with self.assertRaises(ValueError):
                exec(compile(track_source, "primary-track", "exec"), namespace)

            selection = run_summary["pose_selection"]
            self.assertEqual(selection["status"], "failed")
            self.assertTrue(selection["failure_reason"].startswith("ValueError:"))
            self.assertTrue(pose_path.exists())
            self.assertFalse(angle_path.exists())

    def test_uses_first_valid_detection_frame_for_auto_selection(self) -> None:
        candidates = candidates_for_frame(
            frame_index=3,
            boxes=[[40.0, 10.0, 160.0, 190.0]],
        )

        proposal = select_initial_primary_lifter(candidates, frame_width=200)

        self.assertEqual(proposal["frame_index"], 3)
        self.assertEqual(proposal["candidate_id"], 0)

    def test_tracking_uses_center_and_area_continuity_not_candidate_id(self) -> None:
        frame_zero = candidates_for_frame(
            frame_index=0,
            boxes=[[40.0, 10.0, 160.0, 190.0], [250.0, 20.0, 290.0, 100.0]],
        )
        frame_one = candidates_for_frame(
            frame_index=1,
            boxes=[[240.0, 20.0, 295.0, 105.0], [45.0, 12.0, 165.0, 192.0]],
        )
        all_candidates = pd.concat([frame_zero, frame_one], ignore_index=True)
        selected = select_primary_lifter_track(
            all_candidates,
            frame_count=2,
            initial_frame_index=0,
            initial_candidate_id=0,
            keypoint_confidence_threshold=0.5,
        )

        frame_one_ids = selected.loc[
            (selected["frame_index"] == 1) & (selected["joint_name"] == "nose"),
            "candidate_id",
        ]
        self.assertEqual(frame_one_ids.tolist(), [1])
        self.assertEqual(len(selected), 2 * JOINT_COUNT)

    def test_tracking_loss_reasons_keep_every_joint_not_available(self) -> None:
        frame_zero = candidates_for_frame(
            frame_index=0,
            boxes=[[40.0, 10.0, 160.0, 190.0]],
        )
        cases = {
            "no_detection": frame_zero,
            "continuity_mismatch": pd.concat(
                [
                    frame_zero,
                    candidates_for_frame(1, [[900.0, 10.0, 1020.0, 190.0]]),
                ],
                ignore_index=True,
            ),
            "ambiguous_candidates": pd.concat(
                [
                    frame_zero,
                    candidates_for_frame(
                        1,
                        [
                            [42.0, 10.0, 162.0, 190.0],
                            [42.0, 10.0, 162.0, 190.0],
                        ],
                    ),
                ],
                ignore_index=True,
            ),
        }

        for expected_reason, candidates in cases.items():
            with self.subTest(expected_reason=expected_reason):
                selected = select_primary_lifter_track(
                    candidates,
                    frame_count=2,
                    initial_frame_index=0,
                    initial_candidate_id=0,
                    keypoint_confidence_threshold=0.5,
                )
                lost_frame = selected.loc[selected["frame_index"] == 1]
                self.assertEqual(len(lost_frame), JOINT_COUNT)
                self.assertEqual(
                    lost_frame["reason"].unique().tolist(), [expected_reason]
                )
                self.assertTrue(lost_frame["x"].isna().all())

    def test_ambiguous_initial_candidates_are_not_auto_selected(self) -> None:
        candidates = candidates_for_frame(
            frame_index=0,
            boxes=[[20.0, 20.0, 80.0, 180.0], [120.0, 20.0, 180.0, 180.0]],
        )

        proposal = select_initial_primary_lifter(candidates, frame_width=200)

        self.assertIsNone(proposal["candidate_id"])
        self.assertEqual(proposal["status"], "confirmation_required")
        self.assertEqual(proposal["reason"], "ambiguous_candidates")

    def test_low_confidence_and_missing_joints_remain_not_available(self) -> None:
        points = person_keypoints()
        points[COCO_KEYPOINT_NAMES.index("left_knee")][2] = 0.2
        candidates = pose_result_to_candidates(
            result=pose_result([[10.0, 10.0, 100.0, 190.0]], [0.95], [points]),
            video_id="b00_sn_f_01",
            frame_index=0,
            timestamp_seconds=0.0,
        )
        candidates = candidates.loc[candidates["joint_name"] != "left_ankle"]

        selected = select_primary_lifter_track(
            candidates,
            frame_count=1,
            initial_frame_index=0,
            initial_candidate_id=0,
            keypoint_confidence_threshold=0.5,
        )
        knee = selected.loc[selected["joint_name"] == "left_knee"].iloc[0]
        ankle = selected.loc[selected["joint_name"] == "left_ankle"].iloc[0]

        self.assertTrue(np.isnan(knee["x"]))
        self.assertEqual(knee["reason"], "low_keypoint_confidence")
        self.assertTrue(np.isnan(ankle["x"]))
        self.assertEqual(ankle["reason"], "missing_keypoint")

        side_summary = summarize_body_side_confidence(selected)
        self.assertEqual(side_summary["side"].tolist(), ["left", "right"])

        angles = calculate_joint_angles(selected, body_side="left")
        self.assertTrue(np.isnan(angles.loc[0, "knee_angle_deg"]))
        self.assertEqual(angles.loc[0, "knee_status"], "not_available")
        self.assertEqual(angles.loc[0, "knee_reason"], "low_keypoint_confidence")
        self.assertEqual(angles.loc[0, "elbow_status"], "ok")

    def test_calculates_known_angles_for_each_fixed_body_side(self) -> None:
        candidates = candidates_for_frame(
            frame_index=0,
            boxes=[[10.0, 10.0, 100.0, 190.0]],
        )
        selected = select_primary_lifter_track(
            candidates,
            frame_count=1,
            initial_frame_index=0,
            initial_candidate_id=0,
            keypoint_confidence_threshold=0.5,
        )

        left = calculate_joint_angles(selected, body_side="left").iloc[0]
        right = calculate_joint_angles(selected, body_side="right").iloc[0]

        self.assertEqual(left["body_side"], "left")
        for measurement in ("knee", "hip", "elbow"):
            self.assertAlmostEqual(left[f"{measurement}_angle_deg"], 180.0)
            self.assertEqual(left[f"{measurement}_status"], "ok")
        self.assertAlmostEqual(left["trunk_angle_deg"], 0.0)

        self.assertEqual(right["body_side"], "right")
        for measurement in ("knee", "hip", "elbow"):
            self.assertAlmostEqual(right[f"{measurement}_angle_deg"], 90.0)
            self.assertEqual(right[f"{measurement}_status"], "ok")
        self.assertAlmostEqual(right["trunk_angle_deg"], 0.0)

    def test_safe_load_check_passes_or_fails_closed(self) -> None:
        with (
            patch("ultralytics.utils.SAFE_LOAD", True),
            patch("ultralytics.nn.tasks._SafeLoad.SUPPORTED", True),
            patch("ultralytics.nn.tasks._SafeLoad.restricted", return_value=True),
        ):
            result = verify_ultralytics_safe_load()
        self.assertTrue(result["active"])

        with patch("ultralytics.utils.SAFE_LOAD", False):
            with self.assertRaises(RuntimeError):
                verify_ultralytics_safe_load()

    def test_pose_candidate_cache_requires_matching_complete_run(self) -> None:
        candidates = candidates_for_frame(
            frame_index=0,
            boxes=[[10.0, 10.0, 100.0, 190.0]],
        )
        expected_run = {
            "status": "complete",
            "video_id": "b00_sn_f_01",
            "frame_count": 2,
            "processed_frame_count": 2,
            "model": "yolov8n-pose.pt",
            "image_size": 640,
            "bbox_confidence_threshold": 0.25,
        }

        validate_pose_candidate_cache(
            candidates,
            expected_video_id="b00_sn_f_01",
            expected_frame_count=2,
            run_metadata=expected_run,
            expected_run_metadata=expected_run,
        )
        for changed in (
            {**expected_run, "processed_frame_count": 1},
            {**expected_run, "image_size": 320},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    validate_pose_candidate_cache(
                        candidates,
                        expected_video_id="b00_sn_f_01",
                        expected_frame_count=2,
                        run_metadata=changed,
                        expected_run_metadata=expected_run,
                    )

    def test_pose_candidate_cache_is_written_atomically(self) -> None:
        from tempfile import TemporaryDirectory

        candidates = candidates_for_frame(
            frame_index=0,
            boxes=[[10.0, 10.0, 100.0, 190.0]],
        )
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "pose_candidates.csv"
            save_pose_candidate_cache(candidates, output_path)
            saved = pd.read_csv(output_path)

        self.assertEqual(len(saved), len(candidates))
        self.assertEqual(tuple(saved.columns), POSE_CANDIDATE_COLUMNS)

    def test_selected_pose_save_preserves_previous_file_on_write_failure(self) -> None:
        from tempfile import TemporaryDirectory

        candidates = candidates_for_frame(
            frame_index=0,
            boxes=[[10.0, 10.0, 100.0, 190.0]],
        )
        selected = select_primary_lifter_track(
            candidates,
            frame_count=1,
            initial_frame_index=0,
            initial_candidate_id=0,
            keypoint_confidence_threshold=0.5,
        )
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "pose_keypoints.csv"
            save_selected_pose_keypoints(selected, output_path)
            before = output_path.read_bytes()

            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_selected_pose_keypoints(selected, output_path)

            self.assertEqual(output_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
