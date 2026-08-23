"""Fusion keeps one saved measurement row for every decoded frame."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import cv2
import numpy as np
import pandas as pd

from snatch_analysis.pipeline import (
    build_frame_basis,
    fuse_frame_analysis,
    run_fusion,
    summarize_pose_keypoints,
)
from snatch_analysis.visualization import (
    save_angle_plot,
    save_plate_trajectory_plot,
    write_overlay,
)
from snatch_analysis.video import iter_video_frames, read_video_properties
from snatch_analysis.pose import COCO_KEYPOINT_NAMES


def _basis() -> pd.DataFrame:
    return build_frame_basis("b00", frame_count=3, fps=60.0)


def _phase() -> pd.DataFrame:
    return pd.DataFrame(
        {"video_id": ["b00"] * 3, "frame_index": [0, 1, 2], "phase": ["Set-up", "First Pull", "Second Pull"]}
    )


def _angles() -> pd.DataFrame:
    return pd.DataFrame(
        {"video_id": ["b00", "b00"], "frame_index": [0, 2], "knee_angle_deg": [90.0, 110.0], "knee_status": ["ok", "ok"]}
    )


def _plate() -> pd.DataFrame:
    return pd.DataFrame(
        {"video_id": ["b00", "b00"], "frame_index": [0, 2], "center_x_px": [100.0, 120.0], "center_y_px": [200.0, 180.0], "status": ["ok", "ok"], "reason": ["", ""]}
    )


def _pose() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": ["b00"] * len(COCO_KEYPOINT_NAMES),
            "frame_index": [0] * len(COCO_KEYPOINT_NAMES),
            "joint_name": COCO_KEYPOINT_NAMES,
            "x": list(range(len(COCO_KEYPOINT_NAMES))),
            "y": list(range(len(COCO_KEYPOINT_NAMES))),
            "confidence": [0.8] * len(COCO_KEYPOINT_NAMES),
            "status": ["ok"] * len(COCO_KEYPOINT_NAMES),
            "reason": [""] * len(COCO_KEYPOINT_NAMES),
        }
    )


def _write_overlay_inputs(root: Path) -> tuple[Path, Path, Path]:
    video_path = root / "input.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (80, 80))
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    try:
        for _ in range(5):
            writer.write(np.zeros((80, 80, 3), dtype=np.uint8))
    finally:
        writer.release()
    analysis_path = root / "frame_analysis.csv"
    pd.DataFrame(
        {
            "video_id": ["b00"] * 5, "frame_index": [0, 1, 2, 3, 4],
            "phase": ["First Pull"] * 5, "pose_status": ["ok"] * 5,
            "plate_status": ["ok", "ok", "missing", "ok", "ok"],
            "center_x_px": [10.0, 30.0, None, 50.0, 70.0],
            "center_y_px": [10.0, 10.0, None, 10.0, 10.0],
        }
    ).to_csv(analysis_path, index=False)
    pose_path = root / "pose.csv"
    pd.DataFrame(
        {
            "video_id": ["b00"] * 5, "frame_index": [0, 1, 2, 3, 4],
            "joint_name": ["nose"] * 5, "joint_index": [0] * 5,
            "x": [5.0] * 5, "y": [5.0] * 5, "status": ["ok"] * 5,
        }
    ).to_csv(pose_path, index=False)
    return video_path, analysis_path, pose_path


class PipelineTests(unittest.TestCase):
    def test_fusion_left_joins_every_frame_basis_row(self) -> None:
        result = fuse_frame_analysis(_basis(), _phase(), _pose(), _angles(), _plate())

        self.assertEqual(result[["video_id", "frame_index"]].to_dict("records"), [{"video_id": "b00", "frame_index": 0}, {"video_id": "b00", "frame_index": 1}, {"video_id": "b00", "frame_index": 2}])
        self.assertEqual(result.loc[1, "phase"], "First Pull")
        self.assertEqual(result.loc[0, "knee_angle_deg"], 90.0)
        self.assertEqual(result.loc[2, "center_x_px"], 120.0)

    def test_fusion_rejects_duplicate_direct_frame_keys(self) -> None:
        for table_name in ("phase", "angles", "plate"):
            with self.subTest(table_name=table_name):
                tables = {"phase": _phase(), "angles": _angles(), "plate": _plate()}
                tables[table_name] = pd.concat([tables[table_name], tables[table_name].iloc[[0]]], ignore_index=True)
                with self.assertRaisesRegex(ValueError, f"duplicate.*{table_name}"):
                    fuse_frame_analysis(_basis(), tables["phase"], _pose(), tables["angles"], tables["plate"])

    def test_pose_summary_marks_some_available_keypoints_partial(self) -> None:
        pose = _pose().copy()
        pose.loc[1, ["status", "reason"]] = ["not_available", "low_keypoint_confidence"]

        summary = summarize_pose_keypoints(pose).iloc[0]

        self.assertEqual(summary["pose_status"], "partial")
        self.assertEqual(summary["pose_reason"], "low_keypoint_confidence")
        self.assertEqual(summary["pose_available_keypoint_count"], 16)
        self.assertEqual(summary["pose_total_keypoint_count"], 17)

    def test_pose_summary_rejects_incomplete_coco_joint_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_index=0.*missing"):
            summarize_pose_keypoints(_pose().iloc[:1])

    def test_missing_pose_or_angles_keeps_phase_and_plate_measurements(self) -> None:
        result = fuse_frame_analysis(_basis(), _phase(), _pose().iloc[0:0], _angles().iloc[0:0], _plate())

        self.assertEqual(result.loc[1, "phase"], "First Pull")
        self.assertEqual(result.loc[2, "center_x_px"], 120.0)
        self.assertTrue(pd.isna(result.loc[0, "knee_angle_deg"]))
        self.assertEqual(result.loc[0, "pose_status"], "not_available")

    def test_fusion_rejects_phase_table_missing_a_basis_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "phase.*coverage"):
            fuse_frame_analysis(_basis(), _phase().drop(index=1), _pose(), _angles(), _plate())

    def test_fusion_rejects_gapped_frame_basis(self) -> None:
        gapped_basis = pd.DataFrame(
            {"video_id": ["b00", "b00"], "frame_index": [0, 2], "timestamp_seconds": [0.0, 2 / 60]}
        )
        phase = _phase().drop(index=1)

        with self.assertRaisesRegex(ValueError, "zero-based contiguous"):
            fuse_frame_analysis(gapped_basis, phase, _pose(), _angles(), _plate())

    def test_missing_plate_keeps_phase_and_pose_angle_measurements(self) -> None:
        result = fuse_frame_analysis(_basis(), _phase(), _pose(), _angles(), _plate().iloc[0:0])

        self.assertEqual(result.loc[0, "phase"], "Set-up")
        self.assertEqual(result.loc[0, "pose_status"], "ok")
        self.assertEqual(result.loc[0, "knee_angle_deg"], 90.0)
        self.assertTrue(pd.isna(result.loc[0, "center_x_px"]))

    def test_saved_frame_analysis_regenerates_gap_preserving_plots(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frame_analysis_path = root / "frame_analysis.csv"
            pd.DataFrame(
                {
                    "video_id": ["b00"] * 3, "frame_index": [0, 1, 2], "phase": ["Set-up", "First Pull", "Second Pull"],
                    "knee_angle_deg": [90.0, 95.0, 100.0], "hip_angle_deg": [50.0, 55.0, 60.0],
                    "elbow_angle_deg": [170.0, 165.0, 160.0], "trunk_angle_deg": [40.0, 45.0, 50.0],
                    "center_x_px": [100.0, None, 120.0], "center_y_px": [200.0, None, 180.0],
                }
            ).to_csv(frame_analysis_path, index=False)
            angle_path, plate_path = root / "angles.png", root / "plate.png"
            save_angle_plot(frame_analysis_path, angle_path)
            segments = save_plate_trajectory_plot(frame_analysis_path, plate_path)

            self.assertGreater(angle_path.stat().st_size, 0)
            self.assertGreater(plate_path.stat().st_size, 0)
            self.assertEqual([segment.tolist() for segment in segments], [[[100.0, 200.0]], [[120.0, 180.0]]])

    def test_overlay_failure_keeps_measurement_csv_and_reports_visualization(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "frame_analysis.csv"
            result = run_fusion(_basis(), _phase(), _pose(), _angles(), _plate(), output_path=output_path, visualization_writer=lambda _: (_ for _ in ()).throw(RuntimeError("overlay unavailable")))

            self.assertTrue(output_path.exists())
            self.assertEqual(len(pd.read_csv(output_path)), 3)
            self.assertEqual(result["measurement_status"], "ok")
            self.assertEqual(result["visualization_status"], "failed")
            self.assertIn("overlay unavailable", result["visualization_reason"])

    def test_fusion_rejects_frame_table_from_another_video(self) -> None:
        with self.assertRaisesRegex(ValueError, "video_id"):
            fuse_frame_analysis(_basis(), _phase(), _pose(), _angles().assign(video_id="other"), _plate())

    def test_overlay_rejects_duplicate_or_mismatched_cached_inputs_before_output(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path, analysis_path, pose_path = _write_overlay_inputs(root)
            saved_analysis = pd.read_csv(analysis_path)
            duplicate_analysis = pd.concat([saved_analysis, saved_analysis.iloc[[0]]], ignore_index=True)
            duplicate_path = root / "duplicate.csv"
            duplicate_analysis.to_csv(duplicate_path, index=False)
            output_path = root / "overlay.mp4"
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_overlay(video_path, duplicate_path, pose_path, output_path)
            self.assertFalse(output_path.exists())

            mismatched_pose = pd.read_csv(pose_path).assign(video_id="other")
            mismatched_pose.to_csv(pose_path, index=False)
            with self.assertRaisesRegex(ValueError, "video_id"):
                write_overlay(video_path, analysis_path, pose_path, output_path)
            self.assertFalse(output_path.exists())

    def test_overlay_keeps_full_contiguous_trajectory_visible(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path, analysis_path, pose_path = _write_overlay_inputs(root)
            output_path = root / "overlay.mp4"
            write_overlay(video_path, analysis_path, pose_path, output_path)
            frames = [frame for _, _, frame in iter_video_frames(output_path, expected_frame_count=5)]

            self.assertEqual(len(frames), 5)
            self.assertGreater(int(frames[4][10, 20][0]), int(frames[4][10, 20][2]) + 40)
            self.assertGreater(int(frames[4][10, 60][0]), int(frames[4][10, 60][2]) + 40)
            self.assertLess(int(frames[4][10, 40][0]), 40)
            properties = read_video_properties(output_path)
            self.assertEqual((properties["width"], properties["height"], properties["frame_count"]), (80, 80, 5))

    def test_save_json_atomically_replaces_valid_json(self) -> None:
        from snatch_analysis.pipeline import save_json_atomically

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "run_summary.json"
            output_path.write_text('{"previous": true}\n', encoding="utf-8")

            save_json_atomically({"fusion": {"status": "ok"}}, output_path)

            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), {"fusion": {"status": "ok"}})

    def test_save_json_atomically_cleans_temp_after_serialization_failure(self) -> None:
        from snatch_analysis.pipeline import save_json_atomically

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = root / "run_summary.json"
            original = '{"previous": true}\n'
            output_path.write_text(original, encoding="utf-8")

            with self.assertRaises(TypeError):
                save_json_atomically({"not_serializable": object()}, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.iterdir()), [output_path])


if __name__ == "__main__":
    unittest.main()
