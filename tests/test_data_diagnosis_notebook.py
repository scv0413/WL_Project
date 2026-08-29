from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import numpy as np
import pandas as pd

from snatch_analysis.plate import (
    build_pilot_ground_truth_template,
    save_pilot_ground_truth,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_data_diagnosis.ipynb"


class DataDiagnosisNotebookTests(unittest.TestCase):
    def test_default_run_preserves_existing_manual_metadata(self) -> None:
        self.assertTrue(NOTEBOOK_PATH.exists(), "data diagnosis notebook is missing")

        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "pyproject.toml").touch()
            video_directory = run_root / "data" / "raw" / "videos"
            video_directory.mkdir(parents=True)
            self._write_test_video(video_directory / "b00_sn_f_01.mp4")

            self._execute_notebook(run_root)
            manifest_path = run_root / "data" / "metadata" / "videos.csv"
            manifest = pd.read_csv(manifest_path)
            manifest.loc[0, "has_plate"] = "yes"
            manifest.loc[0, "camera_motion"] = "fixed"
            manifest.loc[0, "visible_body_side"] = "left"
            manifest.to_csv(manifest_path, index=False)

            self._execute_notebook(run_root)
            reloaded_manifest = pd.read_csv(manifest_path)

            self.assertEqual(reloaded_manifest.loc[0, "has_plate"], "yes")
            self.assertEqual(reloaded_manifest.loc[0, "camera_motion"], "fixed")
            self.assertEqual(reloaded_manifest.loc[0, "visible_body_side"], "left")

    def test_applies_visible_body_side_labels_from_labeling_cell(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "pyproject.toml").touch()
            video_directory = run_root / "data" / "raw" / "videos"
            video_directory.mkdir(parents=True)
            self._write_test_video(video_directory / "b00_sn_f_01.mp4")
            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
            label_cell = next(
                cell
                for cell in notebook.cells
                if cell.get("id") == "visible-body-side-labels"
            )
            label_cell.source = (
                'VISIBLE_BODY_SIDE_LABELS = {"b00_sn_f_01": "right"}'
            )

            self._execute_notebook(run_root, notebook)
            manifest = pd.read_csv(run_root / "data" / "metadata" / "videos.csv")

            self.assertEqual(manifest.loc[0, "visible_body_side"], "right")

    def test_u2_summary_exposes_the_video_frame_contract(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "pyproject.toml").touch()
            video_directory = run_root / "data" / "raw" / "videos"
            video_directory.mkdir(parents=True)
            self._write_test_video(video_directory / "b00_sn_f_01.mp4")

            executed_notebook = self._execute_notebook(run_root)
            summary_cells = [
                cell
                for cell in executed_notebook.cells
                if cell.get("id") == "video-contract-summary"
            ]

            self.assertEqual(len(summary_cells), 1)
            summary_text = summary_cells[0]["outputs"][-1]["data"]["text/plain"]
            for expected_text in (
                "b00_sn_f_01",
                "25.0",
                "0 ~ 4",
                "(48, 64, 3)",
                "generator streaming",
                "PASS",
            ):
                self.assertIn(expected_text, summary_text)

            check_cells = [
                cell
                for cell in executed_notebook.cells
                if cell.get("id") == "video-contract-checks"
            ]
            self.assertEqual(len(check_cells), 1)
            checks_text = check_cells[0]["outputs"][-1]["data"]["text/plain"]
            self.assertIn("fps_matches_manifest", checks_text)

    def test_u2_contract_mismatch_fails_loudly(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "pyproject.toml").touch()
            video_directory = run_root / "data" / "raw" / "videos"
            video_directory.mkdir(parents=True)
            self._write_test_video(video_directory / "b00_sn_f_01.mp4")

            self._execute_notebook(run_root)
            manifest_path = run_root / "data" / "metadata" / "videos.csv"
            manifest = pd.read_csv(manifest_path)
            manifest.loc[0, "frame_count"] = 4
            manifest.to_csv(manifest_path, index=False)

            with self.assertRaisesRegex(
                CellExecutionError,
                "count_matches_manifest",
            ):
                self._execute_notebook(run_root)

    def test_preview_rejects_a_video_id_changed_after_contract_check(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "pyproject.toml").touch()
            video_directory = run_root / "data" / "raw" / "videos"
            video_directory.mkdir(parents=True)
            self._write_test_video(video_directory / "b00_sn_f_01.mp4")
            self._write_test_video(video_directory / "b01_sn_f_01.mp4")

            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
            preview_index = next(
                index
                for index, cell in enumerate(notebook.cells)
                if cell.get("id") == "video-preview-frame"
            )
            notebook.cells.insert(
                preview_index,
                nbformat.v4.new_code_cell('VIDEO_ID = "b01_sn_f_01"'),
            )

            with self.assertRaisesRegex(
                CellExecutionError,
                "5~6단계 셀을 다시 실행",
            ):
                self._execute_notebook(run_root, notebook)

    def test_u9_saves_person_split_and_blind_phase_samples_before_bbox(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "pyproject.toml").touch()
            video_directory = run_root / "data" / "raw" / "videos"
            video_directory.mkdir(parents=True)
            video_ids = (
                "b00_sn_f_01",
                "a01_sn_f_01",
                "b01_sn_f_01",
                "c01_sn_f_01",
                "a02_sn_f_01",
                "a03_sn_f_01",
            )
            for video_id in video_ids:
                self._write_test_video(video_directory / f"{video_id}.mp4", frames=10)

            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
            settings = next(
                cell for cell in notebook.cells if cell.get("id") == "u9-settings"
            )
            roles = {
                "a01": "model_selection_pilot",
                "b01": "model_selection_pilot",
                "c01": "model_selection_pilot",
                "a02": "rule_development",
                "a03": "rule_blind",
            }
            selected = [
                "a01_sn_f_01",
                "b01_sn_f_01",
                "c01_sn_f_01",
                "a02_sn_f_01",
                "a03_sn_f_01",
            ]
            labels = {
                video_id: {"has_plate": "yes", "visible_body_side": "left"}
                for video_id in selected
            }
            events = {
                video_id: {
                    "bar_off_floor": 2,
                    "knee_pass": 4,
                    "max_extension": 6,
                    "catch_stable": 7,
                    "recovery_stable": 9,
                }
                for video_id in selected[:3]
            }
            settings.source = (
                "RUN_U9_SETUP = True\n"
                f"U9_EVALUATION_PERSON_ROLES = {roles!r}\n"
                f"U9_SELECTED_VIDEO_IDS = {selected!r}\n"
                f"U9_VIDEO_LABELS = {labels!r}\n"
                f"U9_EVENT_FRAMES = {events!r}\n"
                'U9_ANNOTATOR = "coach"\n'
                "OVERWRITE_U9_PHASE_EVENTS = False\n"
                "RUN_U9_MANUAL_BBOX_LABELING = False\n"
                "OVERWRITE_U9_GROUND_TRUTH = False"
            )

            self._execute_notebook(run_root, notebook)

            split = pd.read_csv(
                run_root / "data" / "splits" / "evaluation_videos.csv"
            )
            phase_events = pd.read_csv(
                run_root / "data" / "annotations" / "phase_events.csv"
            )
            self.assertEqual(len(split), 5)
            self.assertEqual(
                split["split_role"].value_counts().to_dict(),
                {
                    "model_selection_pilot": 3,
                    "rule_development": 1,
                    "rule_blind": 1,
                },
            )
            self.assertEqual(len(phase_events), 15)
            self.assertFalse(
                (run_root / "data" / "annotations" / "pilot_frame_ground_truth.csv").exists()
            )

    def test_u9_rejects_core_training_person_in_evaluation_roles(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "pyproject.toml").touch()
            video_directory = run_root / "data" / "raw" / "videos"
            video_directory.mkdir(parents=True)
            video_ids = (
                "b00_sn_f_01",
                "a01_sn_f_01",
                "c01_sn_f_01",
                "a02_sn_f_01",
                "a03_sn_f_01",
            )
            for video_id in video_ids:
                self._write_test_video(video_directory / f"{video_id}.mp4", frames=10)
            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
            settings = next(
                cell for cell in notebook.cells if cell.get("id") == "u9-settings"
            )
            roles = {
                "a01": "model_selection_pilot",
                "b00": "model_selection_pilot",
                "c01": "model_selection_pilot",
                "a02": "rule_development",
                "a03": "rule_blind",
            }
            selected = [
                "a01_sn_f_01",
                "b00_sn_f_01",
                "c01_sn_f_01",
                "a02_sn_f_01",
                "a03_sn_f_01",
            ]
            labels = {
                video_id: {"has_plate": "yes", "visible_body_side": "left"}
                for video_id in selected
            }
            settings.source = self._u9_settings_source(roles, selected, labels, {})

            with self.assertRaisesRegex(CellExecutionError, "Core training person"):
                self._execute_notebook(run_root, notebook)

    def test_u9_rejects_ground_truth_from_different_sample_frames(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "pyproject.toml").touch()
            video_directory = run_root / "data" / "raw" / "videos"
            video_directory.mkdir(parents=True)
            video_ids = (
                "b00_sn_f_01",
                "a01_sn_f_01",
                "b01_sn_f_01",
                "c01_sn_f_01",
                "a02_sn_f_01",
                "a03_sn_f_01",
            )
            for video_id in video_ids:
                self._write_test_video(video_directory / f"{video_id}.mp4", frames=20)
            roles = {
                "a01": "model_selection_pilot",
                "b01": "model_selection_pilot",
                "c01": "model_selection_pilot",
                "a02": "rule_development",
                "a03": "rule_blind",
            }
            selected = [
                "a01_sn_f_01",
                "b01_sn_f_01",
                "c01_sn_f_01",
                "a02_sn_f_01",
                "a03_sn_f_01",
            ]
            labels = {
                video_id: {"has_plate": "yes", "visible_body_side": "left"}
                for video_id in selected
            }
            events = {
                video_id: {
                    "bar_off_floor": 2,
                    "knee_pass": 5,
                    "max_extension": 9,
                    "catch_stable": 14,
                    "recovery_stable": 19,
                }
                for video_id in selected[:3]
            }
            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
            settings = next(
                cell for cell in notebook.cells if cell.get("id") == "u9-settings"
            )
            settings.source = self._u9_settings_source(
                roles, selected, labels, events
            )
            self._execute_notebook(run_root, notebook)

            evaluation_split = pd.read_csv(
                run_root / "data" / "splits" / "evaluation_videos.csv"
            )
            pilot = evaluation_split.loc[
                evaluation_split["split_role"] == "model_selection_pilot"
            ]
            stale_samples = pd.DataFrame(
                [
                    {"video_id": video_id, "frame_index": frame_index}
                    for video_id in pilot["video_id"]
                    for frame_index in range(10)
                ]
            )
            stale = build_pilot_ground_truth_template(stale_samples, evaluation_split)
            stale.loc[:, [
                "plate_bbox_x", "plate_bbox_y",
                "plate_bbox_width", "plate_bbox_height",
            ]] = [1, 1, 5, 5]
            stale.loc[:, [
                "lifter_bbox_x", "lifter_bbox_y",
                "lifter_bbox_width", "lifter_bbox_height",
            ]] = [10, 1, 20, 40]
            ground_truth_path = (
                run_root / "data" / "annotations" / "pilot_frame_ground_truth.csv"
            )
            save_pilot_ground_truth(stale, evaluation_split, ground_truth_path)

            with self.assertRaisesRegex(
                CellExecutionError,
                "OVERWRITE_U9_GROUND_TRUTH",
            ):
                self._execute_notebook(run_root, notebook)

    @staticmethod
    def _execute_notebook(
        run_root: Path,
        notebook: nbformat.NotebookNode | None = None,
    ) -> nbformat.NotebookNode:
        if notebook is None:
            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        executor = ExecutePreprocessor(timeout=60, kernel_name="python3")
        executed_notebook, _ = executor.preprocess(
            notebook,
            {"metadata": {"path": run_root}},
        )
        return executed_notebook

    @staticmethod
    def _write_test_video(path: Path, *, frames: int = 5) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (64, 48),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the test MP4")
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        for _ in range(frames):
            writer.write(frame)
        writer.release()

    @staticmethod
    def _u9_settings_source(
        roles: dict[str, str],
        selected: list[str],
        labels: dict[str, dict[str, str]],
        events: dict[str, dict[str, int]],
    ) -> str:
        return (
            "RUN_U9_SETUP = True\n"
            f"U9_EVALUATION_PERSON_ROLES = {roles!r}\n"
            f"U9_SELECTED_VIDEO_IDS = {selected!r}\n"
            f"U9_VIDEO_LABELS = {labels!r}\n"
            f"U9_EVENT_FRAMES = {events!r}\n"
            'U9_ANNOTATOR = "coach"\n'
            "OVERWRITE_U9_PHASE_EVENTS = False\n"
            "RUN_U9_MANUAL_BBOX_LABELING = False\n"
            "OVERWRITE_U9_GROUND_TRUTH = False"
        )
    
    def test_u9_has_ground_truth_overlay_contact_sheet(self) -> None:
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

        overlay_cells = [
            cell
            for cell in notebook.cells
            if "U9_GROUND_TRUTH_CONTACT_SHEET_PATH" in cell.source
        ]

        self.assertEqual(len(overlay_cells), 1)
        overlay_cell = overlay_cells[0]

        for expected in (
            '"plate"',
            '"lifter"',
            '{prefix}_bbox_x',
            '{prefix}_bbox_y',
            '{prefix}_bbox_width',
            '{prefix}_bbox_height',
            "cv2.rectangle",
            "figure.savefig",
            "U9_GROUND_TRUTH_REVIEWED",
        ):
            self.assertIn(expected, overlay_cell.source)
            
if __name__ == "__main__":
    unittest.main()
