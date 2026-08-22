from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import numpy as np
import pandas as pd


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
