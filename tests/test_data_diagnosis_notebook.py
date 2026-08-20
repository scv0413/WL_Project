from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
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
            self._write_test_video(video_directory / "c01_sn_f_01.mp4")

            self._execute_notebook(run_root)
            manifest_path = run_root / "data" / "metadata" / "videos.csv"
            manifest = pd.read_csv(manifest_path)
            manifest.loc[0, "has_plate"] = "yes"
            manifest.loc[0, "camera_motion"] = "fixed"
            manifest.to_csv(manifest_path, index=False)

            self._execute_notebook(run_root)
            reloaded_manifest = pd.read_csv(manifest_path)

            self.assertEqual(reloaded_manifest.loc[0, "has_plate"], "yes")
            self.assertEqual(reloaded_manifest.loc[0, "camera_motion"], "fixed")

    @staticmethod
    def _execute_notebook(run_root: Path) -> None:
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        executor = ExecutePreprocessor(timeout=60, kernel_name="python3")
        executor.preprocess(notebook, {"metadata": {"path": run_root}})

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
