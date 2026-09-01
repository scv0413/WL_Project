from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import numpy as np
import pandas as pd

from snatch_analysis.phases import (
    build_phase_by_frame,
    prepare_phase_annotations,
    validate_phase_events,
)


VALID_EVENTS = {
    "bar_off_floor": 2,
    "knee_pass": 4,
    "max_extension": 6,
    "catch_stable": 7,
    "recovery_stable": 9,
}
H_EVENTS = {
    "bar_off_floor": 2,
    "knee_pass": 4,
    "max_extension": 6,
    "catch_stable": 7,
    "recovery_stable": None,
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_pose_baseline.ipynb"


class PhaseTests(unittest.TestCase):
    def test_assigns_every_frame_by_the_phase_boundary_contract(self) -> None:
        phases = build_phase_by_frame("b00_sn_f_01", VALID_EVENTS, frame_count=10)

        self.assertEqual(phases["frame_index"].tolist(), list(range(10)))
        self.assertEqual(
            phases["phase"].tolist(),
            [
                "Set-up",
                "Set-up",
                "First Pull",
                "First Pull",
                "Second Pull",
                "Second Pull",
                "Drop Under & Catch",
                "Drop Under & Catch",
                "Recovery",
                "Recovery",
            ],
        )
        self.assertEqual(
            phases["recovery_complete"].tolist(),
            [False] * 9 + [True],
        )
        self.assertEqual(phases["phase_status"].tolist(), ["ok"] * 10)
        self.assertEqual(phases["na_reason"].tolist(), [""] * 10)

    def test_accepts_h_events_with_no_recovery_frame(self) -> None:
        self.assertEqual(
            validate_phase_events(H_EVENTS, frame_count=10, completion="h"),
            H_EVENTS,
        )

    def test_rejects_completion_contract_mismatches(self) -> None:
        with self.assertRaisesRegex(ValueError, "recovery_stable.*h"):
            validate_phase_events(
                {**H_EVENTS, "recovery_stable": 9},
                frame_count=10,
                completion="h",
            )
        with self.assertRaisesRegex(ValueError, "recovery_stable"):
            validate_phase_events(H_EVENTS, frame_count=10, completion="f")
        with self.assertRaisesRegex(ValueError, "unknown completion"):
            validate_phase_events(VALID_EVENTS, frame_count=10, completion="x")

    def test_marks_h_frames_after_catch_as_video_ended(self) -> None:
        phases = build_phase_by_frame(
            "b00_sn_h_01", H_EVENTS, frame_count=10, completion="h"
        )

        self.assertEqual(
            phases["phase"].tolist(),
            [
                "Set-up",
                "Set-up",
                "First Pull",
                "First Pull",
                "Second Pull",
                "Second Pull",
                "Drop Under & Catch",
                "Drop Under & Catch",
                "N/A",
                "N/A",
            ],
        )
        self.assertEqual(
            phases["phase_status"].tolist(),
            ["ok"] * 8 + ["not_applicable"] * 2,
        )
        self.assertEqual(
            phases["na_reason"].tolist(), [""] * 8 + ["video_ended"] * 2
        )
        self.assertTrue(phases["recovery_complete"].isna().all())

    def test_reuses_legacy_f_source_without_na_reason(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "phase_events.csv"
            pd.DataFrame(
                [
                    ["b00_sn_f_01", name, frame, "coach", ""]
                    for name, frame in VALID_EVENTS.items()
                ],
                columns=[
                    "video_id",
                    "event_name",
                    "frame_index",
                    "annotator",
                    "notes",
                ],
            ).to_csv(annotation_path, index=False)

            events, phases, action = prepare_phase_annotations(
                video_id="b00_sn_f_01",
                event_frames={},
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=root / "artifacts",
            )

            self.assertEqual(action, "reused")
            self.assertEqual(events["na_reason"].tolist(), [""] * 5)
            self.assertEqual(phases["phase"].iloc[-1], "Recovery")

    def test_persists_h_recovery_as_nullable_video_ended(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            events, phases, action = prepare_phase_annotations(
                video_id="b00_sn_h_01",
                event_frames=H_EVENTS,
                frame_count=10,
                completion="h",
                annotation_path=root / "phase_events.csv",
                artifact_directory=root / "artifacts",
            )

            recovery = events.set_index("event_name").loc["recovery_stable"]
            self.assertEqual(action, "created")
            self.assertTrue(pd.isna(recovery["frame_index"]))
            self.assertEqual(recovery["na_reason"], "video_ended")
            self.assertEqual(
                pd.read_csv(root / "phase_events.csv").columns.tolist(),
                [
                    "video_id",
                    "event_name",
                    "frame_index",
                    "annotator",
                    "notes",
                    "na_reason",
                ],
            )
            self.assertTrue(phases["recovery_complete"].isna().all())

    def test_rejects_duplicate_or_reversed_event_frames(self) -> None:
        invalid_cases = {
            "duplicate": {**VALID_EVENTS, "max_extension": 4},
            "reversed": {**VALID_EVENTS, "knee_pass": 1},
        }

        for case_name, event_frames in invalid_cases.items():
            with self.subTest(case_name=case_name):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    validate_phase_events(event_frames, frame_count=10)

    def test_rejects_events_outside_the_video(self) -> None:
        invalid_cases = {
            "negative": {**VALID_EVENTS, "bar_off_floor": -1},
            "past_last_frame": {**VALID_EVENTS, "recovery_stable": 10},
        }

        for case_name, event_frames in invalid_cases.items():
            with self.subTest(case_name=case_name):
                with self.assertRaisesRegex(ValueError, "outside frame range"):
                    validate_phase_events(event_frames, frame_count=10)

    def test_rejects_missing_event_names(self) -> None:
        incomplete_events = dict(VALID_EVENTS)
        incomplete_events.pop("catch_stable")

        with self.assertRaisesRegex(ValueError, "missing events: catch_stable"):
            validate_phase_events(incomplete_events, frame_count=10)

    def test_reuses_existing_source_until_overwrite_is_explicit(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "data" / "annotations" / "phase_events.csv"
            artifact_directory = root / "artifacts" / "b00_sn_f_01"

            events, phases, action = prepare_phase_annotations(
                video_id="b00_sn_f_01",
                event_frames=VALID_EVENTS,
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=artifact_directory,
                annotator="coach",
            )

            self.assertEqual(action, "created")
            self.assertEqual(len(events), 5)
            self.assertEqual(len(phases), 10)
            self.assertTrue(annotation_path.exists())
            self.assertTrue((artifact_directory / "phase_events.csv").exists())
            self.assertTrue((artifact_directory / "phase_by_frame.csv").exists())

            manually_edited = pd.read_csv(annotation_path)
            manually_edited["notes"] = manually_edited["notes"].astype(object)
            manually_edited.loc[
                manually_edited["event_name"] == "catch_stable", "notes"
            ] = "캐치가 안정된 첫 프레임"
            manually_edited.to_csv(annotation_path, index=False)

            changed_events = {**VALID_EVENTS, "recovery_stable": 8}
            reused, _, action = prepare_phase_annotations(
                video_id="b00_sn_f_01",
                event_frames=changed_events,
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=artifact_directory,
            )

            self.assertEqual(action, "reused")
            self.assertEqual(
                reused.set_index("event_name").loc["recovery_stable", "frame_index"],
                9,
            )
            self.assertEqual(
                reused.set_index("event_name").loc["catch_stable", "notes"],
                "캐치가 안정된 첫 프레임",
            )

            overwritten, _, action = prepare_phase_annotations(
                video_id="b00_sn_f_01",
                event_frames=changed_events,
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=artifact_directory,
                overwrite=True,
            )

            self.assertEqual(action, "overwritten")
            self.assertEqual(
                overwritten.set_index("event_name").loc[
                    "recovery_stable", "frame_index"
                ],
                8,
            )
            saved = pd.read_csv(annotation_path)
            self.assertEqual(len(saved), 5)

    def test_prepare_requires_manual_events_for_a_new_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            with self.assertRaisesRegex(
                ValueError,
                "다섯 이벤트 프레임을 직접 입력",
            ):
                prepare_phase_annotations(
                    video_id="b00_sn_f_01",
                    event_frames={name: None for name in VALID_EVENTS},
                    frame_count=10,
                    annotation_path=root / "phase_events.csv",
                    artifact_directory=root / "artifacts",
                )

    def test_rejects_unknown_events_in_an_existing_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "phase_events.csv"
            prepare_phase_annotations(
                video_id="b00_sn_f_01",
                event_frames=VALID_EVENTS,
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=root / "artifacts",
            )
            events = pd.read_csv(annotation_path)
            events.loc[len(events)] = [
                "b00_sn_f_01",
                "unexpected_event",
                5,
                "",
                "",
                "",
            ]
            events.to_csv(annotation_path, index=False)

            with self.assertRaisesRegex(ValueError, "unknown events"):
                prepare_phase_annotations(
                    video_id="b00_sn_f_01",
                    event_frames=VALID_EVENTS,
                    frame_count=10,
                    annotation_path=annotation_path,
                    artifact_directory=root / "artifacts",
                )

    def test_preserves_other_videos_when_appending_and_overwriting(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "phase_events.csv"
            prepare_phase_annotations(
                video_id="a00_sn_f_01",
                event_frames=VALID_EVENTS,
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=root / "artifacts" / "a00_sn_f_01",
                notes="preserve me",
            )
            original_other_rows = pd.read_csv(annotation_path)

            _, _, action = prepare_phase_annotations(
                video_id="b00_sn_f_01",
                event_frames=VALID_EVENTS,
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=root / "artifacts" / "b00_sn_f_01",
            )
            self.assertEqual(action, "appended")

            prepare_phase_annotations(
                video_id="b00_sn_f_01",
                event_frames={**VALID_EVENTS, "recovery_stable": 8},
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=root / "artifacts" / "b00_sn_f_01",
                overwrite=True,
            )
            saved_other_rows = pd.read_csv(annotation_path).loc[
                lambda table: table["video_id"] == "a00_sn_f_01"
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(saved_other_rows, original_other_rows)

    def test_failed_write_does_not_damage_existing_annotations(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "phase_events.csv"
            prepare_phase_annotations(
                video_id="b00_sn_f_01",
                event_frames=VALID_EVENTS,
                frame_count=10,
                annotation_path=annotation_path,
                artifact_directory=root / "artifacts",
            )
            original_bytes = annotation_path.read_bytes()

            def fail_after_partial_write(
                _: pd.DataFrame,
                destination: str | Path,
                *args: object,
                **kwargs: object,
            ) -> None:
                Path(destination).write_text("partial", encoding="utf-8")
                raise OSError("disk full")

            with patch.object(
                pd.DataFrame,
                "to_csv",
                autospec=True,
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    prepare_phase_annotations(
                        video_id="b00_sn_f_01",
                        event_frames={**VALID_EVENTS, "recovery_stable": 8},
                        frame_count=10,
                        annotation_path=annotation_path,
                        artifact_directory=root / "artifacts",
                        overwrite=True,
                    )

            self.assertEqual(annotation_path.read_bytes(), original_bytes)

    def test_notebook_requires_manual_events_before_saving(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = self._build_notebook_input(Path(temporary_directory))

            with self.assertRaisesRegex(
                CellExecutionError,
                "다섯 이벤트 프레임을 직접 입력",
            ):
                self._execute_notebook(run_root)

    def test_notebook_reports_an_unreadable_video_before_numeric_checks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = self._build_notebook_input(Path(temporary_directory))
            (run_root / "data" / "raw" / "videos" / "b00_sn_f_01.mp4").unlink()

            with self.assertRaisesRegex(
                CellExecutionError,
                "B00 영상 기준 실패: read_status_ok",
            ):
                self._execute_notebook(run_root)

    def test_notebook_saves_validated_events_and_ten_phase_rows(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_root = self._build_notebook_input(Path(temporary_directory))
            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
            event_cell = next(
                cell for cell in notebook.cells if cell.get("id") == "phase-event-inputs"
            )
            event_cell["source"] = "EVENT_FRAMES = " + repr(VALID_EVENTS)

            executed = self._execute_notebook(run_root, notebook)
            events = pd.read_csv(
                run_root / "data" / "annotations" / "phase_events.csv"
            )
            phases = pd.read_csv(
                run_root
                / "artifacts"
                / "b00_sn_f_01"
                / "phase_by_frame.csv"
            )

            self.assertEqual(len(events), 5)
            self.assertEqual(len(phases), 10)
            self.assertEqual(phases.loc[0, "phase"], "Set-up")
            self.assertEqual(phases.loc[9, "phase"], "Recovery")
            summary_cell = next(
                cell
                for cell in executed.cells
                if cell.get("id") == "phase-save-summary"
            )
            summary_text = summary_cell["outputs"][-1]["data"]["text/plain"]
            self.assertIn("created", summary_text)
            self.assertIn("10", summary_text)

    @staticmethod
    def _build_notebook_input(run_root: Path) -> Path:
        (run_root / "pyproject.toml").touch()
        video_directory = run_root / "data" / "raw" / "videos"
        metadata_directory = run_root / "data" / "metadata"
        video_directory.mkdir(parents=True)
        metadata_directory.mkdir(parents=True)
        video_path = video_directory / "b00_sn_f_01.mp4"

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (64, 48),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the test MP4")
        for frame_index in range(10):
            frame = np.full((48, 64, 3), frame_index * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        pd.DataFrame(
            [
                {
                    "video_id": "b00_sn_f_01",
                    "path": video_path.as_posix(),
                    "fps": 25.0,
                    "frame_count": 10,
                    "width": 64,
                    "height": 48,
                    "read_status": "ok",
                }
            ]
        ).to_csv(metadata_directory / "videos.csv", index=False)
        return run_root

    @staticmethod
    def _execute_notebook(
        run_root: Path,
        notebook: nbformat.NotebookNode | None = None,
    ) -> nbformat.NotebookNode:
        if notebook is None:
            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        executor = ExecutePreprocessor(timeout=60, kernel_name="python3")
        executed, _ = executor.preprocess(
            notebook,
            {"metadata": {"path": run_root}},
        )
        return executed


if __name__ == "__main__":
    unittest.main()
