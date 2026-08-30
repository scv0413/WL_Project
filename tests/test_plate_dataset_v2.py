import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from snatch_analysis.plate import (
    cancel_plate_label,
    sample_time_bin_frames,
    save_u10_plate_dataset_yaml,
    validate_u10_plate_selection,
)


class TimeBinSamplingTests(unittest.TestCase):
    def test_samples_four_unique_frames_from_each_of_five_time_bins(self) -> None:
        sampled = sample_time_bin_frames(
            "a04_sn_f_01",
            frame_count=100,
            sample_count=20,
        )

        self.assertEqual(len(sampled), 20)
        self.assertFalse(sampled["frame_index"].duplicated().any())
        self.assertEqual(
            sampled["sampling_bin"].value_counts().sort_index().to_dict(),
            {1: 4, 2: 4, 3: 4, 4: 4, 5: 4},
        )
        self.assertTrue(sampled["frame_index"].between(0, 99).all())

    def test_rejects_sample_count_other_than_twenty(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            sample_time_bin_frames("a04_sn_f_01", 100, sample_count=25)

    def test_rejects_more_samples_than_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_count"):
            sample_time_bin_frames("a04_sn_f_01", 19, sample_count=20)


class PlateDatasetSelectionTests(unittest.TestCase):
    def test_accepts_four_train_and_two_validation_videos_without_person_leakage(
        self,
    ) -> None:
        validated = validate_u10_plate_selection(
            self._selection(),
            self._manifest(),
            baseline_video_id="b00_sn_f_01",
        )

        self.assertEqual(len(validated), 6)
        self.assertEqual(
            validated["dataset_role"].value_counts().to_dict(),
            {"train": 4, "validation": 2},
        )

    def test_rejects_same_person_in_train_and_validation(self) -> None:
        selection = self._selection()
        selection.loc[5, ["video_id", "person_id"]] = [
            "a04_sn_h_02",
            "a04",
        ]

        with self.assertRaisesRegex(ValueError, "person_id"):
            validate_u10_plate_selection(
                selection,
                self._manifest(),
                baseline_video_id="b00_sn_f_01",
            )

    def test_rejects_unknown_has_plate(self) -> None:
        manifest = self._manifest()
        manifest.loc[
            manifest["video_id"] == "a04_sn_f_01",
            "has_plate",
        ] = "unknown"

        with self.assertRaisesRegex(ValueError, "has_plate"):
            validate_u10_plate_selection(
                self._selection(),
                manifest,
                baseline_video_id="b00_sn_f_01",
            )

    def test_rejects_evaluation_role(self) -> None:
        manifest = self._manifest()
        manifest.loc[
            manifest["video_id"] == "a04_sn_f_01",
            "split_role",
        ] = "model_selection_pilot"

        with self.assertRaisesRegex(ValueError, "training_development"):
            validate_u10_plate_selection(
                self._selection(),
                manifest,
                baseline_video_id="b00_sn_f_01",
            )

    def test_rejects_wrong_video_count_and_duplicate_video(self) -> None:
        too_short = self._selection().iloc[:5].copy()
        duplicated = self._selection()
        duplicated.loc[5, "video_id"] = duplicated.loc[0, "video_id"]

        for selection in (too_short, duplicated):
            with self.subTest(video_ids=selection["video_id"].tolist()):
                with self.assertRaisesRegex(ValueError, "신규 영상 6개"):
                    validate_u10_plate_selection(
                        selection,
                        self._manifest(),
                        baseline_video_id="b00_sn_f_01",
                    )

    def test_rejects_wrong_train_validation_counts(self) -> None:
        selection = self._selection()
        selection.loc[4, "dataset_role"] = "train"

        with self.assertRaisesRegex(ValueError, "train 4개"):
            validate_u10_plate_selection(
                selection,
                self._manifest(),
                baseline_video_id="b00_sn_f_01",
            )

    def test_rejects_unreadable_video(self) -> None:
        manifest = self._manifest()
        manifest.loc[
            manifest["video_id"] == "b02_sn_f_01",
            "read_status",
        ] = "error"

        with self.assertRaisesRegex(ValueError, "read_status"):
            validate_u10_plate_selection(
                self._selection(),
                manifest,
                baseline_video_id="b00_sn_f_01",
            )

    def test_rejects_fewer_than_four_people(self) -> None:
        selection = self._selection()
        selection.loc[1:3, ["video_id", "person_id"]] = [
            ["a04_sn_f_02", "a04"],
            ["a04_sn_f_03", "a04"],
            ["a04_sn_f_04", "a04"],
        ]

        with self.assertRaisesRegex(ValueError, "최소 네 사람"):
            validate_u10_plate_selection(
                selection,
                self._manifest(),
                baseline_video_id="b00_sn_f_01",
            )

    def test_rejects_blank_selection_reason(self) -> None:
        selection = self._selection()
        selection.loc[0, "selection_reason"] = " "

        with self.assertRaisesRegex(ValueError, "selection_reason"):
            validate_u10_plate_selection(
                selection,
                self._manifest(),
                baseline_video_id="b00_sn_f_01",
            )

    def test_rejects_b00_in_new_video_selection(self) -> None:
        selection = self._selection()
        selection.loc[0, ["video_id", "person_id"]] = ["b00_sn_f_01", "b00"]

        with self.assertRaisesRegex(ValueError, "B00 baseline"):
            validate_u10_plate_selection(
                selection,
                self._manifest(),
                baseline_video_id="b00_sn_f_01",
            )

    @staticmethod
    def _selection() -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("a04_sn_f_01", "a04", "train", "green plate and wide view"),
                ("a05_sn_f_01", "a05", "train", "small plate in frame"),
                ("b02_sn_f_01", "b02", "train", "dark plate and new background"),
                ("c02_sn_f_01", "c02", "train", "partial plate occlusion"),
                (
                    "b03_sn_f_01",
                    "b03",
                    "validation",
                    "different camera distance",
                ),
                (
                    "c03_sn_f_01",
                    "c03",
                    "validation",
                    "different lifter and floor",
                ),
            ],
            columns=(
                "video_id",
                "person_id",
                "dataset_role",
                "selection_reason",
            ),
        )

    @staticmethod
    def _manifest() -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("a04_sn_f_01", "a04", "training_development", "ok", "yes"),
                ("a04_sn_f_02", "a04", "training_development", "ok", "yes"),
                ("a04_sn_f_03", "a04", "training_development", "ok", "yes"),
                ("a04_sn_f_04", "a04", "training_development", "ok", "yes"),
                ("a04_sn_h_02", "a04", "training_development", "ok", "yes"),
                ("a05_sn_f_01", "a05", "training_development", "ok", "yes"),
                ("b02_sn_f_01", "b02", "training_development", "ok", "yes"),
                ("c02_sn_f_01", "c02", "training_development", "ok", "yes"),
                ("b03_sn_f_01", "b03", "training_development", "ok", "yes"),
                ("c03_sn_f_01", "c03", "training_development", "ok", "yes"),
                ("b00_sn_f_01", "b00", "training_development", "ok", "yes"),
                ("a01_sn_f_01", "a01", "model_selection_pilot", "ok", "yes"),
            ],
            columns=(
                "video_id",
                "person_id",
                "split_role",
                "read_status",
                "has_plate",
            ),
        )


class PlateDatasetFileSafetyTests(unittest.TestCase):
    def test_writes_yaml_only_for_current_split_and_exact_file_pairs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "v2"
            baseline_root = root / "baseline"
            video_split = self._video_split()
            sample_frames = self._sample_frames(video_split)
            self._write_dataset_files(
                dataset_root,
                baseline_root,
                sample_frames,
            )

            counts = save_u10_plate_dataset_yaml(
                dataset_root,
                baseline_root,
                video_split,
                sample_frames,
            )

            self.assertEqual(
                counts,
                {
                    "train_images": 110,
                    "train_labels": 110,
                    "validation_images": 40,
                    "validation_labels": 40,
                },
            )
            self.assertEqual(
                (dataset_root / "dataset.yaml").read_text(encoding="utf-8"),
                f"path: {dataset_root.resolve()}\n"
                "train: images/train\n"
                "val: images/validation\n"
                "names:\n"
                "  0: plate\n",
            )

    def test_removes_existing_yaml_when_stale_files_do_not_match_split(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "v2"
            baseline_root = root / "baseline"
            video_split = self._video_split()
            sample_frames = self._sample_frames(video_split)
            self._write_dataset_files(
                dataset_root,
                baseline_root,
                sample_frames,
            )
            stale_image = dataset_root / "images/train/old_split_frame.jpg"
            stale_image.touch()
            yaml_path = dataset_root / "dataset.yaml"
            yaml_path.write_text("stale", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dataset files do not match"):
                save_u10_plate_dataset_yaml(
                    dataset_root,
                    baseline_root,
                    video_split,
                    sample_frames,
                )

            self.assertFalse(yaml_path.exists())

    def test_removes_existing_yaml_when_sample_does_not_match_current_split(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "v2"
            baseline_root = root / "baseline"
            old_split = self._video_split()
            sample_frames = self._sample_frames(old_split)
            self._write_dataset_files(dataset_root, baseline_root, sample_frames)
            current_split = old_split.copy()
            current_split.loc[5, "video_id"] = "c04_sn_f_01"
            yaml_path = dataset_root / "dataset.yaml"
            yaml_path.write_text("stale", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "video_split and sample_frames"):
                save_u10_plate_dataset_yaml(
                    dataset_root,
                    baseline_root,
                    current_split,
                    sample_frames,
                )

            self.assertFalse(yaml_path.exists())

    def test_removes_existing_yaml_when_baseline_pair_is_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "v2"
            baseline_root = root / "baseline"
            video_split = self._video_split()
            sample_frames = self._sample_frames(video_split)
            self._write_dataset_files(dataset_root, baseline_root, sample_frames)
            next((baseline_root / "labels").glob("*.txt")).unlink()
            yaml_path = dataset_root / "dataset.yaml"
            yaml_path.write_text("stale", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "baseline"):
                save_u10_plate_dataset_yaml(
                    dataset_root,
                    baseline_root,
                    video_split,
                    sample_frames,
                )

            self.assertFalse(yaml_path.exists())

    def test_removes_existing_yaml_when_bbox_label_is_empty(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "v2"
            baseline_root = root / "baseline"
            video_split = self._video_split()
            sample_frames = self._sample_frames(video_split)
            self._write_dataset_files(dataset_root, baseline_root, sample_frames)
            next((dataset_root / "labels/train").glob("*.txt")).write_text(
                "", encoding="utf-8"
            )
            yaml_path = dataset_root / "dataset.yaml"
            yaml_path.write_text("stale", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "plate label"):
                save_u10_plate_dataset_yaml(
                    dataset_root,
                    baseline_root,
                    video_split,
                    sample_frames,
                )

            self.assertFalse(yaml_path.exists())

    def test_rejects_filenames_swapped_between_sample_frames(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "v2"
            baseline_root = root / "baseline"
            video_split = self._video_split()
            sample_frames = self._sample_frames(video_split)
            self._write_dataset_files(dataset_root, baseline_root, sample_frames)
            first_image = sample_frames.loc[0, "image_filename"]
            first_label = sample_frames.loc[0, "label_filename"]
            sample_frames.loc[0, ["image_filename", "label_filename"]] = sample_frames.loc[
                1, ["image_filename", "label_filename"]
            ].to_list()
            sample_frames.loc[1, ["image_filename", "label_filename"]] = [
                first_image,
                first_label,
            ]

            with self.assertRaisesRegex(ValueError, "filenames must match"):
                save_u10_plate_dataset_yaml(
                    dataset_root,
                    baseline_root,
                    video_split,
                    sample_frames,
                )

    def test_rejects_extra_images_ultralytics_would_find_recursively(self) -> None:
        for relative_path in (
            "images/train/old_split.png",
            "images/train/nested/old_split.jpg",
        ):
            with self.subTest(relative_path=relative_path):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    dataset_root = root / "v2"
                    baseline_root = root / "baseline"
                    video_split = self._video_split()
                    sample_frames = self._sample_frames(video_split)
                    self._write_dataset_files(
                        dataset_root,
                        baseline_root,
                        sample_frames,
                    )
                    extra_image = dataset_root / relative_path
                    extra_image.parent.mkdir(parents=True, exist_ok=True)
                    extra_image.touch()
                    yaml_path = dataset_root / "dataset.yaml"
                    yaml_path.write_text("stale", encoding="utf-8")

                    with self.assertRaisesRegex(
                        ValueError, "dataset files do not match"
                    ):
                        save_u10_plate_dataset_yaml(
                            dataset_root,
                            baseline_root,
                            video_split,
                            sample_frames,
                        )

                    self.assertFalse(yaml_path.exists())

    def test_cancelled_overwrite_removes_old_label(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "plate.txt"
            label_path.write_text("old bbox", encoding="utf-8")

            action = cancel_plate_label(label_path, overwrite=True)

            self.assertEqual(action, "plate_label_removed")
            self.assertFalse(label_path.exists())

    @staticmethod
    def _write_dataset_files(
        dataset_root: Path,
        baseline_root: Path,
        sample_frames: pd.DataFrame,
    ) -> None:
        baseline_stems = {f"b00_frame_{index:04d}" for index in range(30)}
        for directory_name, suffix in (("images", ".jpg"), ("labels", ".txt")):
            baseline_directory = baseline_root / directory_name
            baseline_directory.mkdir(parents=True, exist_ok=True)
            for stem in baseline_stems:
                output_path = baseline_directory / f"{stem}{suffix}"
                if suffix == ".txt":
                    output_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                else:
                    output_path.touch()

        for role in ("train", "validation"):
            role_samples = sample_frames.loc[sample_frames["dataset_role"].eq(role)]
            stems = {Path(name).stem for name in role_samples["image_filename"]}
            if role == "train":
                stems |= baseline_stems
            for directory_name, suffix in (("images", ".jpg"), ("labels", ".txt")):
                directory = dataset_root / directory_name / role
                directory.mkdir(parents=True, exist_ok=True)
                for stem in stems:
                    output_path = directory / f"{stem}{suffix}"
                    if suffix == ".txt":
                        output_path.write_text(
                            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                        )
                    else:
                        output_path.touch()

    @staticmethod
    def _video_split() -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("a04_sn_f_01", "train"),
                ("a05_sn_f_01", "train"),
                ("b02_sn_f_01", "train"),
                ("c02_sn_f_01", "train"),
                ("b03_sn_f_01", "validation"),
                ("c03_sn_f_01", "validation"),
            ],
            columns=("video_id", "dataset_role"),
        )

    @staticmethod
    def _sample_frames(video_split: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in video_split.itertuples(index=False):
            for frame_index in range(20):
                stem = f"{row.video_id}_frame_{frame_index:04d}"
                rows.append(
                    {
                        "video_id": row.video_id,
                        "frame_index": frame_index,
                        "dataset_role": row.dataset_role,
                        "image_filename": f"{stem}.jpg",
                        "label_filename": f"{stem}.txt",
                    }
                )
        return pd.DataFrame(rows)


class PlateDatasetV2NotebookTests(unittest.TestCase):
    @staticmethod
    def _notebook() -> dict:
        return json.loads(
            Path("notebooks/06_plate_dataset_v2.ipynb").read_text(encoding="utf-8")
        )

    def test_notebook_is_source_only(self) -> None:
        code_cells = [
            cell for cell in self._notebook()["cells"] if cell["cell_type"] == "code"
        ]
        self.assertTrue(all(cell["execution_count"] is None for cell in code_cells))
        self.assertTrue(all(cell["outputs"] == [] for cell in code_cells))

    def test_every_code_cell_compiles(self) -> None:
        for cell in self._notebook()["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")

if __name__ == "__main__":
    unittest.main()
