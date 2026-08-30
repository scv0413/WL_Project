import unittest

import pandas as pd

from snatch_analysis.plate import (
    sample_time_bin_frames,
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


if __name__ == "__main__":
    unittest.main()
