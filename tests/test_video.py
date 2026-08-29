from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from snatch_analysis.video import (
    iter_video_frames,
    read_video_frame,
    read_video_properties,
)


class VideoTests(unittest.TestCase):
    def test_properties_use_decoded_count_when_container_overreports(self) -> None:
        capture = _OverreportedFrameCountCapture(
            reported_frame_count=5,
            decoded_frame_count=3,
        )

        with patch("snatch_analysis.video.cv2.VideoCapture", return_value=capture):
            properties = read_video_properties("overreported.mp4")

        self.assertEqual(properties["frame_count"], 3)
        self.assertAlmostEqual(properties["duration_sec"], 3 / 24, places=6)
        self.assertTrue(capture.released)

    def test_iterates_every_frame_with_index_timestamp_and_decoded_shape(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "five_frames.mp4"
            self._write_test_video(video_path)

            properties = read_video_properties(video_path)
            frames = list(iter_video_frames(video_path, expected_frame_count=5))

            self.assertAlmostEqual(properties["fps"], 29.97, places=3)
            self.assertEqual([item[0] for item in frames], list(range(5)))
            for index, timestamp_sec, _ in frames:
                self.assertAlmostEqual(timestamp_sec, index / 29.97, places=5)
            self.assertTrue(all(item[2].shape == (48, 64, 3) for item in frames))
            self.assertGreater(frames[-1][2].mean(), frames[0][2].mean() + 100)

            last_frame = read_video_frame(video_path, 4)
            self.assertGreater(last_frame.mean(), frames[0][2].mean() + 100)

    def test_raises_when_decode_ends_before_expected_frame_count(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "short.mp4"
            self._write_test_video(video_path)

            with self.assertRaisesRegex(
                RuntimeError,
                r"expected=6, actual=5, last_success_index=4",
            ):
                list(iter_video_frames(video_path, expected_frame_count=6))

    def test_raises_when_opened_video_cannot_decode_its_first_frame(self) -> None:
        capture = _OpenedEmptyCapture()

        with patch("snatch_analysis.video.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "Could not decode first frame"):
                list(iter_video_frames("empty.mp4"))

        self.assertTrue(capture.released)

    @staticmethod
    def _write_test_video(path: Path) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            29.97,
            (64, 48),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the test MP4")

        for index in range(5):
            frame = np.full((48, 64, 3), index * 40, dtype=np.uint8)
            writer.write(frame)
        writer.release()


class _OpenedEmptyCapture:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return 29.97 if property_id == cv2.CAP_PROP_FPS else 0.0

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        self.released = True


class _OverreportedFrameCountCapture:
    def __init__(self, reported_frame_count: int, decoded_frame_count: int) -> None:
        self.reported_frame_count = reported_frame_count
        self.remaining_frames = decoded_frame_count
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        properties = {
            cv2.CAP_PROP_FPS: 24.0,
            cv2.CAP_PROP_FRAME_COUNT: float(self.reported_frame_count),
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
        }
        return properties.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.remaining_frames == 0:
            return False, None
        self.remaining_frames -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


if __name__ == "__main__":
    unittest.main()
