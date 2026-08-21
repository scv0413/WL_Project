import math
import unittest

from snatch_analysis.geometry import (
    calculate_internal_angle,
    calculate_trunk_inclination,
)


class GeometryTest(unittest.TestCase):
    def test_internal_angle_for_right_and_straight_angles(self) -> None:
        right = calculate_internal_angle((1.0, 0.0), (0.0, 0.0), (0.0, 1.0))
        straight = calculate_internal_angle((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))

        self.assertEqual(right["status"], "ok")
        self.assertAlmostEqual(right["angle_deg"], 90.0)
        self.assertAlmostEqual(straight["angle_deg"], 180.0)

    def test_missing_or_non_finite_coordinate_returns_reason(self) -> None:
        missing = calculate_internal_angle(None, (0.0, 0.0), (1.0, 0.0))
        non_finite = calculate_internal_angle(
            (math.nan, 0.0), (0.0, 0.0), (1.0, 0.0)
        )

        for result in (missing, non_finite):
            self.assertIsNone(result["angle_deg"])
            self.assertEqual(result["status"], "not_available")
            self.assertEqual(result["reason"], "missing_coordinate")

    def test_zero_length_vector_returns_reason(self) -> None:
        result = calculate_internal_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))

        self.assertIsNone(result["angle_deg"])
        self.assertEqual(result["status"], "not_available")
        self.assertEqual(result["reason"], "zero_length_vector")

    def test_internal_angle_handles_extreme_finite_vectors(self) -> None:
        tiny = math.ulp(0.0)
        right = calculate_internal_angle((tiny, 0.0), (0.0, 0.0), (0.0, tiny))
        parallel = calculate_internal_angle(
            (1e200, 1e200), (0.0, 0.0), (2e200, 2e200)
        )
        overflow = calculate_internal_angle(
            (1e308, 0.0), (-1e308, 0.0), (0.0, 1.0)
        )

        self.assertAlmostEqual(right["angle_deg"], 90.0)
        self.assertAlmostEqual(parallel["angle_deg"], 0.0)
        self.assertIsNone(overflow["angle_deg"])
        self.assertEqual(overflow["reason"], "non_finite_vector")

    def test_trunk_inclination_uses_upward_screen_vertical(self) -> None:
        vertical = calculate_trunk_inclination((0.0, 0.0), (0.0, 2.0))
        horizontal = calculate_trunk_inclination((2.0, 2.0), (0.0, 2.0))
        downward = calculate_trunk_inclination((0.0, 4.0), (0.0, 2.0))

        self.assertAlmostEqual(vertical["angle_deg"], 0.0)
        self.assertAlmostEqual(horizontal["angle_deg"], 90.0)
        self.assertAlmostEqual(downward["angle_deg"], 180.0)
        self.assertTrue(0.0 <= downward["angle_deg"] <= 180.0)

    def test_trunk_inclination_handles_missing_and_large_coordinates(self) -> None:
        missing_hip = calculate_trunk_inclination((0.0, 0.0), None)
        invalid_shoulder = calculate_trunk_inclination(
            (math.nan, 0.0), (0.0, 0.0)
        )
        coincident = calculate_trunk_inclination((0.0, 0.0), (0.0, 0.0))
        large = calculate_trunk_inclination((1e20, 1e20 - 1e8), (1e20, 1e20))

        for result in (missing_hip, invalid_shoulder):
            self.assertIsNone(result["angle_deg"])
            self.assertEqual(result["status"], "not_available")
            self.assertEqual(result["reason"], "missing_coordinate")
        self.assertIsNone(coincident["angle_deg"])
        self.assertEqual(coincident["reason"], "zero_length_vector")
        self.assertEqual(large["status"], "ok")
        self.assertAlmostEqual(large["angle_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
