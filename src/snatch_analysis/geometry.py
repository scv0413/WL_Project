"""Pose 모델과 독립적인 2차원 기하 계산."""

from collections.abc import Sequence
import math

Point = Sequence[float | None] | None
Measurement = dict[str, float | str | None]


def _coordinates(point: Point) -> tuple[float, float] | None:
    if point is None or len(point) != 2 or point[0] is None or point[1] is None:
        return None

    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, OverflowError):
        return None

    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _angle_between(
    vector_a: tuple[float, float],
    vector_b: tuple[float, float],
) -> Measurement:
    if not all(math.isfinite(value) for value in (*vector_a, *vector_b)):
        return {
            "angle_deg": None,
            "status": "not_available",
            "reason": "non_finite_vector",
        }

    scale_a = max(abs(vector_a[0]), abs(vector_a[1]))
    scale_b = max(abs(vector_b[0]), abs(vector_b[1]))
    if scale_a == 0.0 or scale_b == 0.0:
        return {
            "angle_deg": None,
            "status": "not_available",
            "reason": "zero_length_vector",
        }

    normalized_a = (vector_a[0] / scale_a, vector_a[1] / scale_a)
    normalized_b = (vector_b[0] / scale_b, vector_b[1] / scale_b)
    cross = normalized_a[0] * normalized_b[1] - normalized_a[1] * normalized_b[0]
    dot = normalized_a[0] * normalized_b[0] + normalized_a[1] * normalized_b[1]
    angle_deg = math.degrees(math.atan2(abs(cross), dot))
    return {"angle_deg": angle_deg, "status": "ok", "reason": ""}


def calculate_internal_angle(
    point_a: Point,
    vertex: Point,
    point_c: Point,
) -> Measurement:
    """세 점이 만드는 0~180도 내부각과 측정 상태를 반환한다."""
    a = _coordinates(point_a)
    b = _coordinates(vertex)
    c = _coordinates(point_c)
    if a is None or b is None or c is None:
        return {
            "angle_deg": None,
            "status": "not_available",
            "reason": "missing_coordinate",
        }

    vector_ba = (a[0] - b[0], a[1] - b[1])
    vector_bc = (c[0] - b[0], c[1] - b[1])
    return _angle_between(vector_ba, vector_bc)


def calculate_trunk_inclination(shoulder: Point, hip: Point) -> Measurement:
    """엉덩이→어깨 벡터와 화면 위쪽 수직선 사이의 각도를 반환한다."""
    shoulder_coordinates = _coordinates(shoulder)
    hip_coordinates = _coordinates(hip)
    if shoulder_coordinates is None or hip_coordinates is None:
        return {
            "angle_deg": None,
            "status": "not_available",
            "reason": "missing_coordinate",
        }

    trunk_vector = (
        shoulder_coordinates[0] - hip_coordinates[0],
        shoulder_coordinates[1] - hip_coordinates[1],
    )
    return _angle_between(trunk_vector, (0.0, -1.0))
