from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

SENSOR_TYPE_IR_MOTION = 0
SENSOR_TYPE_DOOR_BREAK = 1
SENSOR_TYPE_WINDOW_OPEN = 2

ROOM_TYPE_CORRIDOR = 0
ROOM_TYPE_HALLWAY = 1
ROOM_TYPE_KITCHEN = 2
ROOM_TYPE_LIVING_ROOM = 3
ROOM_TYPE_BEDROOM = 4

SUPPORTED_SENSOR_TYPES = {
    SENSOR_TYPE_IR_MOTION,
    SENSOR_TYPE_DOOR_BREAK,
    SENSOR_TYPE_WINDOW_OPEN,
}
SUPPORTED_ROOM_TYPES = {
    ROOM_TYPE_CORRIDOR,
    ROOM_TYPE_HALLWAY,
    ROOM_TYPE_KITCHEN,
    ROOM_TYPE_LIVING_ROOM,
    ROOM_TYPE_BEDROOM,
}


def detect_intrusion_rooms(
    sensor_vectors: Iterable[Sequence[int]],
    *,
    strict: bool = False,
    sensor_failure_threshold: int = 4,
) -> List[int]:
    """Detect rooms with intrusion from triggered sensor vectors.

    Args:
        sensor_vectors: Iterable of vectors in format
            [sensor_type, sensor_id, room_id, room_type], where:
            - sensor_type: 0 (IR), 1 (door break), 2 (window open)
            - room_type: 0 corridor, 1 hallway, 2 kitchen,
              3 living room, 4 bedroom
        strict: If True, malformed vectors raise ValueError.
            If False, malformed vectors are ignored.
        sensor_failure_threshold: Sensors with hit count greater than or equal
            to this value in one batch are considered faulty and ignored.

    Detection rule:
        - A room is marked as intrusion if it has a perimeter trigger
          (door/window) plus at least one more independent sensor.
        - For less noisy room types (hallway/living room/bedroom), an
          additional fallback is allowed: 3+ independent IR sensors.
        - Very noisy room types (corridor/kitchen) require perimeter
          confirmation.

    Returns:
        Sorted list of room IDs where intrusion is detected.
    """
    if sensor_failure_threshold < 1:
        raise ValueError("sensor_failure_threshold must be at least 1.")

    sensor_hit_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    room_types: Dict[int, int] = {}

    for vector in sensor_vectors:
        if len(vector) != 4:
            if strict:
                raise ValueError("Each sensor vector must contain exactly 4 values.")
            continue

        sensor_type, sensor_id, room_id, room_type = vector

        if not (
            isinstance(sensor_type, int)
            and isinstance(sensor_id, int)
            and isinstance(room_id, int)
            and isinstance(room_type, int)
        ):
            if strict:
                raise ValueError(
                    "sensor_type, sensor_id, room_id and room_type must be integers."
                )
            continue

        if (
            sensor_type not in SUPPORTED_SENSOR_TYPES
            or sensor_id < 0
            or room_id < 0
            or room_type not in SUPPORTED_ROOM_TYPES
        ):
            if strict:
                raise ValueError("Sensor values are out of allowed range.")
            continue

        if room_id in room_types and room_types[room_id] != room_type:
            if strict:
                raise ValueError("Conflicting room_type for the same room_id.")
            continue

        room_types[room_id] = room_type
        sensor_hit_counts[(room_id, sensor_type, sensor_id)] += 1

    filtered_active_sensors: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for (room_id, sensor_type, sensor_id), hits in sensor_hit_counts.items():
        if hits >= sensor_failure_threshold:
            continue
        filtered_active_sensors[room_id].add((sensor_type, sensor_id))

    intrusion_rooms: List[int] = []
    for room_id, sensors in filtered_active_sensors.items():
        if not sensors:
            continue

        room_type = room_types.get(room_id)
        if room_type is None:
            continue

        distinct_total = len(sensors)
        has_perimeter = any(
            sensor_type in (SENSOR_TYPE_DOOR_BREAK, SENSOR_TYPE_WINDOW_OPEN)
            for sensor_type, _ in sensors
        )
        ir_count = sum(
            1 for sensor_type, _ in sensors if sensor_type == SENSOR_TYPE_IR_MOTION
        )

        if room_type in (ROOM_TYPE_CORRIDOR, ROOM_TYPE_KITCHEN):
            intrusion = has_perimeter and distinct_total >= 2
        else:
            intrusion = (has_perimeter and distinct_total >= 2) or ir_count >= 3

        if intrusion:
            intrusion_rooms.append(room_id)

    return sorted(intrusion_rooms)


def detect_intrusion(
    sensor_vectors: Iterable[Sequence[int]],
    *,
    strict: bool = False,
    sensor_failure_threshold: int = 4,
) -> bool:
    """Backward-compatible boolean wrapper around `detect_intrusion_rooms`.

    Args:
        sensor_vectors: Iterable of vectors in format
            [sensor_type, sensor_id, room_id, room_type].
        strict: Passed through to `detect_intrusion_rooms`.
        sensor_failure_threshold: Passed through to `detect_intrusion_rooms`.

    Returns:
        True if at least one room is detected as intrusion, otherwise False.
    """
    return bool(
        detect_intrusion_rooms(
            sensor_vectors,
            strict=strict,
            sensor_failure_threshold=sensor_failure_threshold,
        )
    )


if __name__ == "__main__":
    example = [
        [SENSOR_TYPE_IR_MOTION, 5, 101, ROOM_TYPE_BEDROOM],
        [SENSOR_TYPE_WINDOW_OPEN, 7, 101, ROOM_TYPE_BEDROOM],
        [SENSOR_TYPE_IR_MOTION, 9, 103, ROOM_TYPE_CORRIDOR],
    ]
    print("Intrusion rooms:", detect_intrusion_rooms(example))
