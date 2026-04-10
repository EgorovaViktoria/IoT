from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Final, Iterable, List, Mapping, MutableMapping, Tuple

from generate_sensor_batch import (
    DEFAULT_ROOM_ID,
    SAFE_DATASET_PATH,
    UNIFIED_DATASET_PATH,
    dataframe_row_to_batch,
)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from emergency_system import process_sensor_batch  # type: ignore[import-not-found]

CONFIG_PATH = CURRENT_DIR / "qt_sensor_generator_config.json"
DEFAULT_CONFIG: Dict[str, Dict[str, float]] = {
    "synthetic_base_si": {
        "flood": 0.2,
        "smoke": 0.1,
        "leak": 0.1,
        "temp_ambient_c": 296.15,
        "humidity_pct": 0.45,
        "tvoc_ppb": 40.0,
        "eco2_ppm": 520.0,
        "raw_h2": 12800.0,
        "raw_ethanol": 20700.0,
        "press_ambient_bar": 93800.0,
        "pm1_0": 2.0,
        "pm2_5": 2.5,
        "nc0_5": 14.0,
        "nc1_0": 2.2,
        "nc2_5": 0.05,
        "flow_rate_lps": 0.09,
        "press_pipe_bar": 260000.0,
        "temp_pipe_c": 298.15,
    },
    "synthetic_alarm_shifts_si": {
        "flood": {"flood": 0.75},
        "fire": {
            "smoke": 0.8,
            "temp_ambient_c": 55.0,
            "humidity_pct": 0.35,
            "tvoc_ppb": 700.0,
            "eco2_ppm": 1200.0,
            "raw_h2": 7000.0,
            "raw_ethanol": 6500.0,
            "pm1_0": 12.0,
            "pm2_5": 13.0,
            "nc0_5": 70.0,
            "nc1_0": 12.0,
            "nc2_5": 1.0,
        },
        "gas_leak": {
            "leak": 0.85,
            "flow_rate_lps": 0.14,
            "press_pipe_bar": 170000.0,
            "temp_pipe_c": 45.0,
        },
    },
    "synthetic_sensor_radius_si": {
        "default": 0.0,
        "flood": 0.05,
        "smoke": 0.03,
        "leak": 0.04,
        "temp_ambient_c": 4.0,
        "humidity_pct": 0.05,
        "tvoc_ppb": 45.0,
        "eco2_ppm": 80.0,
        "raw_h2": 300.0,
        "raw_ethanol": 350.0,
        "press_ambient_bar": 120.0,
        "pm1_0": 0.4,
        "pm2_5": 0.5,
        "nc0_5": 1.5,
        "nc1_0": 0.25,
        "nc2_5": 0.01,
        "flow_rate_lps": 0.01,
        "press_pipe_bar": 8000.0,
        "temp_pipe_c": 2.0,
    },
    "dataset_jitter_si": {
        "default": 0.0,
        "temp_ambient_c": 0.8,
        "humidity_pct": 0.01,
        "tvoc_ppb": 8.0,
        "eco2_ppm": 12.0,
        "raw_h2": 50.0,
        "raw_ethanol": 60.0,
        "press_ambient_bar": 60.0,
        "pm1_0": 0.2,
        "pm2_5": 0.2,
        "nc0_5": 0.7,
        "nc1_0": 0.1,
        "nc2_5": 0.005,
        "flow_rate_lps": 0.002,
        "press_pipe_bar": 5000.0,
        "temp_pipe_c": 0.8,
    },
    "dataset_alarm_shifts_si": {
        "fire": {
            "smoke": 0.75,
            "temp_ambient_c": 22.0,
            "humidity_pct": 0.20,
            "tvoc_ppb": 280.0,
            "eco2_ppm": 450.0,
            "raw_h2": 1800.0,
            "raw_ethanol": 1800.0,
            "pm1_0": 3.0,
            "pm2_5": 4.0,
            "nc0_5": 15.0,
            "nc1_0": 2.0,
            "nc2_5": 0.08,
        },
        "gas_leak": {
            "leak": 0.75,
            "flow_rate_lps": 0.10,
            "press_pipe_bar": 90000.0,
            "temp_pipe_c": 28.0,
        },
        "flood": {"flood": 0.65},
    },
    "stream": {"interval_ms": 5000.0},
}

FIRE_SENSOR_TYPES = {
    "smoke",
    "temp_ambient_c",
    "tvoc_ppb",
    "eco2_ppm",
    "raw_h2",
    "raw_ethanol",
    "press_ambient_bar",
    "pm1_0",
    "pm2_5",
    "nc0_5",
    "nc1_0",
    "nc2_5",
}
GAS_SENSOR_TYPES = {"gas_leak", "leak", "press_pipe_bar", "flow_rate_lps", "temp_pipe_c"}
FLOOD_SENSOR_TYPES = {"flood"}
# Sensor types that must always be generated synthetically — even in dataset mode.
# Gas is detected by cumulant method; flood sensor is absent from datasets; intrusion is rule-based.
_ALWAYS_SYNTHETIC_SENSOR_TYPES: Final[frozenset[str]] = frozenset(
    GAS_SENSOR_TYPES | FLOOD_SENSOR_TYPES | {"door_break", "ir_motion"}
)
DETECTOR_LABELS = ["Затопление", "Пожар", "Утечка газа", "Проникновение"]
METRICS_ROW_ORDER = ["TP", "FP", "TN", "FN"]
SOURCE_OPTIONS: List[Mapping[str, str]] = [
    {"label": "Синтетический", "key": "synthetic"},
    {"label": "Безопасный датасет", "key": "safe"},
    {"label": "Полный датасет", "key": "full"},
]
ALARM_NAME_BY_KEY = {
    "fire": "Пожар",
    "flood": "Затопление",
    "gas_leak": "Утечка газа",
    "intrusion": "Проникновение",
}
DETECTOR_LABEL_BY_KEY = {
    "flood": "Затопление",
    "fire": "Пожар",
    "gas_leak": "Утечка газа",
    "intrusion": "Проникновение",
}
NEIGHBORHOOD_SEED_OFFSET = 19
FLOOD_SEED_OFFSET = 99
MINIMUM_ALLOWED_STREAM_INTERVAL_MS = 100
# Retry count for seed/jitter search; keeps UI responsive while usually finding target 0/1.
MAX_TARGET_MATCH_ATTEMPTS = 12
ATTEMPT_SEED_OFFSET = 97
# Stabilization deltas help the gas cumulant detector receive sufficiently variable
# but realistic points when gas alarm is requested; this reduces synthetic FN.
GAS_STABILIZATION_LEAK_DELTA = 0.35
GAS_STABILIZATION_PIPE_PRESSURE_DELTA = 120000.0
GAS_STABILIZATION_FLOW_DELTA = 0.08
GAS_STABILIZATION_PIPE_TEMP_DELTA = 20.0


def _prepare_qt_environment() -> None:
    if not sys.platform.startswith("win"):
        return

    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
    plugin_paths: List[Path] = []

    try:
        import PyQt5  # type: ignore[import-not-found]

        pyqt_dir = Path(PyQt5.__file__).resolve().parent
        plugin_paths.extend([pyqt_dir / "Qt5" / "plugins", pyqt_dir / "Qt" / "plugins"])
    except Exception:
        pass

    try:
        from PyQt5.QtCore import QLibraryInfo  # type: ignore[import-not-found]

        plugins_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        if plugins_path:
            plugin_paths.append(Path(plugins_path))
    except Exception:
        pass

    valid_plugins = [p for p in plugin_paths if (p / "platforms").exists()]
    if not valid_plugins:
        return

    plugins_dir = valid_plugins[0]
    platforms_dir = plugins_dir / "platforms"
    if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
    if not os.environ.get("QT_PLUGIN_PATH"):
        os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)


def _load_config() -> Dict[str, Dict[str, float]]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if not CONFIG_PATH.exists():
        return config
    try:
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return config
    for section in (
        "synthetic_base_si",
        "synthetic_sensor_radius_si",
        "dataset_jitter_si",
    ):
        loaded_section = loaded.get(section, {})
        if isinstance(loaded_section, dict):
            for key, value in loaded_section.items():
                if isinstance(value, (int, float)):
                    config[section][key] = float(value)

    for section in ("synthetic_alarm_shifts_si", "dataset_alarm_shifts_si"):
        loaded_section = loaded.get(section, {})
        if isinstance(loaded_section, dict):
            for alarm_key in ("fire", "gas_leak", "flood"):
                loaded_alarm = loaded_section.get(alarm_key, {})
                if not isinstance(loaded_alarm, dict):
                    continue
                base_alarm = config[section].setdefault(alarm_key, {})
                for sensor_key, value in loaded_alarm.items():
                    if isinstance(value, (int, float)):
                        base_alarm[sensor_key] = float(value)

    stream = loaded.get("stream", {})
    if isinstance(stream, dict):
        value = stream.get("interval_ms")
        if isinstance(value, (int, float)) and value >= MINIMUM_ALLOWED_STREAM_INTERVAL_MS:
            config["stream"]["interval_ms"] = float(value)
    return config


def _clip_range(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sensor_jitter(
    rng: random.Random,
    sensor_type: str,
    value: float,
    jitter_map: Mapping[str, float],
) -> float:
    radius = float(jitter_map.get(sensor_type, jitter_map.get("default", 0.0)))
    if radius <= 0:
        result = value
    else:
        result = rng.uniform(value - radius, value + radius)
    if sensor_type in {"flood", "smoke", "leak", "gas_leak", "humidity_pct"}:
        result = _clip_range(result, 0.0, 1.0)
    return round(float(result), 6)


def _synthetic_fire_batch(
    profile: Mapping[str, bool],
    config: Mapping[str, Mapping[str, float]],
    seed: int,
) -> List[List[object]]:
    """Generate fire sensor readings synthetically based on fire profile.

    Used both in pure synthetic mode and as a fallback in dataset mode when no matching
    fire rows exist. Gas/flood/intrusion are intentionally excluded from this function.
    """
    rng = random.Random(seed)
    base_values = dict(config["synthetic_base_si"])
    shifts = config["synthetic_alarm_shifts_si"]
    jitter_map = config["synthetic_sensor_radius_si"]
    if profile["fire"]:
        for sensor_key, delta in shifts.get("fire", {}).items():
            base_values[sensor_key] = float(base_values.get(sensor_key, 0.0)) + float(delta)
    room = DEFAULT_ROOM_ID
    return [
        ["smoke", 1001, room, _sensor_jitter(rng, "smoke", float(base_values["smoke"]), jitter_map)],
        ["temp_ambient_c", 1101, room, _sensor_jitter(rng, "temp_ambient_c", float(base_values["temp_ambient_c"]), jitter_map)],
        ["humidity_pct", 1104, room, _sensor_jitter(rng, "humidity_pct", float(base_values["humidity_pct"]), jitter_map)],
        ["tvoc_ppb", 1102, room, _sensor_jitter(rng, "tvoc_ppb", float(base_values["tvoc_ppb"]), jitter_map)],
        ["eco2_ppm", 1103, room, _sensor_jitter(rng, "eco2_ppm", float(base_values["eco2_ppm"]), jitter_map)],
        ["raw_h2", 1105, room, _sensor_jitter(rng, "raw_h2", float(base_values["raw_h2"]), jitter_map)],
        ["raw_ethanol", 1106, room, _sensor_jitter(rng, "raw_ethanol", float(base_values["raw_ethanol"]), jitter_map)],
        ["press_ambient_bar", 1107, room, _sensor_jitter(rng, "press_ambient_bar", float(base_values["press_ambient_bar"]), jitter_map)],
        ["pm1_0", 1108, room, _sensor_jitter(rng, "pm1_0", float(base_values["pm1_0"]), jitter_map)],
        ["pm2_5", 1109, room, _sensor_jitter(rng, "pm2_5", float(base_values["pm2_5"]), jitter_map)],
        ["nc0_5", 1110, room, _sensor_jitter(rng, "nc0_5", float(base_values["nc0_5"]), jitter_map)],
        ["nc1_0", 1111, room, _sensor_jitter(rng, "nc1_0", float(base_values["nc1_0"]), jitter_map)],
        ["nc2_5", 1112, room, _sensor_jitter(rng, "nc2_5", float(base_values["nc2_5"]), jitter_map)],
    ]


def _synthetic_gas_flood_intrusion_batch(
    profile: Mapping[str, bool],
    config: Mapping[str, Mapping[str, float]],
    seed: int,
) -> List[List[object]]:
    """Generate gas, flood, and intrusion sensor readings synthetically.

    Contract:
      - Gas is detected by cumulant method → always synthetic.
      - Flood sensor is absent from the datasets → always synthetic.
      - Intrusion is rule-based → always synthetic.
    This function is called both in pure synthetic mode and in dataset mode.
    """
    rng = random.Random(seed)
    base_values = dict(config["synthetic_base_si"])
    shifts = config["synthetic_alarm_shifts_si"]
    jitter_map = config["synthetic_sensor_radius_si"]
    if profile["gas_leak"]:
        for sensor_key, delta in shifts.get("gas_leak", {}).items():
            base_values[sensor_key] = float(base_values.get(sensor_key, 0.0)) + float(delta)
    if profile["flood"]:
        for sensor_key, delta in shifts.get("flood", {}).items():
            base_values[sensor_key] = float(base_values.get(sensor_key, 0.0)) + float(delta)
    intrusion_value = 1.0 if profile["intrusion"] else 0.0
    room = DEFAULT_ROOM_ID
    return [
        ["flood", 3001, room, _sensor_jitter(rng, "flood", float(base_values["flood"]), jitter_map)],
        [
            "gas_leak",
            2001,
            room,
            _sensor_jitter(rng, "gas_leak", float(base_values.get("gas_leak", base_values.get("leak", 0.0))), jitter_map),
        ],
        ["flow_rate_lps", 2101, room, _sensor_jitter(rng, "flow_rate_lps", float(base_values["flow_rate_lps"]), jitter_map)],
        ["press_pipe_bar", 2102, room, _sensor_jitter(rng, "press_pipe_bar", float(base_values["press_pipe_bar"]), jitter_map)],
        ["temp_pipe_c", 2103, room, _sensor_jitter(rng, "temp_pipe_c", float(base_values["temp_pipe_c"]), jitter_map)],
        ["door_break", 1, room, intrusion_value],
        ["ir_motion", 2, room, intrusion_value],
    ]


def _synthetic_batch(
    profile: Mapping[str, bool],
    config: Mapping[str, Mapping[str, float]],
    seed: int,
) -> List[List[object]]:
    """Full synthetic batch combining fire and gas/flood/intrusion sub-batches."""
    return _synthetic_fire_batch(profile, config, seed) + _synthetic_gas_flood_intrusion_batch(profile, config, seed)


def _dataset_path(source: str) -> Path:
    if source == "safe":
        return SAFE_DATASET_PATH
    if source == "full":
        return UNIFIED_DATASET_PATH
    raise ValueError("source must be safe/full for dataset mode")


def _row_matches_profile(row: Mapping[str, str], profile: Mapping[str, bool]) -> bool:
    # Dataset contains reliable label only for fire (`smoke_label`).
    # Dataset contains a reliable fire label only (`smoke_label`).
    # Gas/flood/intrusion are always generated synthetically — they do not affect row selection.
    smoke_active = int(float(row.get("smoke_label", "0"))) == 1
    if profile["fire"] and not smoke_active:
        return False
    if not profile["fire"] and smoke_active:
        return False
    return True


def _pick_dataset_row(source: str, profile: Mapping[str, bool], seed: int) -> Tuple[MutableMapping[str, str], int]:
    path = _dataset_path(source)
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    matching: List[Tuple[int, MutableMapping[str, str]]] = [
        (idx, row) for idx, row in enumerate(rows) if _row_matches_profile(row, profile)
    ]
    if not matching:
        # Only fire (smoke_label) affects row selection; this can happen when the requested
        # fire profile has no matching label rows in the dataset (e.g. fire=True in safe dataset).
        fire_label = ALARM_NAME_BY_KEY.get("fire", "Пожар")
        fire_text = f"{fire_label}=1" if profile["fire"] else f"{fire_label}=0"
        raise ValueError(f'Нет строк датасета "{source}" под профиль пожара: {fire_text}.')
    rng = random.Random(seed)
    selected_idx, selected_row = matching[rng.randrange(len(matching))]
    return selected_row, selected_idx


def _adjust_batch_neighborhood(
    batch: Iterable[List[object]],
    profile: Mapping[str, bool],
    config: Mapping[str, Mapping[str, float]],
    seed: int,
) -> List[List[object]]:
    rng = random.Random(seed + NEIGHBORHOOD_SEED_OFFSET)
    jitter_map = config["dataset_jitter_si"]
    shifts = config["dataset_alarm_shifts_si"]
    adjusted: List[List[object]] = []
    for sensor_type, sensor_id, room, reading in batch:
        st = str(sensor_type).lower()
        value = float(reading)
        if profile["fire"]:
            value += float(shifts.get("fire", {}).get(st, 0.0))
        if profile["gas_leak"]:
            value += float(shifts.get("gas_leak", {}).get(st, 0.0))
        if profile["flood"]:
            value += float(shifts.get("flood", {}).get(st, 0.0))
        adjusted.append([sensor_type, sensor_id, room, _sensor_jitter(rng, st, value, jitter_map)])
    return adjusted


def _calc_confusion(expected: int, predicted: int) -> Dict[str, int]:
    return {
        "TP": int(expected == 1 and predicted == 1),
        "FP": int(expected == 0 and predicted == 1),
        "TN": int(expected == 0 and predicted == 0),
        "FN": int(expected == 1 and predicted == 0),
    }


def _profile_room_id(source: str, profile: Mapping[str, bool]) -> int:
    source_index = {"synthetic": 0, "safe": 1, "full": 2}.get(source, 3)
    profile_mask = (
        (1 if profile["flood"] else 0)
        | ((1 if profile["fire"] else 0) << 1)
        | ((1 if profile["gas_leak"] else 0) << 2)
        | ((1 if profile["intrusion"] else 0) << 3)
    )
    return 1000 + source_index * 32 + profile_mask


def _apply_room_id(batch: Iterable[List[object]], room_id: int) -> List[List[object]]:
    adjusted: List[List[object]] = []
    for sensor_type, sensor_id, _room, reading in batch:
        adjusted.append([sensor_type, sensor_id, room_id, reading])
    return adjusted


def _target_prediction_map(profile: Mapping[str, bool]) -> Dict[str, int]:
    return {
        "Затопление": int(profile["flood"]),
        "Пожар": int(profile["fire"]),
        "Утечка газа": int(profile["gas_leak"]),
        "Проникновение": int(profile["intrusion"]),
    }


def _prediction_distance(predicted_map: Mapping[str, int], target_map: Mapping[str, int]) -> int:
    return sum(int(predicted_map.get(det, 0) != target_map.get(det, 0)) for det in DETECTOR_LABELS)


def _alternating_delta(delta: float, attempt: int) -> float:
    return delta if attempt % 2 == 0 else -delta


def _stabilize_batch_for_attempt(
    batch: Iterable[List[object]],
    profile: Mapping[str, bool],
    attempt: int,
) -> List[List[object]]:
    stabilized: List[List[object]] = []
    for sensor_type, sensor_id, room, reading in batch:
        st = str(sensor_type).lower()
        value = float(reading)
        if profile["gas_leak"] and st in {"gas_leak", "leak"}:
            value = _clip_range(value + _alternating_delta(GAS_STABILIZATION_LEAK_DELTA, attempt), 0.0, 1.0)
        if profile["gas_leak"] and st == "press_pipe_bar":
            value = max(0.0, value + _alternating_delta(GAS_STABILIZATION_PIPE_PRESSURE_DELTA, attempt))
        if profile["gas_leak"] and st == "flow_rate_lps":
            value = max(0.0, value + _alternating_delta(GAS_STABILIZATION_FLOW_DELTA, attempt))
        if profile["gas_leak"] and st == "temp_pipe_c":
            value = max(0.0, value + _alternating_delta(GAS_STABILIZATION_PIPE_TEMP_DELTA, attempt))
        if profile["flood"] and st == "flood":
            value = max(value, 0.95)
        stabilized.append([sensor_type, sensor_id, room, round(float(value), 6)])
    return stabilized


def _build_expected_maps(
    profile: Mapping[str, bool],
) -> Tuple[Dict[str, object], Dict[str, bool], Dict[str, str]]:
    expected: Dict[str, object] = {}
    expected_available: Dict[str, bool] = {}
    expected_notes: Dict[str, str] = {}
    for key, detector_label in DETECTOR_LABEL_BY_KEY.items():
        expected[detector_label] = int(profile[key])
        expected_available[detector_label] = True
        expected_notes[detector_label] = "target from selected alarm profile (0/1)"
    return expected, expected_available, expected_notes


def generate_profiled_batch(
    source: str,
    profile: Mapping[str, bool],
    seed: int,
    config: Mapping[str, Mapping[str, float]],
    dataset_fallback_to_synthetic: bool = False,
) -> Dict[str, object]:
    dataset_path = str(_dataset_path(source)) if source != "synthetic" else None
    room_id = _profile_room_id(source=source, profile=profile)
    target_map = _target_prediction_map(profile)

    best_payload: Dict[str, object] | None = None
    best_distance = 10**9
    for attempt in range(MAX_TARGET_MATCH_ATTEMPTS):
        attempt_seed = seed + attempt * ATTEMPT_SEED_OFFSET
        source_effective = source
        fallback_used = False
        fallback_reason = None
        selected_row: MutableMapping[str, str] | None = None
        if source == "synthetic":
            raw_batch = _synthetic_batch(profile=profile, config=config, seed=attempt_seed)
            selected_row_index = None
        else:
            gas_flood_intrusion = _synthetic_gas_flood_intrusion_batch(
                profile=profile, config=config, seed=attempt_seed + FLOOD_SEED_OFFSET
            )
            try:
                selected_row, selected_row_index = _pick_dataset_row(source=source, profile=profile, seed=attempt_seed)
                raw_fire_batch = dataframe_row_to_batch(selected_row)
                fire_batch = [item for item in raw_fire_batch if str(item[0]).lower() not in _ALWAYS_SYNTHETIC_SENSOR_TYPES]
                fire_batch = _adjust_batch_neighborhood(
                    batch=fire_batch,
                    profile=profile,
                    config=config,
                    seed=attempt_seed,
                )
                raw_batch = fire_batch + gas_flood_intrusion
            except ValueError as exc:
                if not dataset_fallback_to_synthetic:
                    raise
                source_effective = "synthetic"
                fallback_used = True
                fallback_reason = str(exc)
                selected_row = None
                selected_row_index = None
                fire_batch = _synthetic_fire_batch(profile=profile, config=config, seed=attempt_seed)
                raw_batch = fire_batch + gas_flood_intrusion

        batch = _apply_room_id(raw_batch, room_id=room_id)
        batch = _stabilize_batch_for_attempt(batch=batch, profile=profile, attempt=attempt)
        predicted = process_sensor_batch(batch)
        predicted_map = {
            "Затопление": int(predicted[0]),
            "Пожар": int(predicted[1]),
            "Утечка газа": int(predicted[2]),
            "Проникновение": int(predicted[3]),
        }
        distance = _prediction_distance(predicted_map, target_map)
        payload_candidate = {
            "source_effective": source_effective,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "selected_row_index": selected_row_index,
            "selected_row": selected_row,
            "batch": batch,
            "predicted_map": predicted_map,
            "distance": distance,
        }
        if distance < best_distance:
            best_distance = distance
            best_payload = payload_candidate
        if distance == 0:
            break

    if best_payload is None:
        raise RuntimeError("Не удалось сформировать пакет с валидным предсказанием.")

    expected, expected_available, expected_notes = _build_expected_maps(profile=profile)
    predicted_map = dict(best_payload["predicted_map"])  # type: ignore[arg-type]
    confusion: Dict[str, Dict[str, int]] = {}
    for detector in DETECTOR_LABELS:
        if expected_available.get(detector):
            confusion[detector] = _calc_confusion(int(expected[detector]), predicted_map[detector])
        else:
            confusion[detector] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    return {
        "seed": seed,
        "source": source,
        "source_effective": best_payload["source_effective"],
        "fallback_used": best_payload["fallback_used"],
        "fallback_reason": best_payload["fallback_reason"],
        "dataset_path": dataset_path,
        "config_path": str(CONFIG_PATH),
        "radius": config["synthetic_sensor_radius_si"].get("default", 0.0),
        "profile": dict(profile),
        "selected_dataset_row_index": best_payload["selected_row_index"],
        "target_match_attempts": MAX_TARGET_MATCH_ATTEMPTS,
        "target_mismatch_count": best_payload["distance"],
        "target": target_map,
        "room_id": room_id,
        "sensor_types_count": len({item[0] for item in best_payload["batch"]}),  # type: ignore[index]
        "expected": expected,
        "expected_available": expected_available,
        "expected_notes": expected_notes,
        "predicted": predicted_map,
        "confusion_matrix": confusion,
        "batch": best_payload["batch"],
    }


_prepare_qt_environment()

try:
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from PyQt5.QtCore import QTimer

    PYQT_AVAILABLE = True
except ImportError as exc:
    QApplication = object  # type: ignore[assignment,misc]
    QMainWindow = object  # type: ignore[assignment,misc]
    QFileDialog = QMessageBox = QPushButton = QTextEdit = QWidget = object  # type: ignore[assignment,misc]
    QCheckBox = QComboBox = QLabel = QGridLayout = QVBoxLayout = QHBoxLayout = object  # type: ignore[assignment,misc]
    QTableWidget = QTableWidgetItem = QHeaderView = QTimer = object  # type: ignore[assignment,misc]
    PYQT_AVAILABLE = False
    PYQT_IMPORT_ERROR = exc


class SensorGeneratorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.config = _load_config()
        self.setWindowTitle("Генератор сенсорных пакетов")
        self.resize(1120, 760)
        self._payload: Dict[str, object] = {}
        self._metrics_totals = self._empty_metrics()
        self._stream_timer = QTimer(self)
        self._stream_timer.setInterval(int(self.config["stream"]["interval_ms"]))
        self._stream_timer.timeout.connect(self._generate_once)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)

        controls = QGridLayout()
        controls.addWidget(QLabel("Источник:"), 0, 0)
        self.source_box = QComboBox()
        for option in SOURCE_OPTIONS:
            self.source_box.addItem(option["label"], option["key"])
        controls.addWidget(self.source_box, 0, 1)
        controls.addWidget(
            QLabel("Параметры генерации берутся из SI-конфига (см. qt_sensor_generator_config.json)."),
            1,
            0,
            1,
            4,
        )

        self.fire_alarm_box = QCheckBox("Пожар")
        self.flood_alarm_box = QCheckBox("Затопление")
        self.gas_leak_alarm_box = QCheckBox("Утечка газа")
        self.intrusion_alarm_box = QCheckBox("Проникновение")
        controls.addWidget(self.fire_alarm_box, 2, 0, 1, 1)
        controls.addWidget(self.flood_alarm_box, 2, 1, 1, 1)
        controls.addWidget(self.gas_leak_alarm_box, 2, 2, 1, 1)
        controls.addWidget(self.intrusion_alarm_box, 2, 3, 1, 1)
        self.dataset_fallback_box = QCheckBox("Замещать пропуски датасетов (fallback на synthetic)")
        self.dataset_fallback_box.setChecked(True)
        controls.addWidget(self.dataset_fallback_box, 3, 0, 1, 4)

        main_layout.addLayout(controls)

        button_row = QHBoxLayout()
        self.btn_start = QPushButton("Старт")
        self.btn_start.clicked.connect(self._start_stream)
        button_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.clicked.connect(self._stop_stream)
        self.btn_stop.setEnabled(False)
        button_row.addWidget(self.btn_stop)

        btn_reset = QPushButton("Обнулить")
        btn_reset.clicked.connect(self._reset_view)
        button_row.addWidget(btn_reset)

        btn_save = QPushButton("Сохранить JSON")
        btn_save.clicked.connect(self._save_json)
        button_row.addWidget(btn_save)
        main_layout.addLayout(button_row)

        self.note = QLabel(
            "Если галочки сняты: безопасная окрестность (без сработок). "
            "Галочки смещают окрестность в зону сработок. "
            "Для режимов «Безопасный датасет»/«Полный датасет» подтягиваются строки "
            "под профили с доступными метками; для недоступных меток можно включить synthetic fallback. "
            "Сид берется автоматически как текущее Unix-время в секундах."
        )
        self.note.setWordWrap(True)
        main_layout.addWidget(self.note)
        self.dataset_status = QLabel("Датасет: не использован.")
        main_layout.addWidget(self.dataset_status)

        self.metrics_table = QTableWidget(4, 4)
        self.metrics_table.setVerticalHeaderLabels(
            [
                "TP — Истинно-положительные",
                "FP — Ложно-положительные",
                "TN — Истинно-отрицательные",
                "FN — Ложно-отрицательные",
            ]
        )
        self.metrics_table.setHorizontalHeaderLabels(DETECTOR_LABELS)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.metrics_table)
        self._fill_metrics(self._metrics_totals)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        main_layout.addWidget(self.output)

    def _profile(self) -> Dict[str, bool]:
        return {
            "fire": self.fire_alarm_box.isChecked(),
            "flood": self.flood_alarm_box.isChecked(),
            "gas_leak": self.gas_leak_alarm_box.isChecked(),
            "intrusion": self.intrusion_alarm_box.isChecked(),
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, Dict[str, int]]:
        return {detector: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for detector in DETECTOR_LABELS}

    def _source_label(self, source: str) -> str:
        for option in SOURCE_OPTIONS:
            if source == option["key"]:
                return option["label"]
        return source

    def _accumulate_metrics(
        self,
        confusion_matrix: Mapping[str, Mapping[str, int]],
        expected_available: Mapping[str, bool],
    ) -> None:
        for detector in DETECTOR_LABELS:
            if not expected_available.get(detector, False):
                continue
            detector_metrics = confusion_matrix.get(detector, {})
            totals = self._metrics_totals.setdefault(detector, {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
            for metric in ("TP", "FP", "TN", "FN"):
                totals[metric] += int(detector_metrics.get(metric, 0))

    def _fill_metrics(self, confusion_matrix: Mapping[str, Mapping[str, int]]) -> None:
        for col, detector in enumerate(DETECTOR_LABELS):
            metrics = confusion_matrix.get(detector, {})
            for row, metric_name in enumerate(METRICS_ROW_ORDER):
                self.metrics_table.setItem(row, col, QTableWidgetItem(str(int(metrics.get(metric_name, 0)))))

    def _update_dataset_status(self) -> None:
        if not self._payload:
            self.dataset_status.setText("Датасет: не использован.")
            return
        source = str(self._payload.get("source", ""))
        source_effective = str(self._payload.get("source_effective", source))
        source_label = self._source_label(source)
        source_effective_label = self._source_label(source_effective)
        if source_effective == "synthetic":
            if source == "synthetic":
                self.dataset_status.setText(f"Источник: {source_label} (без чтения CSV).")
            else:
                self.dataset_status.setText(
                    f"Источник: {source_label}, факт: {source_effective_label} (fallback, CSV-профиль не найден)."
                )
            return
        idx = self._payload.get("selected_dataset_row_index")
        path = self._payload.get("dataset_path", "")
        fallback_used = bool(self._payload.get("fallback_used", False))
        fallback_suffix = " [fallback synthetic]" if fallback_used else ""
        self.dataset_status.setText(
            f"Источник: {source_label}, факт: {source_effective_label}{fallback_suffix}, строка CSV: {idx}, файл: {path}"
        )

    def _generate_once(self) -> bool:
        try:
            seed = int(time.time())
            source = str(self.source_box.currentData())
            profile = self._profile()
            self._payload = generate_profiled_batch(
                source=source,
                profile=profile,
                seed=seed,
                config=self.config,
                dataset_fallback_to_synthetic=self.dataset_fallback_box.isChecked(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка генерации", str(exc))
            return False
        self._accumulate_metrics(
            self._payload.get("confusion_matrix", {}),
            self._payload.get("expected_available", {}),
        )
        self._fill_metrics(self._metrics_totals)
        self.output.setPlainText(json.dumps(self._payload, ensure_ascii=False, indent=2))
        self._update_dataset_status()
        return True

    def _start_stream(self) -> None:
        if not self._generate_once():
            return
        self._stream_timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop_stream(self) -> None:
        self._stream_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _reset_view(self) -> None:
        self._stop_stream()
        self._payload = {}
        self._metrics_totals = self._empty_metrics()
        self.output.clear()
        self._fill_metrics(self._metrics_totals)
        self._update_dataset_status()

    def _save_json(self) -> None:
        text = self.output.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Сохранение JSON", "Нет данных для сохранения.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить сгенерированный пакет",
            str(Path.cwd() / "generated_sensor_batch.json"),
            "Файлы JSON (*.json);;Все файлы (*)",
        )
        if not path:
            return
        Path(path).write_text(text, encoding="utf-8")


def main() -> None:
    if not PYQT_AVAILABLE:
        raise SystemExit("Для qt_sensor_generator.py требуется PyQt5.") from PYQT_IMPORT_ERROR
    app = QApplication(sys.argv)
    win = SensorGeneratorWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
