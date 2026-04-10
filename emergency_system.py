from __future__ import annotations

import json
import sys
import threading
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

CURRENT_DIR = Path(__file__).resolve().parent
UTILITES_DIR = CURRENT_DIR / "utilites"
MODELS_DIR = CURRENT_DIR / "models"
if str(UTILITES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITES_DIR))

from intrusion_detection import (  # type: ignore[import-not-found]
    ROOM_TYPE_LIVING_ROOM,
    detect_intrusion,
)

try:
    from cumulants import cumulants_2_to_6 as cumulants_2_to_6_external  # type: ignore[import-not-found]
except Exception:
    cumulants_2_to_6_external = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore[assignment]

try:
    import joblib
except Exception:
    joblib = None  # type: ignore[assignment]

try:
    from sklearn.exceptions import InconsistentVersionWarning  # type: ignore[import-not-found]
except Exception:
    InconsistentVersionWarning = None  # type: ignore[assignment]


PacketLike = Union[Sequence[object], Mapping[str, object]]
FLOOD_SENSOR_CODE = 300
FIRE_SENSOR_CODE = 100
LEAK_SENSOR_CODE = 200
ROOM_ID_START = 1000
WINDOW_SIZE = 5
GAS_HISTORY_SIZE = 32
# Low minimum to allow short stream scenarios in tests/GUI while still returning 0 for truly insufficient history.
# With 2 points, higher cumulants degenerate near zero and detection mostly relies on K2 + mean-level thresholds.
GAS_MIN_HISTORY = 2
GAS_K3_WEIGHT = 0.2
GAS_K4_WEIGHT = 0.05
GAS_MEAN_THRESHOLD = 0.55
GAS_SCORE_THRESHOLD = 0.01

FIRE_SENSOR_TYPES = {
    "fire",
    "smoke",
    "temperature",
    "temp",
    "co2",
    "tvoc",
    "eco2",
    "temp_ambient_c",
    "humidity_pct",
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
GAS_SENSOR_TYPES = {
    "gas_leak",
    "leak",
    "water_leak",
    "pipe_pressure",
    "flow",
    "flow_rate",
    "press_pipe_bar",
    "flow_rate_lps",
    "temp_pipe_c",
}
FLOOD_SENSOR_TYPES = {
    "flood",
    "water_level",
    "waterlevel",
    "rain",
    "drain",
}

FIRE_FEATURE_MAP = {
    "temp_ambient_c": "Temperature[C]",
    "humidity_pct": "Humidity[%]",
    "tvoc_ppb": "TVOC[ppb]",
    "eco2_ppm": "eCO2[ppm]",
    "raw_h2": "Raw H2",
    "raw_ethanol": "Raw Ethanol",
    "press_ambient_bar": "Pressure[hPa]",
    "pm1_0": "PM1.0",
    "pm2_5": "PM2.5",
    "nc0_5": "NC0.5",
    "nc1_0": "NC1.0",
    "nc2_5": "NC2.5",
}
FLOOD_FEATURE_MAP = {
    "flood": "flood_level_norm",
    "water_level": "flood_level_norm",
    "waterlevel": "flood_level_norm",
    "rain": "flood_level_norm",
    "drain": "flood_level_norm",
}
FIRE_RAW_FEATURE_COLUMNS = list(FIRE_FEATURE_MAP.values())
FLOOD_RAW_FEATURE_COLUMNS = ["flood_level_norm"]

_FIRE_HISTORY: MutableMapping[str, Deque[Dict[str, float]]] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
_FLOOD_HISTORY: MutableMapping[str, Deque[Dict[str, float]]] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
_GAS_SIGNAL_HISTORY: MutableMapping[str, Deque[float]] = defaultdict(lambda: deque(maxlen=GAS_HISTORY_SIZE))
_FIRE_HISTORY_LOCK = threading.Lock()
_FLOOD_HISTORY_LOCK = threading.Lock()
_GAS_SIGNAL_HISTORY_LOCK = threading.Lock()
_WARNED_MESSAGES: Set[str] = set()
_SKLEARN_VERSION_WARNING_EMITTED = False


def _to_model_temperature_c(value_k: float) -> float:
    return float(value_k) - 273.15


def _to_model_humidity_pct(value_fraction: float) -> float:
    return float(value_fraction) * 100.0


def _to_model_pressure_hpa(value_pa: float) -> float:
    return float(value_pa) / 100.0


def _to_model_pressure_bar(value_pa: float) -> float:
    return float(value_pa) / 100000.0


def _to_model_flow_lps(value_m3s: float) -> float:
    return float(value_m3s) * 1000.0


SI_TO_MODEL_VALUE_TRANSFORMS = {
    "temp_ambient_c": _to_model_temperature_c,
    "temp_pipe_c": _to_model_temperature_c,
    "humidity_pct": _to_model_humidity_pct,
    "press_ambient_bar": _to_model_pressure_hpa,
    "press_pipe_bar": _to_model_pressure_bar,
    "flow_rate_lps": _to_model_flow_lps,
}

# SI-threshold fallback rules (fire/gas leak). flood remains normalized flood_level_norm (0..1).
FIRE_SI_SENSOR_THRESHOLDS = {
    "fire": 0.7,
    "smoke": 0.7,
    "temp_ambient_c": 333.15,  # K
    "tvoc_ppb": 300.0,
    "eco2_ppm": 1200.0,
    "raw_h2": 14000.0,
    "raw_ethanol": 22000.0,
    "pm1_0": 8.0,
    "pm2_5": 10.0,
    "nc0_5": 40.0,
    "nc1_0": 6.0,
    "nc2_5": 0.2,
}
GAS_LEAK_SI_SENSOR_THRESHOLDS = {
    "gas_leak": 0.7,
    "leak": 0.7,
    "water_leak": 0.7,
    "press_pipe_bar": 350000.0,  # Pa
    "flow_rate_lps": 0.175,  # m^3/s
    "temp_pipe_c": 329.15,  # K
    "tvoc_ppb": 300.0,
    "eco2_ppm": 1500.0,
    "raw_h2": 15000.0,
    "raw_ethanol": 23000.0,
}


@dataclass(frozen=True)
class SensorPacket:
    sensor_type: Union[int, str]
    sensor_id: int
    room: Union[int, str]
    reading: float


def _normalize_packets(raw_packets: Iterable[PacketLike]) -> List[SensorPacket]:
    packets: List[SensorPacket] = []
    for item in raw_packets:
        if isinstance(item, Mapping):
            required = ("sensor_type", "sensor_id", "room", "reading")
            missing = [k for k in required if k not in item]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            sensor_type = item["sensor_type"]
            sensor_id = int(item["sensor_id"])
            room = item["room"]
            reading = float(item["reading"])
        else:
            if len(item) != 4:
                raise ValueError("Each sensor vector must contain exactly 4 values.")
            sensor_type, sensor_id, room, reading = item
            sensor_id = int(sensor_id)
            reading = float(reading)

        if isinstance(room, str) and room.isdigit():
            room = int(room)

        packets.append(
            SensorPacket(
                sensor_type=sensor_type,
                sensor_id=sensor_id,
                room=room,
                reading=reading,
            )
        )
    return packets


def _sensor_key(sensor_type: Union[int, str]) -> str:
    return str(sensor_type).strip().lower()


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def reset_state() -> None:
    with _FIRE_HISTORY_LOCK:
        _FIRE_HISTORY.clear()
    with _FLOOD_HISTORY_LOCK:
        _FLOOD_HISTORY.clear()
    with _GAS_SIGNAL_HISTORY_LOCK:
        _GAS_SIGNAL_HISTORY.clear()


def _joblib_load_with_version_warning(model_path: Path):
    global _SKLEARN_VERSION_WARNING_EMITTED
    if InconsistentVersionWarning is None:
        return joblib.load(model_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        pkg = joblib.load(model_path)

    inconsistent_warnings = [
        warning for warning in caught if issubclass(warning.category, InconsistentVersionWarning)
    ]
    for warning in caught:
        if not issubclass(warning.category, InconsistentVersionWarning):
            warnings.warn_explicit(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
            )
    if inconsistent_warnings and not _SKLEARN_VERSION_WARNING_EMITTED:
        _warn_once(
            "Обнаружена несовместимость версии sklearn при загрузке .pkl модели. "
            "Рекомендуется установить зависимости из Code/requirements.txt "
            "(scikit-learn==1.8.0). Предупреждение показано один раз."
        )
        _SKLEARN_VERSION_WARNING_EMITTED = True
    return pkg


@lru_cache(maxsize=4)
def _load_model_package(model_filename: str) -> Optional[Mapping[str, object]]:
    if joblib is None:
        _warn_once("joblib не установлен, используется fallback на пороговые правила.")
        return None
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        _warn_once(f"Файл модели не найден: {model_path}. Используется fallback на пороговые правила.")
        return None
    try:
        pkg = _joblib_load_with_version_warning(model_path)
    except Exception as exc:
        _warn_once(f"Не удалось загрузить модель {model_path}: {exc}. Используется fallback.")
        return None

    if not isinstance(pkg, Mapping):
        _warn_once(f"Некорректный формат пакета модели {model_path}. Используется fallback.")
        return None
    if "model" not in pkg or "features" not in pkg or "threshold" not in pkg:
        _warn_once(f"В пакете модели {model_path} отсутствуют model/features/threshold. Используется fallback.")
        return None
    return pkg


def _collect_room_feature_updates(
    packets: List[SensorPacket],
    feature_map: Mapping[str, str],
) -> Dict[str, Dict[str, float]]:
    updates: Dict[str, Dict[str, float]] = {}
    for packet in packets:
        sensor_key = _sensor_key(packet.sensor_type)
        if sensor_key not in feature_map:
            continue
        room_key = str(packet.room)
        feature_name = feature_map[sensor_key]
        transform = SI_TO_MODEL_VALUE_TRANSFORMS.get(sensor_key)
        feature_value = transform(packet.reading) if transform else packet.reading
        room_row = updates.setdefault(room_key, {})
        current_value = room_row.get(feature_name)
        if current_value is None:
            room_row[feature_name] = feature_value
        else:
            room_row[feature_name] = max(current_value, feature_value)
    return updates


def _room_signal_updates_for_gas_leak(packets: List[SensorPacket]) -> Dict[str, float]:
    updates: Dict[str, float] = {}
    for packet in packets:
        sensor_key = _sensor_key(packet.sensor_type)
        if sensor_key not in GAS_SENSOR_TYPES:
            continue
        if sensor_key in {"gas_leak", "leak", "water_leak"}:
            score = float(packet.reading)
        elif sensor_key == "press_pipe_bar":
            score = float(packet.reading) / 500000.0
        elif sensor_key == "flow_rate_lps":
            score = float(packet.reading) / 0.3
        elif sensor_key == "temp_pipe_c":
            score = _to_model_temperature_c(float(packet.reading)) / 100.0
        else:
            score = float(packet.reading)
        room_key = str(packet.room)
        score = max(0.0, min(1.5, score))
        updates[room_key] = max(score, updates.get(room_key, 0.0))
    return updates


def _cumulants_2_to_6_fallback(signal: Sequence[float]) -> Dict[int, float]:
    if not signal:
        return {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
    mean = sum(signal) / len(signal)

    def _mu(power: int) -> float:
        return sum((x - mean) ** power for x in signal) / len(signal)

    mu2 = _mu(2)
    mu3 = _mu(3)
    mu4 = _mu(4)
    mu5 = _mu(5)
    mu6 = _mu(6)
    # Exact cumulant coefficients from central moments:
    # K4 = mu4 - 3*mu2^2; K5 = mu5 - 10*mu3*mu2; K6 = mu6 - 15*mu4*mu2 - 10*mu3^2 + 30*mu2^3.
    return {
        2: mu2,
        3: mu3,
        4: mu4 - 3.0 * (mu2**2),
        5: mu5 - 10.0 * mu3 * mu2,
        6: mu6 - 15.0 * mu4 * mu2 - 10.0 * (mu3**2) + 30.0 * (mu2**3),
    }


def _cumulants_2_to_6(signal: Sequence[float]) -> Dict[int, float]:
    if cumulants_2_to_6_external is not None and np is not None:
        try:
            return cumulants_2_to_6_external(np.array(signal, dtype=float))
        except Exception:
            pass
    return _cumulants_2_to_6_fallback(signal)


def _rolling_features_dataframe(raw_df: "pd.DataFrame") -> "pd.DataFrame":
    features_agg = raw_df.rolling(WINDOW_SIZE, min_periods=1).agg(["mean", "std", "max", "min"])
    features_agg.columns = [
        "_".join(column).replace(" ", "_").replace("(", "").replace(")", "").replace("°", "").strip()
        for column in features_agg.columns
    ]
    return features_agg.fillna(0.0)


def _predict_from_history_rows(
    history_rows: Sequence[Mapping[str, float]],
    raw_feature_columns: Sequence[str],
    pkg: Mapping[str, object],
) -> Optional[int]:
    if pd is None:
        _warn_once("pandas не установлен, используется fallback на пороговые правила.")
        return None

    model = pkg.get("model")
    model_features = pkg.get("features")
    threshold = pkg.get("threshold")
    if model is None or not isinstance(model_features, Sequence) or threshold is None:
        return None
    if not hasattr(model, "predict_proba"):
        _warn_once("Модель не поддерживает predict_proba, используется fallback на пороговые правила.")
        return None

    try:
        raw_df = pd.DataFrame(history_rows)
        for col in raw_feature_columns:
            if col not in raw_df.columns:
                raw_df[col] = 0.0
        raw_df = raw_df[list(raw_feature_columns)].fillna(0.0)
        aggregated_df = _rolling_features_dataframe(raw_df)

        for feature_name in model_features:
            if feature_name not in aggregated_df.columns:
                aggregated_df[feature_name] = 0.0
        model_input = aggregated_df[list(model_features)]

        probabilities = model.predict_proba(model_input)[:, 1]
        if len(probabilities) == 0:
            return 0
        return int(float(probabilities[-1]) >= float(threshold))
    except Exception as exc:
        _warn_once(f"Ошибка ML-инференса: {exc}. Используется fallback на пороговые правила.")
        return None


def detect_flood_threshold(packets: List[SensorPacket], threshold: float = 0.9) -> int:
    readings = [
        p.reading
        for p in packets
        if _sensor_key(p.sensor_type) in FLOOD_SENSOR_TYPES or p.sensor_type == FLOOD_SENSOR_CODE
    ]
    if not readings:
        return 0

    max_reading = max(readings)
    return int(max_reading >= threshold)


def detect_fire_threshold(packets: List[SensorPacket]) -> int:
    for packet in packets:
        sensor_key = _sensor_key(packet.sensor_type)
        threshold = FIRE_SI_SENSOR_THRESHOLDS.get(sensor_key)
        if threshold is None and packet.sensor_type == FIRE_SENSOR_CODE:
            threshold = FIRE_SI_SENSOR_THRESHOLDS["fire"]
        if threshold is None:
            continue
        if packet.reading >= threshold:
            return 1
    return 0


def detect_gas_leak_threshold(packets: List[SensorPacket]) -> int:
    for packet in packets:
        sensor_key = _sensor_key(packet.sensor_type)
        threshold = GAS_LEAK_SI_SENSOR_THRESHOLDS.get(sensor_key)
        if threshold is None and packet.sensor_type == LEAK_SENSOR_CODE:
            threshold = GAS_LEAK_SI_SENSOR_THRESHOLDS["gas_leak"]
        if threshold is None:
            continue
        if packet.reading >= threshold:
            return 1
    return 0


def _detect_with_model_or_fallback(
    packets: List[SensorPacket],
    model_filename: str,
    feature_map: Mapping[str, str],
    raw_feature_columns: Sequence[str],
    history: MutableMapping[str, Deque[Dict[str, float]]],
    history_lock: threading.Lock,
    fallback_fn: Callable[[List[SensorPacket]], int],
) -> int:
    pkg = _load_model_package(model_filename)
    if pkg is None:
        return int(fallback_fn(packets))

    room_updates = _collect_room_feature_updates(packets, feature_map)
    if not room_updates:
        return int(fallback_fn(packets))

    room_histories: Dict[str, List[Mapping[str, float]]] = {}
    with history_lock:
        for room_key, room_snapshot in room_updates.items():
            history[room_key].append(room_snapshot)
            room_histories[room_key] = list(history[room_key])

    prediction_flags: List[int] = []
    for room_key in room_updates:
        predicted = _predict_from_history_rows(
            history_rows=room_histories.get(room_key, []),
            raw_feature_columns=raw_feature_columns,
            pkg=pkg,
        )
        if predicted is None:
            return int(fallback_fn(packets))
        prediction_flags.append(predicted)

    return int(any(prediction_flags))


def detect_fire(packets: List[SensorPacket]) -> int:
    return _detect_with_model_or_fallback(
        packets=packets,
        model_filename="smoke_model.pkl",
        feature_map=FIRE_FEATURE_MAP,
        raw_feature_columns=FIRE_RAW_FEATURE_COLUMNS,
        history=_FIRE_HISTORY,
        history_lock=_FIRE_HISTORY_LOCK,
        fallback_fn=detect_fire_threshold,
    )


def detect_flood_ml(packets: List[SensorPacket]) -> int:
    return _detect_with_model_or_fallback(
        packets=packets,
        model_filename="leak_model.pkl",
        feature_map=FLOOD_FEATURE_MAP,
        raw_feature_columns=FLOOD_RAW_FEATURE_COLUMNS,
        history=_FLOOD_HISTORY,
        history_lock=_FLOOD_HISTORY_LOCK,
        fallback_fn=detect_flood_threshold,
    )


def detect_flood(packets: List[SensorPacket]) -> int:
    return detect_flood_ml(packets)


def detect_intrusion_alarm(packets: List[SensorPacket]) -> int:
    intrusion_type_map = {
        "ir_motion": 0,
        "door_break": 1,
        "window_open": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }

    room_map: Dict[str, int] = {}
    next_room_id = ROOM_ID_START
    vectors: List[Tuple[int, int, int, int]] = []

    for p in packets:
        st = _sensor_key(p.sensor_type)
        if st not in intrusion_type_map and p.sensor_type not in (0, 1, 2):
            continue
        if p.reading <= 0:
            continue

        if isinstance(p.room, int):
            room_id = p.room
        else:
            if p.room not in room_map:
                room_map[p.room] = next_room_id
                next_room_id += 1
            room_id = room_map[p.room]

        sensor_type = int(p.sensor_type) if p.sensor_type in (0, 1, 2) else intrusion_type_map[st]
        vectors.append((sensor_type, p.sensor_id, room_id, ROOM_TYPE_LIVING_ROOM))

    return int(detect_intrusion(vectors, strict=False))


def detect_gas_leak(packets: List[SensorPacket]) -> int:
    room_updates = _room_signal_updates_for_gas_leak(packets)
    if not room_updates:
        return 0

    room_histories: Dict[str, List[float]] = {}
    with _GAS_SIGNAL_HISTORY_LOCK:
        for room_key, signal_value in room_updates.items():
            _GAS_SIGNAL_HISTORY[room_key].append(float(signal_value))
            room_histories[room_key] = list(_GAS_SIGNAL_HISTORY[room_key])

    predictions: List[int] = []
    for room_key in room_updates:
        signal = room_histories.get(room_key, [])
        if len(signal) < GAS_MIN_HISTORY:
            # Insufficient history for reliable cumulant calculation; safely return no alarm for this room.
            predictions.append(0)
            continue
        cumulants = _cumulants_2_to_6(signal)
        k2 = abs(float(cumulants.get(2, 0.0)))
        k3 = abs(float(cumulants.get(3, 0.0)))
        k4 = abs(float(cumulants.get(4, 0.0)))
        recent_mean = sum(signal[-GAS_MIN_HISTORY:]) / GAS_MIN_HISTORY
        score = k2 + GAS_K3_WEIGHT * k3 + GAS_K4_WEIGHT * k4
        predictions.append(int(recent_mean >= GAS_MEAN_THRESHOLD and score >= GAS_SCORE_THRESHOLD))

    if any(predictions):
        return 1
    return 0


def process_sensor_batch(raw_packets: Iterable[PacketLike]) -> List[int]:
    packets = _normalize_packets(raw_packets)

    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {
            "flood": pool.submit(detect_flood, packets),
            "fire": pool.submit(detect_fire, packets),
            "gas_leak": pool.submit(detect_gas_leak, packets),
            "intrusion": pool.submit(detect_intrusion_alarm, packets),
        }

        return [
            future_map["flood"].result(),
            future_map["fire"].result(),
            future_map["gas_leak"].result(),
            future_map["intrusion"].result(),
        ]


def main() -> None:
    data = json.load(sys.stdin)
    result = process_sensor_batch(data)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
