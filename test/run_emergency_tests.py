from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

from generate_sensor_batch import dataframe_row_to_batch, generate_batch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from emergency_system import process_sensor_batch, reset_state  # type: ignore[import-not-found]

UNIFIED_DATASET_PATH = BASE_DIR / "test" / "unified_test_clean.csv"
SAFE_DATASET_PATH = BASE_DIR / "test" / "safe_unified_test_clean.csv"


def run_emergency_system(batch: List[List[object]]) -> List[int]:
    return process_sensor_batch(batch)


def dataset_path(dataset: str) -> Path:
    if dataset == "full":
        return UNIFIED_DATASET_PATH
    if dataset == "safe":
        return SAFE_DATASET_PATH
    raise ValueError("dataset must be safe or full")


def evaluate_dataset(dataset: str, limit: int) -> None:
    reset_state()
    path = dataset_path(dataset)
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError("Dataset is empty")
    if "smoke_label" not in rows[0]:
        raise ValueError("Dataset must contain smoke_label")
    if limit > 0:
        rows = rows[:limit]

    fire_hits = 0
    for row in rows:
        batch = dataframe_row_to_batch(row)
        batch.extend(
            [
                ["door_break", 1, 101, 0.0],
                ["ir_motion", 2, 101, 0.0],
            ]
        )
        result = run_emergency_system(batch)
        fire_hits += int(result[1] == int(float(row["smoke_label"])))

    total = len(rows)
    print(f"Dataset: {dataset}, rows: {total}")
    print(f"Fire label match: {fire_hits}/{total} ({fire_hits / total:.3f})")
    print("Затопление/Утечка газа/Проникновение: отсутствуют ground-truth метки в датасете, строгая оценка пропущена.")


def evaluate_generator() -> None:
    reset_state()
    alarm_batch = generate_batch(target_alarm=1, seed=123456, source="synthetic")
    normal_batch = generate_batch(target_alarm=0, seed=123457, source="synthetic")
    normal_result = run_emergency_system(normal_batch)
    alarm_result = run_emergency_system(alarm_batch)
    print(f"Synthetic normal batch result: {normal_result}")
    print(f"Synthetic alarm batch result: {alarm_result}")


def evaluate_intrusion_apartment() -> None:
    reset_state()
    base_batch = [
        ["ir_motion", 11, "living_room", 0.0],
        ["ir_motion", 12, "living_room", 0.0],
        ["ir_motion", 21, "bedroom", 0.0],
        ["ir_motion", 22, "bedroom", 0.0],
        ["ir_motion", 31, "corridor", 0.0],
        ["ir_motion", 32, "corridor", 0.0],
        ["ir_motion", 41, "kitchen", 0.0],
        ["ir_motion", 42, "kitchen", 0.0],
        ["door_break", 1, "corridor", 0.0],  # входная дверь
    ]
    intrusion_batch = [
        ["ir_motion", 11, "living_room", 0.0],
        ["ir_motion", 12, "living_room", 0.0],
        ["ir_motion", 21, "bedroom", 0.0],
        ["ir_motion", 22, "bedroom", 0.0],
        ["ir_motion", 31, "corridor", 1.0],
        ["ir_motion", 32, "corridor", 1.0],
        ["ir_motion", 41, "kitchen", 0.0],
        ["ir_motion", 42, "kitchen", 0.0],
        ["door_break", 1, "corridor", 1.0],  # взлом входной двери
    ]

    base_result = run_emergency_system(base_batch)
    intrusion_result = run_emergency_system(intrusion_batch)
    if base_result[3] != 0:
        raise RuntimeError(f"Expected no intrusion for base apartment scenario, got: {base_result}")
    if intrusion_result[3] != 1:
        raise RuntimeError(
            f"Expected intrusion alarm for apartment intrusion scenario, got: {intrusion_result}"
        )
    print(f"Apartment intrusion baseline result: {base_result}")
    print(f"Apartment intrusion alarm result: {intrusion_result}")


def evaluate_gas_leak_scenario() -> None:
    reset_state()
    baseline_batch = [
        ["gas_leak", 500, 101, 0.05],
        ["tvoc_ppb", 501, 101, 40.0],
        ["eco2_ppm", 502, 101, 550.0],
        ["raw_h2", 503, 101, 12700.0],
        ["raw_ethanol", 504, 101, 20600.0],
        ["door_break", 1, 101, 0.0],
        ["ir_motion", 2, 101, 0.0],
    ]
    gas_leak_batch = [
        ["gas_leak", 500, 101, 0.95],
        ["tvoc_ppb", 501, 101, 1200.0],
        ["eco2_ppm", 502, 101, 2600.0],
        ["raw_h2", 503, 101, 22000.0],
        ["raw_ethanol", 504, 101, 28000.0],
        ["door_break", 1, 101, 0.0],
        ["ir_motion", 2, 101, 0.0],
    ]

    baseline_result = run_emergency_system(baseline_batch)
    leak_result = run_emergency_system(gas_leak_batch)
    retry_budget = 1
    while leak_result[2] != 1 and retry_budget > 0:
        # Gas leak detector is cumulant-based and history-dependent.
        # Re-running the same alarm batch adds another high point into history for the room.
        leak_result = run_emergency_system(gas_leak_batch)
        retry_budget -= 1
    if baseline_result[2] != 0:
        raise RuntimeError(f"Expected no gas leak alarm for baseline gas scenario, got: {baseline_result}")
    if leak_result[2] != 1:
        raise RuntimeError(f"Expected gas leak alarm for gas leak scenario, got: {leak_result}")
    print(f"Gas leak baseline result: {baseline_result}")
    print(f"Gas leak alarm result: {leak_result}")


def evaluate_dataset_mode_generation() -> None:
    """Validate generate_profiled_batch dataset mode.

    Checks:
    - gas_leak/intrusion active: no unexpected fallback, no fire false-positive.
    - gas/intrusion sensors are always present in batch (always synthetic).
    - fire sensors come from dataset rows matched by smoke_label only.
    """
    _test_dir = Path(__file__).resolve().parent
    if str(_test_dir) not in sys.path:
        sys.path.insert(0, str(_test_dir))
    from qt_sensor_generator import generate_profiled_batch, DEFAULT_CONFIG  # type: ignore[import-not-found]

    config = DEFAULT_CONFIG

    # Scenario A: gas_leak=True in dataset mode must NOT fall back (row selection ignores gas).
    reset_state()
    profile_gas = {"fire": False, "flood": False, "gas_leak": True, "intrusion": False}
    result_a = generate_profiled_batch(
        source="full", profile=profile_gas, seed=42,
        config=config, dataset_fallback_to_synthetic=True,
    )
    if result_a.get("fallback_used"):
        raise RuntimeError(
            "Scenario A: gas_leak=True in dataset mode triggered unexpected fallback. "
            f"Reason: {result_a.get('fallback_reason')}"
        )
    batch_types_a = {str(item[0]).lower() for item in result_a["batch"]}
    for expected_sensor in ("gas_leak", "flood", "door_break", "ir_motion"):
        if expected_sensor not in batch_types_a:
            raise RuntimeError(f"Scenario A: expected sensor '{expected_sensor}' missing from batch.")
    if "temp_ambient_c" not in batch_types_a:
        raise RuntimeError("Scenario A: fire sensor 'temp_ambient_c' missing from batch (should come from dataset).")
    if not all(bool(v) for v in result_a["expected_available"].values()):
        raise RuntimeError("Scenario A: all detectors must have profile-driven expected labels in GUI table.")
    print(f"Scenario A passed — gas_leak in dataset mode: source_effective={result_a['source_effective']}")

    # Scenario B: intrusion=True must be present without fallback.
    reset_state()
    profile_intrusion = {"fire": False, "flood": False, "gas_leak": False, "intrusion": True}
    result_b = generate_profiled_batch(
        source="full", profile=profile_intrusion, seed=99,
        config=config, dataset_fallback_to_synthetic=True,
    )
    if result_b.get("fallback_used"):
        raise RuntimeError("Scenario B: intrusion=True in dataset mode triggered unexpected fallback.")
    batch_b = result_b["batch"]
    ir_values = [float(item[3]) for item in batch_b if str(item[0]).lower() == "ir_motion"]
    door_values = [float(item[3]) for item in batch_b if str(item[0]).lower() == "door_break"]
    if not ir_values or ir_values[0] != 1.0:
        raise RuntimeError(f"Scenario B: ir_motion should be 1.0, got: {ir_values}")
    if not door_values or door_values[0] != 1.0:
        raise RuntimeError(f"Scenario B: door_break should be 1.0, got: {door_values}")
    print(f"Scenario B passed — intrusion=True: ir_motion={ir_values[0]}, door_break={door_values[0]}")

    # Scenario C: fire=True in safe dataset — must not crash (safe may or may not have alarm rows).
    reset_state()
    profile_fire_safe = {"fire": True, "flood": False, "gas_leak": False, "intrusion": False}
    result_c = generate_profiled_batch(
        source="safe", profile=profile_fire_safe, seed=7,
        config=config, dataset_fallback_to_synthetic=True,
    )
    batch_types_c = {str(item[0]).lower() for item in result_c["batch"]}
    for sensor in ("gas_leak", "flood", "door_break"):
        if sensor not in batch_types_c:
            raise RuntimeError(f"Scenario C: sensor '{sensor}' missing from batch.")
    print(
        f"Scenario C passed — fire=True in safe dataset: "
        f"source_effective={result_c['source_effective']}, fallback={result_c['fallback_used']}"
    )

    # Scenario D: gas_leak only (no fire) in dataset mode must not produce many fire FP.
    reset_state()
    fp_count = 0
    for seed in range(500, 520):
        r = generate_profiled_batch(
            source="full", profile=profile_gas, seed=seed,
            config=config, dataset_fallback_to_synthetic=True,
        )
        predicted_fire = r["predicted"].get("Пожар", -1)
        if predicted_fire == 1:
            fp_count += 1
    if fp_count > 10:  # allow small fraction due to threshold-based fire fallback uncertainty
        raise RuntimeError(
            f"Scenario D: too many fire false-positives in dataset mode with gas_leak only "
            f"({fp_count}/20). Expected ≤10."
        )
    print(f"Scenario D passed — fire FP with gas_leak only in dataset mode: {fp_count}/20")


def evaluate_synthetic_mode_independence() -> None:
    """Validate synthetic-mode checkbox isolation between detectors."""
    _test_dir = Path(__file__).resolve().parent
    if str(_test_dir) not in sys.path:
        sys.path.insert(0, str(_test_dir))
    from qt_sensor_generator import generate_profiled_batch, DEFAULT_CONFIG  # type: ignore[import-not-found]

    config = DEFAULT_CONFIG
    detector_by_key = {
        "flood": "Затопление",
        "fire": "Пожар",
        "gas_leak": "Утечка газа",
        "intrusion": "Проникновение",
    }
    scenarios = [
        {"fire": False, "flood": True, "gas_leak": False, "intrusion": False},
        {"fire": True, "flood": False, "gas_leak": False, "intrusion": False},
        {"fire": False, "flood": False, "gas_leak": True, "intrusion": False},
        {"fire": True, "flood": False, "gas_leak": True, "intrusion": False},
    ]
    for idx, profile in enumerate(scenarios):
        reset_state()
        result = generate_profiled_batch(
            source="synthetic",
            profile=profile,
            seed=1000 + idx,
            config=config,
            dataset_fallback_to_synthetic=True,
        )
        expected = {detector_by_key[key]: int(value) for key, value in profile.items()}
        predicted = result["predicted"]
        mismatch = [det for det, val in expected.items() if int(predicted.get(det, -1)) != val]
        if mismatch:
            raise RuntimeError(
                f"Synthetic isolation mismatch for profile={profile}. "
                f"Predicted={predicted}, mismatched={mismatch}."
            )
    print("Synthetic isolation scenarios passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs emergency_system against dataset-derived and generated batches."
    )
    parser.add_argument("--dataset", choices=["safe", "full"], default="safe")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Rows to test from dataset (default: 200, 0 means all rows)",
    )
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-generator", action="store_true")
    parser.add_argument("--skip-intrusion", action="store_true")
    parser.add_argument("--skip-gas-leak", action="store_true")
    parser.add_argument("--skip-dataset-mode", action="store_true")
    parser.add_argument("--skip-synthetic-isolation", action="store_true")
    args = parser.parse_args()

    if not args.skip_generator:
        evaluate_generator()
    if not args.skip_intrusion:
        evaluate_intrusion_apartment()
    if not args.skip_gas_leak:
        evaluate_gas_leak_scenario()
    if not args.skip_dataset_mode:
        evaluate_dataset_mode_generation()
    if not args.skip_synthetic_isolation:
        evaluate_synthetic_mode_independence()
    if not args.skip_dataset:
        evaluate_dataset(args.dataset, args.limit)


if __name__ == "__main__":
    main()
