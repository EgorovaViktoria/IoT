"""
Shared inference logic for IoT model testing UI.

Provides:
  - apply_windowing: rolling window aggregation
  - evaluate_classifier: metrics + optional proba flip
  - run_smoke_test: fire/smoke model evaluation
  - run_leak_test: water leak model evaluation
  - run_gas_test: cumulants-based gas detection evaluation
  - run_intrusion_test: rule-based intrusion detection evaluation
"""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_UTILITS_DIR = _REPO_ROOT / "utilits"
if str(_UTILITS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILITS_DIR))

DEFAULT_CSV_PATH = str(_REPO_ROOT / "test" / "unified_test_clean.csv")
DEFAULT_SMOKE_MODEL = str(_REPO_ROOT / "models" / "smoke_model.pkl")
DEFAULT_LEAK_MODEL = str(_REPO_ROOT / "models" / "leak_model.pkl")
DEFAULT_OUT_DIR = str(_REPO_ROOT / "test_results")

# ---------------------------------------------------------------------------
# Column mappings (CSV col → feature name used during training)
# ---------------------------------------------------------------------------
SMOKE_COLUMN_MAP: Dict[str, str] = {
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
SMOKE_FEATURE_COLS = list(SMOKE_COLUMN_MAP.values())

LEAK_COLUMN_MAP: Dict[str, str] = {
    "press_pipe_bar": "Pressure (bar)",
    "flow_rate_lps": "Flow Rate (L/s)",
    "temp_pipe_c": "Temperature (C)",
}
LEAK_FEATURE_COLS = list(LEAK_COLUMN_MAP.values())

# Gas detection: columns used to form the normalised signal
GAS_PIPE_COLS = ["press_pipe_bar", "flow_rate_lps", "temp_pipe_c"]

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def apply_windowing(
    X: pd.DataFrame,
    y: pd.Series,
    window_size: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Rolling window aggregation (mean/std/max/min) for X; rolling-max for y."""
    X_agg = X.rolling(window_size, min_periods=1).agg(["mean", "std", "max", "min"])
    X_agg.columns = [
        "_".join(c)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("°", "")
        .strip()
        for c in X_agg.columns
    ]
    y_agg = y.rolling(window_size, min_periods=1).max().astype(int)
    return X_agg.fillna(0), y_agg


# ---------------------------------------------------------------------------
# Classifier evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    name: str = "Model",
    flip_if_low_auc: bool = False,
    log_fn: Callable[[str], None] = print,
) -> Dict:
    """Return dict with metrics, pred, proba arrays.

    If flip_if_low_auc is True and roc_auc < 0.5, inverts probabilities
    (same logic as fire_test.py).
    """
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.5

    if flip_if_low_auc and auc < 0.5:
        log_fn("⚠  ROC AUC < 0.5 — inverting probabilities (flip logic).")
        proba = 1.0 - proba
        auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.5

    pred = (proba >= threshold).astype(int)
    pr_auc = (
        float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else 0.0
    )

    return {
        "model": name,
        "threshold": float(threshold),
        "test_size": int(len(y)),
        "positives": int(y.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": pr_auc,
        "roc_auc": float(auc),
        "pred": pred,
        "proba": proba,
    }


# ---------------------------------------------------------------------------
# Smoke test (mirrors fire_test.py)
# ---------------------------------------------------------------------------

def run_smoke_test(
    model_path: str,
    csv_path: str,
    window_size: int = 5,
    out_dir: str = DEFAULT_OUT_DIR,
    log_fn: Callable[[str], None] = print,
) -> Tuple[Dict, pd.DataFrame]:
    """Run smoke/fire model evaluation.

    Returns (metrics_dict, result_dataframe).
    Saves result_dataframe to out_dir/fire.csv.
    """
    log_fn(f"Loading model: {model_path}")
    pkg = joblib.load(model_path)
    log_fn(f"Loading dataset: {csv_path}")
    unified = pd.read_csv(csv_path)
    log_fn(f"Total test rows: {len(unified)}")

    required_raw = list(SMOKE_COLUMN_MAP.keys()) + ["smoke_label"]
    missing_cols = [c for c in required_raw if c not in unified.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    df = unified.rename(columns=SMOKE_COLUMN_MAP)
    df_feat = df[SMOKE_FEATURE_COLS].fillna(0)
    y_raw = df["smoke_label"].reindex(df_feat.index).fillna(0).astype(int)

    X_win, y_win = apply_windowing(df_feat, y_raw, window_size)

    features = pkg["features"]
    missing = [f for f in features if f not in X_win.columns]
    if missing:
        log_fn(f"⚠  Missing features: {missing} — filling with 0.")
        for m in missing:
            X_win[m] = 0

    X_test = X_win[features]
    y_test = y_win

    metrics = evaluate_classifier(
        pkg["model"], X_test, y_test, pkg["threshold"],
        name="smoke", flip_if_low_auc=True, log_fn=log_fn,
    )

    result_df = pd.DataFrame({
        "y_true_raw": df["smoke_label"].values,
        "y_true_win": y_win.values,
        "y_pred": metrics["pred"],
        "y_proba": metrics["proba"],
    })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fire.csv")
    result_df.to_csv(out_path, index=False)
    log_fn(f"Results saved to {out_path}")

    return metrics, result_df


# ---------------------------------------------------------------------------
# Leak test (mirrors water_test.py)
# ---------------------------------------------------------------------------

def run_leak_test(
    model_path: str,
    csv_path: str,
    window_size: int = 5,
    out_dir: str = DEFAULT_OUT_DIR,
    log_fn: Callable[[str], None] = print,
) -> Tuple[Dict, pd.DataFrame]:
    """Run water-leak model evaluation.

    Returns (metrics_dict, result_dataframe).
    Saves result_dataframe to out_dir/leak.csv.
    """
    log_fn(f"Loading model: {model_path}")
    pkg = joblib.load(model_path)
    log_fn(f"Loading dataset: {csv_path}")
    unified = pd.read_csv(csv_path)
    log_fn(f"Total test rows: {len(unified)}")

    required_raw = list(LEAK_COLUMN_MAP.keys()) + ["leak_label"]
    missing_cols = [c for c in required_raw if c not in unified.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    df = unified.rename(columns=LEAK_COLUMN_MAP)

    if "Leak Status" not in df.columns:
        df["Leak Status"] = df["leak_label"]

    y_raw = df["leak_label"].copy()
    feature_cols_raw = LEAK_FEATURE_COLS + ["Leak Status"]

    valid_idx = df[feature_cols_raw].dropna().index
    X_raw = df.loc[valid_idx, feature_cols_raw]
    y_raw = y_raw.loc[valid_idx]

    X_win, y_win = apply_windowing(X_raw, y_raw, window_size)

    features = pkg["features"]
    missing = [f for f in features if f not in X_win.columns]
    if missing:
        log_fn(f"⚠  Missing features: {missing} — filling with 0.")
        for m in missing:
            X_win[m] = 0

    X_test = X_win[features]
    y_test = y_win.reindex(X_test.index)

    metrics = evaluate_classifier(
        pkg["model"], X_test, y_test, pkg["threshold"],
        name="leak", flip_if_low_auc=False, log_fn=log_fn,
    )

    result_df = pd.DataFrame({
        "y_true_raw": df["leak_label"].values,
        "y_true_win": y_win.values,
        "y_pred": metrics["pred"],
        "y_proba": metrics["proba"],
    })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "leak.csv")
    result_df.to_csv(out_path, index=False)
    log_fn(f"Results saved to {out_path}")

    return metrics, result_df


# ---------------------------------------------------------------------------
# Gas detection test (cumulants-based streaming simulation)
# ---------------------------------------------------------------------------

_GAS_MEAN_THRESHOLD = 0.55
_GAS_SCORE_THRESHOLD = 0.01
_GAS_K3_WEIGHT = 0.2
_GAS_K4_WEIGHT = 0.05
_GAS_HISTORY_SIZE = 32
_GAS_MIN_HISTORY = 2


def _central_moment(x: np.ndarray, s: int) -> float:
    m = float(np.mean(x))
    return float(np.mean((x - m) ** s))


def _cumulants_2_to_4(signal: List[float]) -> Tuple[float, float, float]:
    arr = np.array(signal, dtype=np.float64)
    mu2 = _central_moment(arr, 2)
    mu3 = _central_moment(arr, 3)
    mu4 = _central_moment(arr, 4)
    k2 = mu2
    k3 = mu3
    k4 = mu4 - 3.0 * (mu2 ** 2)
    return k2, k3, k4


def _gas_signal_from_row(row: pd.Series) -> float:
    """Compute the normalised gas-signal score from a CSV row.

    Converts CSV values (bar, L/s, °C) to normalised scores using the same
    scaling as emergency_system._room_signal_updates_for_gas_leak which
    expects SI units (Pa, m³/s, K) coming from hardware packets.
    """
    # press_pipe_bar: bar → Pa → score = Pa / 500000
    press_score = float(row.get("press_pipe_bar", 0.0)) * 100000.0 / 500000.0
    # flow_rate_lps: L/s → m³/s → score = m³/s / 0.3
    flow_score = float(row.get("flow_rate_lps", 0.0)) * 0.001 / 0.3
    # temp_pipe_c: °C (already Celsius) → score = C / 100
    temp_score = float(row.get("temp_pipe_c", 0.0)) / 100.0

    score = max(press_score, flow_score, temp_score)
    return max(0.0, min(1.5, score))


def run_gas_test(
    csv_path: str,
    out_dir: str = DEFAULT_OUT_DIR,
    log_fn: Callable[[str], None] = print,
) -> Tuple[Dict, pd.DataFrame]:
    """Simulate streaming gas detection on the unified CSV.

    Uses the cumulants algorithm from emergency_system.detect_gas_leak.
    Ground truth is 'leak_label' (closest available label for pipe anomalies).
    Returns (metrics_dict, result_dataframe).
    Saves result_dataframe to out_dir/gas.csv.
    """
    log_fn(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    log_fn(f"Total rows: {len(df)}")

    required = ["press_pipe_bar", "flow_rate_lps", "temp_pipe_c", "leak_label"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    history: deque = deque(maxlen=_GAS_HISTORY_SIZE)
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    for _, row in df.iterrows():
        signal_val = _gas_signal_from_row(row)
        history.append(signal_val)

        label = int(row["leak_label"])
        y_true.append(label)

        if len(history) < _GAS_MIN_HISTORY:
            y_pred.append(0)
            y_score.append(0.0)
            continue

        k2, k3, k4 = _cumulants_2_to_4(list(history))
        recent_mean = sum(list(history)[-_GAS_MIN_HISTORY:]) / _GAS_MIN_HISTORY
        cumulant_score = abs(k2) + _GAS_K3_WEIGHT * abs(k3) + _GAS_K4_WEIGHT * abs(k4)

        # Normalised confidence proxy (clipped to [0, 1])
        confidence = min(1.0, max(0.0, recent_mean))

        pred = int(recent_mean >= _GAS_MEAN_THRESHOLD and cumulant_score >= _GAS_SCORE_THRESHOLD)
        y_pred.append(pred)
        y_score.append(float(confidence))

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)

    n_classes = len(np.unique(y_true_arr))
    metrics = {
        "model": "gas_cumulants",
        "test_size": len(y_true_arr),
        "positives": int(y_true_arr.sum()),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)) if n_classes > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)) if n_classes > 1 else 0.5,
        "pred": y_pred_arr,
        "proba": y_score_arr,
    }

    result_df = pd.DataFrame({
        "y_true_raw": y_true_arr,
        "y_pred": y_pred_arr,
        "y_score": y_score_arr,
    })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gas.csv")
    result_df.to_csv(out_path, index=False)
    log_fn(f"Results saved to {out_path}")

    return metrics, result_df


# ---------------------------------------------------------------------------
# Intrusion detection test (rule-based)
# ---------------------------------------------------------------------------

def run_intrusion_test(
    csv_path: str,
    out_dir: str = DEFAULT_OUT_DIR,
    log_fn: Callable[[str], None] = print,
) -> Tuple[Dict, pd.DataFrame]:
    """Run rule-based intrusion detection on a sensor-vector CSV.

    Expected CSV columns: sensor_type (int 0-2), sensor_id (int),
    room_id (int), room_type (int 0-4), batch_id (int, groups rows into
    batches), expected (int 0/1 optional ground truth).

    Returns (metrics_dict, result_dataframe).
    Saves result_dataframe to out_dir/intrusion.csv.
    """
    try:
        from intrusion_detection import detect_intrusion_rooms  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "Could not import intrusion_detection. "
            "Ensure utilits/ is in the Python path."
        )

    log_fn(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    log_fn(f"Total rows: {len(df)}")

    required = ["sensor_type", "sensor_id", "room_id", "room_type", "batch_id"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV missing required columns: {missing_cols}\n"
            "Expected: sensor_type, sensor_id, room_id, room_type, batch_id, "
            "[expected]"
        )

    has_ground_truth = "expected" in df.columns

    batch_ids = df["batch_id"].unique()
    y_true: List[int] = []
    y_pred: List[int] = []
    rooms_detected: List[str] = []

    for bid in sorted(batch_ids):
        batch = df[df["batch_id"] == bid]
        vectors = [
            [
                int(row["sensor_type"]),
                int(row["sensor_id"]),
                int(row["room_id"]),
                int(row["room_type"]),
            ]
            for _, row in batch.iterrows()
        ]
        rooms = detect_intrusion_rooms(vectors, strict=False)
        pred = int(len(rooms) > 0)
        y_pred.append(pred)
        rooms_detected.append(",".join(str(r) for r in rooms))

        if has_ground_truth:
            expected_vals = batch["expected"].dropna().astype(int)
            y_true.append(int(expected_vals.max()) if len(expected_vals) else 0)

    y_pred_arr = np.array(y_pred, dtype=int)

    metrics: Dict = {
        "model": "intrusion_rules",
        "test_size": len(batch_ids),
        "positives": int(y_pred_arr.sum()),
    }

    if has_ground_truth:
        y_true_arr = np.array(y_true, dtype=int)
        metrics.update({
            "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        })
        result_df = pd.DataFrame({
            "batch_id": sorted(batch_ids),
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "rooms": rooms_detected,
        })
    else:
        result_df = pd.DataFrame({
            "batch_id": sorted(batch_ids),
            "y_pred": y_pred_arr,
            "rooms": rooms_detected,
        })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "intrusion.csv")
    result_df.to_csv(out_path, index=False)
    log_fn(f"Results saved to {out_path}")

    return metrics, result_df
