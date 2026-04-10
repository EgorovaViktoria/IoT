"""
GPLA-12 (v1) cumulant features (K2..K6) + 3 derived tasks:
  1) pressure: {0.2, 0.4, 0.5} MPa
  2) noise: {noiseless, noisy}
  3) mic: {mic1, mic2}

Easy run version:
- Put this script in the SAME FOLDER as data.csv and label.csv
- Run: python gpla12_cumulants_3tasks_easy_run.py

Cumulants are computed exactly via central moments:
  K2 = mu2
  K3 = mu3
  K4 = mu4 - 3*mu2^2
  K5 = mu5 - 10*mu3*mu2
  K6 = mu6 - 15*mu4*mu2 - 10*mu3^2 + 30*mu2^3

Filtering:
- GPLA-12 paper does not specify original Fs; after structured tailoring samples have length 1460 for 5 seconds.
- Therefore kHz bands cannot be applied. We provide:
  MODE="nofilter"  -> no filtering
  MODE="bands"     -> normalized bandpass splits (relative to Nyquist)

Edit settings in the "USER SETTINGS" section if needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# =============================================================================
# USER SETTINGS (edit these if needed)
# =============================================================================
MODE = "nofilter"   # "nofilter" or "bands"
WIN_LEN = 256
HOP = 128
TEST_SIZE = 0.25
SEED = 42

# Optional: save extracted features to a CSV (set to "" to disable)
OUT_FEATURES_CSV = "features_gpla12_cumulants_3tasks.csv"
# =============================================================================


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def resolve_dataset_paths(base_dir: str) -> Tuple[str, str]:
    env_data = os.environ.get("GPLA12_DATA_CSV")
    env_label = os.environ.get("GPLA12_LABEL_CSV")
    if env_data and env_label and os.path.exists(env_data) and os.path.exists(env_label):
        return env_data, env_label

    candidates = [
        (os.path.join(base_dir, "data.csv"), os.path.join(base_dir, "label.csv")),
        (
            os.path.join(base_dir, "..", "test", "data.csv"),
            os.path.join(base_dir, "..", "test", "label.csv"),
        ),
        (os.path.join(os.getcwd(), "data.csv"), os.path.join(os.getcwd(), "label.csv")),
    ]

    for data_csv, label_csv in candidates:
        if os.path.exists(data_csv) and os.path.exists(label_csv):
            return data_csv, label_csv

    tried = "\n".join(f"  - {d} | {l}" for d, l in candidates)
    raise FileNotFoundError(
        "Could not locate GPLA-12 files data.csv and label.csv.\n"
        "Provide both files in utilites/, in Code/test/, or set env vars "
        "GPLA12_DATA_CSV and GPLA12_LABEL_CSV.\n"
        f"Tried:\n{tried}"
    )


def load_gpla12_v1(data_csv_path: str, label_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    X = pd.read_csv(data_csv_path, header=None).to_numpy(dtype=np.float64)
    y12 = pd.read_csv(label_csv_path, header=None).to_numpy().reshape(-1).astype(int)
    if X.shape[0] != y12.shape[0]:
        raise ValueError(f"Row mismatch: data has {X.shape[0]} rows, label has {y12.shape[0]} rows")
    if y12.min() < 1 or y12.max() > 12:
        raise ValueError("Expected labels in 1..12 for GPLA-12")
    return X, y12


def map_to_pressure_noise_mic(y12: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = y12.astype(int)

    # mic: 1..6 => mic1(0), 7..12 => mic2(1)
    y_mic = (y >= 7).astype(int)

    # noise: odd => noiseless(0), even => noisy(1)
    y_noise = ((y % 2) == 0).astype(int)

    # pressure within each mic: (1,2)->0.2; (3,4)->0.4; (5,6)->0.5
    idx = ((y - 1) % 6) + 1
    y_pressure = np.zeros_like(y)
    y_pressure[(idx == 1) | (idx == 2)] = 0
    y_pressure[(idx == 3) | (idx == 4)] = 1
    y_pressure[(idx == 5) | (idx == 6)] = 2

    return y_pressure, y_noise, y_mic


# ----------------------------
# Normalized bandpass
# ----------------------------
def _butter_bandpass_norm(lo: float, hi: float, order: int = 8):
    if not (0.0 < lo < hi < 1.0):
        raise ValueError(f"Normalized band must satisfy 0 < lo < hi < 1, got lo={lo}, hi={hi}")
    b, a = butter(order, [lo, hi], btype="bandpass")
    return b, a


def bandpass_filter_norm(x: np.ndarray, lo: float, hi: float, order: int = 8) -> np.ndarray:
    b, a = _butter_bandpass_norm(lo, hi, order=order)
    return filtfilt(b, a, x)


# ----------------------------
# Cumulants (exact formulas)
# ----------------------------
def central_moment(x: np.ndarray, s: int) -> float:
    m = float(np.mean(x))
    return float(np.mean((x - m) ** s))


def cumulants_2_to_6(seg: np.ndarray) -> Dict[int, float]:
    mu2 = central_moment(seg, 2)
    mu3 = central_moment(seg, 3)
    mu4 = central_moment(seg, 4)
    mu5 = central_moment(seg, 5)
    mu6 = central_moment(seg, 6)

    K2 = mu2
    K3 = mu3
    K4 = mu4 - 3.0 * (mu2 ** 2)
    K5 = mu5 - 10.0 * mu3 * mu2
    K6 = mu6 - 15.0 * mu4 * mu2 - 10.0 * (mu3 ** 2) + 30.0 * (mu2 ** 3)
    return {2: K2, 3: K3, 4: K4, 5: K5, 6: K6}


def windowed_cumulant_features(x: np.ndarray, win_len: int = 256, hop: int = 128) -> np.ndarray:
    n = len(x)
    if win_len > n:
        win_len = n
        hop = max(1, n // 2)

    Ks = {k: [] for k in range(2, 7)}
    for start in range(0, n - win_len + 1, hop):
        seg = x[start:start + win_len]
        c = cumulants_2_to_6(seg)
        for k in range(2, 7):
            Ks[k].append(c[k])

    feat: List[float] = []
    for k in range(2, 7):
        arr = np.array(Ks[k], dtype=np.float64)
        feat.append(float(arr.mean()) if arr.size else 0.0)
    for k in range(2, 7):
        arr = np.array(Ks[k], dtype=np.float64)
        feat.append(float(arr.std(ddof=1)) if arr.size > 1 else 0.0)
    return np.array(feat, dtype=np.float64)


@dataclass
class FeatureConfig:
    mode: str = "nofilter"
    win_len: int = 256
    hop: int = 128
    filter_order: int = 8
    bands: Tuple[Tuple[float, float], ...] = (
        (0.02, 0.18),  # low
        (0.18, 0.45),  # mid
        (0.45, 0.90),  # high
    )


def featurize_dataset(X: np.ndarray, cfg: FeatureConfig) -> Tuple[np.ndarray, List[str]]:
    if cfg.mode == "nofilter":
        names = [f"K{k}_mean" for k in range(2, 7)] + [f"K{k}_std" for k in range(2, 7)]
        F = np.vstack([windowed_cumulant_features(X[i], cfg.win_len, cfg.hop) for i in range(X.shape[0])])
        return F, names

    if cfg.mode == "bands":
        names: List[str] = []
        for bi in range(len(cfg.bands)):
            names += [f"b{bi}_K{k}_mean" for k in range(2, 7)] + [f"b{bi}_K{k}_std" for k in range(2, 7)]

        feats = []
        for i in range(X.shape[0]):
            sig = X[i].astype(np.float64)
            all_bands = []
            for (lo, hi) in cfg.bands:
                sig_f = bandpass_filter_norm(sig, lo, hi, order=cfg.filter_order)
                all_bands.append(windowed_cumulant_features(sig_f, cfg.win_len, cfg.hop))
            feats.append(np.concatenate(all_bands))
        return np.vstack(feats), names

    raise ValueError("MODE must be 'nofilter' or 'bands'")


def train_eval_one(F: np.ndarray, y: np.ndarray, task_name: str, test_size: float, seed: int) -> None:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (train_idx, test_idx), = splitter.split(F, y)

    X_train, X_test = F[train_idx], F[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
        )),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n" + "=" * 80)
    print(f"TASK: {task_name}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))


def main():
    base = script_dir()
    data_csv, label_csv = resolve_dataset_paths(base)

    X, y12 = load_gpla12_v1(data_csv, label_csv)
    y_pressure, y_noise, y_mic = map_to_pressure_noise_mic(y12)

    cfg = FeatureConfig(mode=MODE, win_len=WIN_LEN, hop=HOP)
    F, names = featurize_dataset(X, cfg)

    if OUT_FEATURES_CSV:
        out_path = os.path.join(base, OUT_FEATURES_CSV)
        df = pd.DataFrame(F, columns=names)
        df.insert(0, "label12", y12)
        df.insert(1, "pressure", y_pressure)  # 0->0.2, 1->0.4, 2->0.5
        df.insert(2, "noise", y_noise)        # 0->noiseless, 1->noisy
        df.insert(3, "mic", y_mic)            # 0->mic1, 1->mic2
        df.to_csv(out_path, index=False)
        print(f"Saved features to: {out_path}")

    train_eval_one(F, y_pressure, "pressure (0:0.2MPa, 1:0.4MPa, 2:0.5MPa)", TEST_SIZE, SEED)
    train_eval_one(F, y_noise, "noise (0:noiseless, 1:noisy)", TEST_SIZE, SEED)
    train_eval_one(F, y_mic, "microphone (0:mic1, 1:mic2)", TEST_SIZE, SEED)


if __name__ == "__main__":
    main()
