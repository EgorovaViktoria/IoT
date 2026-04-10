
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

#warnings.filterwarnings("ignore")

MODEL_PATH = "models/leak_model.pkl"
UNIFIED_TEST_PATH = "fire_leak_dfs/unified_test_clean.csv" # safe_...


def apply_windowing(X, y, window_size=5):
    # smoothens X data (fire metrics, not target varible) to ignore spikes and thus iliminate false alarms
    X_agg = X.rolling(window_size, min_periods=1).agg(["mean", "std", "max", "min"])

    # reanming cuz it"s required by the rolling window function
    X_agg.columns = ["_".join(c).replace(" ","_").replace("(","").replace(")","").replace("°","").strip() 
                     for c in X_agg.columns]

    y_agg = y.rolling(window_size, min_periods=1).max().astype(int)
    return X_agg.fillna(0), y_agg


def evaluate(model, X, y, threshold, name="Model"):
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.5

    if auc < 0.5:
        proba = 1 - proba
        auc = roc_auc_score(y, proba)

    pred = (proba >= threshold).astype(int)

    return {
        "pred": pred,
        "proba": proba,
        "model": name,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, proba),
        "threshold": threshold,
        "test_size": len(y),
        "positives": int(y.sum())
    }


pkg = joblib.load(MODEL_PATH)
unified = pd.read_csv(UNIFIED_TEST_PATH)

#print(pkg["features"])
print(f"total test rows: {len(unified)}")

# ranaming columsn to match those used during windowing in training
df = unified.rename(columns={
                       "press_pipe_bar": "Pressure (bar)",
                       "flow_rate_lps": "Flow Rate (L/s)",
                       "temp_pipe_c": "Temperature (C)"  # matches "Temperature_C_max" after windowing
})


if "Leak Status" not in df.columns:
    df["Leak Status"] = df["leak_label"]

y_raw = df["leak_label"].copy()
feature_cols_raw = ["Pressure (bar)", 
                    "Flow Rate (L/s)", 
                    "Temperature (C)", 
                    "Leak Status"
]

valid_idx = df[feature_cols_raw].dropna().index
X_raw = df.loc[valid_idx, feature_cols_raw]
y_raw = y_raw.loc[valid_idx]


# applying rolling window (genrates _mean, _std, _max, _min columns)
X_win, y_win = apply_windowing(X_raw, y_raw, window_size=5)


features = pkg["features"]

missing = [f for f in features if f not in X_win.columns]
if missing:
    print(f"missing features detected: {missing} (filling with 0)")
    for m in missing:
        X_win[m] = 0

X_test = X_win[features]
y_test = y_win.reindex(X_test.index)  # windowed y_true


# metrics stuff
print("METRICS:")
metrics = evaluate(pkg["model"], X_test, y_test, pkg["threshold"], "leak")

pd.DataFrame({
    "y_true_raw": df["leak_label"].values,  # raw (instantaneous) y_true 
    "y_true_win": y_win.values,  # y_true windowed
    "y_pred": metrics["pred"],  # y_pred windowed
    "y_proba": metrics["proba"]  # confidence probalbility 
}).to_csv("test_results/leak.csv", index=False)

for key, val in metrics.items():
    if key in ["precision", "recall", "f1", "pr_auc"]:
        print(f"   {key:10s}: {val:.4f}")
print()
print(f"test dataset size: {metrics['test_size']}")
print(f"windowed positives (target=1): {metrics['positives']}")
