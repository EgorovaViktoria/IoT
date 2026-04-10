
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

#warnings.filterwarnings("ignore")

MODEL_PATH = "models/smoke_model.pkl"
UNIFIED_TEST_PATH = "fire_leak_dfs/unified_test_clean.csv" # safe_...


def apply_windowing(X, y, window_size=5):
    # smoothens X data (fire metrics, not target varible) to ignore spikes and thus iliminate false alarms
    X_agg = X.rolling(window_size, min_periods=1).agg(["mean", "std", "max", "min"])

    # reanming cuz it's required by the rolling window function
    X_agg.columns = ["_".join(c).replace(" ","_").replace("(","").replace(")","").replace("°","").strip() 
                     for c in X_agg.columns]
    
    y_agg = y.rolling(window_size, min_periods=1).max().astype(int)
    return X_agg.fillna(0), y_agg


def evaluate(model, X, y, threshold, name="Model"):
    proba = model.predict_proba(X)[:, 1]  # seletcs probablities ONLY for target=1 
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.5
    if auc < 0.5:
        print("UAC < 0.5 !!!!!!!!!!!!!!!!!")
        proba = 1 - proba
        auc = roc_auc_score(y, proba)

    pred = (proba >= threshold).astype(int)

    return {
        "proba": proba,
        "pred": pred,
        "model": name,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": auc,
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
                       "nc2_5": "NC2.5"
})

# listing all the features required for FIRE testing 
feature_cols_raw = ["Temperature[C]", 
                    "Humidity[%]", 
                    "TVOC[ppb]", 
                    "eCO2[ppm]",
                    "Raw H2", 
                    "Raw Ethanol",
                    "Pressure[hPa]", 
                    "PM1.0",
                    "PM2.5",
                    "NC0.5", 
                    "NC1.0", 
                    "NC2.5"]

# seecting only the required fetures + binary target
df_feat = df[feature_cols_raw].fillna(0)  # just in case
y_raw = df["smoke_label"].reindex(df_feat.index).fillna(0).astype(int)

# applying rolling window (genrates _mean, _std, _max, _min columns)
X_win, y_win = apply_windowing(df_feat, y_raw, window_size=5)


features = pkg["features"]
missing = [f for f in features if f not in X_win.columns]
if missing:
    print(f"missing features detected: {missing} (filling with 0)")
    for m in missing: 
        X_win[m] = 0

X_test = X_win[features]
y_test = y_win  # windowed y_true


# metrics stuff
print("METRICS:")
metrics = evaluate(pkg["model"], X_test, y_test, pkg["threshold"], "smoke")

pd.DataFrame({
    "y_true_raw": df["smoke_label"].values,  # raw (instantaneous) y_true 
    "y_true_win": y_win.values,  # y_true windowed
    "y_pred": metrics["pred"],  # y_pred windowed
    "y_proba": metrics["proba"]  # confidence probalbility 
}).to_csv("test_results/fire.csv", index=False)

for key, val in metrics.items():
    if key in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
        print(f"   {key:10s}: {val:.4f}")
print()
print(f"test dataset size: {metrics['test_size']}")
print(f"windowed positives (target=1): {metrics['positives']}")

