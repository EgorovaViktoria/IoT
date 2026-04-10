#!/usr/bin/env python3
"""Analyze alarm-class (label=1) value ranges in the sensor datasets.

Usage
-----
    python3 analyze_dataset_alarm_ranges.py [--dataset PATH] [--label COLUMN]
                                            [--output {stdout,csv,json}]
                                            [--out-file PATH]

The script walks through one or both datasets and, for every alarm label
column it finds (smoke_label, leak_label …), prints the value ranges of all
numeric feature columns **when the label is 1** (alarm condition).

Output includes: count, min, 10th/25th/50th/75th/90th percentiles, max.

Contract context
----------------
- ML detects: fire (smoke_label) and flood (no label in current datasets).
- Cumulant method detects: gas leak (leak_label / gas sensor signal).
- Rule-based: intrusion (no label in datasets).

The ranges reported here help calibrate the synthetic alarm thresholds and
verify that the ML models are operating in a realistic value space.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default dataset paths (relative to this file's grandparent = Code/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent
DEFAULT_DATASETS = {
    "unified_test_clean": _CODE_DIR / "test" / "unified_test_clean.csv",
    "safe_unified_test_clean": _CODE_DIR / "test" / "safe_unified_test_clean.csv",
}

# Columns treated as label indicators (value 0/1)
LABEL_COLUMNS = {"smoke_label", "leak_label"}

# Numeric feature columns to analyse
FEATURE_COLUMNS = [
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
    "press_pipe_bar",
    "flow_rate_lps",
    "temp_pipe_c",
]

PERCENTILES = [10, 25, 50, 75, 90]


def _percentile(sorted_values: List[float], p: float) -> float:
    """Linear-interpolation percentile (like numpy.percentile with method='linear')."""
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values[0]
    index = (p / 100) * (n - 1)
    lo = int(math.floor(index))
    hi = lo + 1
    if hi >= n:
        return sorted_values[-1]
    frac = index - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def analyse_file(
    path: Path,
    label_col: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Return per-feature stats for rows where *label_col* == 1.

    Returns ``None`` when the label column is absent or has no alarm rows.
    """
    if not path.exists():
        print(f"[WARN] Dataset not found: {path}", file=sys.stderr)
        return None

    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print(f"[WARN] Dataset is empty: {path}", file=sys.stderr)
        return None

    if label_col not in rows[0]:
        return None  # label column absent in this file

    alarm_rows = [r for r in rows if str(r.get(label_col, "0")).strip() == "1"]
    if not alarm_rows:
        print(f"[WARN] No alarm rows (label {label_col}=1) found in {path.name}", file=sys.stderr)
        return None

    stats: Dict[str, Dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        values: List[float] = []
        for row in alarm_rows:
            raw = row.get(col, "")
            if raw in ("", None):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if not values:
            continue
        values.sort()
        entry: Dict[str, float] = {
            "count": len(values),
            "min": values[0],
            "max": values[-1],
        }
        for p in PERCENTILES:
            entry[f"p{p}"] = round(_percentile(values, p), 6)
        stats[col] = entry

    return stats


def _collect_results(
    datasets: Dict[str, Path],
    label_columns: List[str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Collect results: { dataset_name: { label_col: { feature: stats } } }"""
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for name, path in datasets.items():
        label_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for label_col in label_columns:
            stats = analyse_file(path, label_col)
            if stats is not None:
                label_results[label_col] = stats
        if label_results:
            results[name] = label_results
    return results


def _print_stdout(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    sep = "=" * 72
    for ds_name, label_map in results.items():
        print(sep)
        print(f"Dataset: {ds_name}")
        print(sep)
        for label_col, feature_map in label_map.items():
            # Counts can differ per feature when some columns have missing values in alarm rows.
            # We display count from the first feature as an approximate row count indicator.
            n_rows = next(iter(feature_map.values()), {}).get("count", "?")
            print(f"\n  Label column : {label_col} = 1  (alarm rows: {int(n_rows) if n_rows != '?' else n_rows})")
            header = f"  {'Feature':<22} {'count':>6} {'min':>12} {'p10':>12} {'p25':>12} {'p50':>12} {'p75':>12} {'p90':>12} {'max':>12}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for feature, s in feature_map.items():
                row = (
                    f"  {feature:<22}"
                    f" {int(s['count']):>6}"
                    f" {s['min']:>12.4f}"
                    f" {s['p10']:>12.4f}"
                    f" {s['p25']:>12.4f}"
                    f" {s['p50']:>12.4f}"
                    f" {s['p75']:>12.4f}"
                    f" {s['p90']:>12.4f}"
                    f" {s['max']:>12.4f}"
                )
                print(row)
        print()


def _write_csv(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], out_path: Path) -> None:
    fieldnames = ["dataset", "label_col", "feature", "count", "min"] + [f"p{p}" for p in PERCENTILES] + ["max"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ds_name, label_map in results.items():
            for label_col, feature_map in label_map.items():
                for feature, s in feature_map.items():
                    row_data = {"dataset": ds_name, "label_col": label_col, "feature": feature}
                    row_data.update({k: v for k, v in s.items()})
                    writer.writerow(row_data)
    print(f"CSV written to {out_path}", file=sys.stderr)


def _write_json(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"JSON written to {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse alarm-class (label=1) value ranges in sensor datasets. "
            "Helps understand thresholds for ML fire/flood detection and "
            "calibrate synthetic alarm generation."
        )
    )
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        action="append",
        dest="dataset_paths",
        help=(
            "Path to a CSV dataset (can be specified multiple times). "
            "If omitted, both default datasets are used."
        ),
    )
    parser.add_argument(
        "--label",
        metavar="COLUMN",
        action="append",
        dest="label_cols",
        help=(
            "Label column name to analyse (can be specified multiple times). "
            "Defaults to smoke_label and leak_label."
        ),
    )
    parser.add_argument(
        "--output",
        choices=["stdout", "csv", "json"],
        default="stdout",
        help="Output format (default: stdout). Use --out-file to set destination for csv/json.",
    )
    parser.add_argument(
        "--out-file",
        metavar="PATH",
        default=None,
        help="File path for csv/json output. Defaults to alarm_ranges.<ext> in the current directory.",
    )
    args = parser.parse_args()

    # Build dataset dict
    if args.dataset_paths:
        datasets: Dict[str, Path] = {}
        for p in args.dataset_paths:
            path = Path(p)
            datasets[path.stem] = path
    else:
        datasets = DEFAULT_DATASETS

    label_columns = sorted(args.label_cols) if args.label_cols else sorted(LABEL_COLUMNS)

    results = _collect_results(datasets, label_columns)
    if not results:
        print("No alarm data found in the specified datasets.", file=sys.stderr)
        sys.exit(1)

    if args.output == "stdout":
        _print_stdout(results)
    elif args.output == "csv":
        out_path = Path(args.out_file) if args.out_file else Path("alarm_ranges.csv")
        _write_csv(results, out_path)
        _print_stdout(results)
    elif args.output == "json":
        out_path = Path(args.out_file) if args.out_file else Path("alarm_ranges.json")
        _write_json(results, out_path)
        _print_stdout(results)


if __name__ == "__main__":
    main()
