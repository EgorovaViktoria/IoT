"""Intrusion detection tab — rule-based algorithm."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from iot.ui.base_tab import BaseModelTab
from iot.ui.inference import DEFAULT_OUT_DIR

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_CSV = str(_REPO_ROOT / "test_results" / "intrusion_sample.csv")

_SAMPLE_DOC = """\
Expected CSV columns:
  sensor_type  — 0 (IR motion), 1 (door break), 2 (window open)
  sensor_id    — unique integer per sensor
  room_id      — integer room identifier
  room_type    — 0 corridor, 1 hallway, 2 kitchen, 3 living room, 4 bedroom
  batch_id     — integer grouping rows into detection batches
  expected     — (optional) 0/1 ground truth per batch
"""


def _write_sample_csv(path: str) -> None:
    """Write a minimal sample CSV for demo purposes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        "sensor_type,sensor_id,room_id,room_type,batch_id,expected",
        # batch 0: no intrusion (only 1 IR, no perimeter)
        "0,1,101,4,0,0",
        # batch 1: intrusion in bedroom (window open + IR motion)
        "2,7,101,4,1,1",
        "0,9,101,4,1,1",
        # batch 2: corridor — needs perimeter + 1 more, only IR → no alarm
        "0,2,102,0,2,0",
        "0,3,102,0,2,0",
        # batch 3: hallway — 3 independent IR sensors → intrusion
        "0,10,103,1,3,1",
        "0,11,103,1,3,1",
        "0,12,103,1,3,1",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")


class IntrusionTab(BaseModelTab):
    """Tab for rule-based intrusion detection testing.

    Requires a CSV with sensor vectors per batch.
    A sample CSV can be generated with the 'Create sample CSV' button.
    """

    RESULT_COLUMNS = ("batch_id", "y_true", "y_pred", "rooms")

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("Configuration — Intrusion Detection (Rule-based)")
        outer = QVBoxLayout(group)

        form = QFormLayout()
        form.setSpacing(6)
        outer.addLayout(form)

        csv_row, self._csv_edit, _ = self._make_file_row(
            "Dataset CSV:", _SAMPLE_CSV, "Select CSV", "CSV files (*.csv)"
        )
        form.addRow(csv_row)

        sample_btn = QPushButton("📄  Create sample CSV")
        sample_btn.setFixedWidth(180)
        sample_btn.clicked.connect(self._create_sample)
        form.addRow(sample_btn)

        doc_label = QLabel(_SAMPLE_DOC)
        doc_label.setStyleSheet("color: #555; font-size: 11px; font-family: monospace;")
        doc_label.setWordWrap(True)
        outer.addWidget(doc_label)

        return group

    def _create_sample(self):
        path = self._csv_edit.text().strip() or _SAMPLE_CSV
        try:
            _write_sample_csv(path)
            self._append_log(f"Sample CSV written to: {path}")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Failed to create sample CSV: {exc}")

    def _run_inference(self) -> Tuple[dict, pd.DataFrame]:
        from iot.ui.inference import run_intrusion_test  # local import to keep lazy
        return run_intrusion_test(
            csv_path=self._csv_edit.text().strip(),
            out_dir=DEFAULT_OUT_DIR,
            log_fn=self._append_log,
        )
