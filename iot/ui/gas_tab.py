"""Gas leak detection tab — cumulants-based statistical algorithm."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
)

from iot.ui.base_tab import BaseModelTab
from iot.ui.inference import (
    DEFAULT_CSV_PATH,
    DEFAULT_OUT_DIR,
    run_gas_test,
)


class GasTab(BaseModelTab):
    """Tab for gas leak detection using the cumulants algorithm.

    This algorithm does not use a sklearn pkl model — it uses a statistical
    cumulants method on the pipe-sensor signal history, mirroring the logic in
    emergency_system.detect_gas_leak.  Ground truth is 'leak_label' from the
    unified CSV (the closest available label for pipe anomalies).
    """

    RESULT_COLUMNS = ("y_true_raw", "y_pred", "y_score")

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("Configuration — Gas Leak (Cumulants)")
        form = QFormLayout(group)
        form.setSpacing(6)

        csv_row, self._csv_edit, _ = self._make_file_row(
            "Dataset CSV:", DEFAULT_CSV_PATH, "Select CSV", "CSV files (*.csv)"
        )
        form.addRow(csv_row)

        note = QLabel(
            "ℹ  Uses columns: press_pipe_bar, flow_rate_lps, temp_pipe_c.\n"
            "   Ground truth: leak_label.  No .pkl model required."
        )
        note.setStyleSheet("color: #555; font-size: 11px;")
        note.setWordWrap(True)
        form.addRow(note)

        return group

    def _run_inference(self) -> Tuple[dict, pd.DataFrame]:
        return run_gas_test(
            csv_path=self._csv_edit.text().strip(),
            out_dir=DEFAULT_OUT_DIR,
            log_fn=self._append_log,
        )
