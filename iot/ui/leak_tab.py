"""Water leak model testing tab."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QSpinBox,
)

from iot.ui.base_tab import BaseModelTab
from iot.ui.inference import (
    DEFAULT_CSV_PATH,
    DEFAULT_LEAK_MODEL,
    DEFAULT_OUT_DIR,
    run_leak_test,
)


class LeakTab(BaseModelTab):
    """Tab for testing the water-leak detection model."""

    RESULT_COLUMNS = ("y_true_raw", "y_true_win", "y_pred", "y_proba")

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("Configuration — Water Leak")
        form = QFormLayout(group)
        form.setSpacing(6)

        csv_row, self._csv_edit, _ = self._make_file_row(
            "Dataset CSV:", DEFAULT_CSV_PATH, "Select CSV", "CSV files (*.csv)"
        )
        form.addRow(csv_row)

        model_row, self._model_edit, _ = self._make_file_row(
            "Model (.pkl):", DEFAULT_LEAK_MODEL, "Select model", "Model files (*.pkl)"
        )
        form.addRow(model_row)

        self._window_spin = QSpinBox()
        self._window_spin.setRange(1, 100)
        self._window_spin.setValue(5)
        form.addRow("Window size:", self._window_spin)

        return group

    def _run_inference(self) -> Tuple[dict, pd.DataFrame]:
        return run_leak_test(
            model_path=self._model_edit.text().strip(),
            csv_path=self._csv_edit.text().strip(),
            window_size=self._window_spin.value(),
            out_dir=DEFAULT_OUT_DIR,
            log_fn=self._append_log,
        )
