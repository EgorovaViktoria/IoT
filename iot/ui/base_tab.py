"""Base class for model-testing tabs."""
from __future__ import annotations

import traceback
from typing import Callable, Optional, Tuple

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class _Worker(QThread):
    """Runs the inference function in a background thread."""

    finished = pyqtSignal(object, object)   # (metrics, result_df)
    log_msg = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, fn: Callable, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            metrics, df = self._fn()
            self.finished.emit(metrics, df)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


class BaseModelTab(QWidget):
    """Shared skeleton for Smoke / Leak / Gas / Intrusion tabs.

    Sub-classes must implement:
      - _build_config_group() -> QGroupBox  (extra controls: model path, etc.)
      - _run_inference() -> (metrics, result_df)
    """

    # Column to show in the preview table (overridden per tab)
    RESULT_COLUMNS: Tuple[str, ...] = ("y_true_raw", "y_true_win", "y_pred", "y_proba")
    PREVIEW_ROWS = 200

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[_Worker] = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Top: config section provided by sub-class
        config_group = self._build_config_group()
        root.addWidget(config_group)

        # Run button
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run test")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background: #2e7d32; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #388e3c; }"
            "QPushButton:disabled { background: #888; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addStretch()
        btn_row.addWidget(self._run_btn)
        root.addLayout(btn_row)

        # Splitter: metrics + log on top, table on bottom
        splitter = QSplitter()
        splitter.setOrientation(0x2)  # Vertical

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Metrics box
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self._metrics_label = QLabel("—")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setStyleSheet("font-family: monospace; font-size: 13px;")
        metrics_layout.addWidget(self._metrics_label)
        top_layout.addWidget(metrics_group, 1)

        # Log box
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setStyleSheet("font-family: monospace; font-size: 11px;")
        log_layout.addWidget(self._log)
        top_layout.addWidget(log_group, 2)

        splitter.addWidget(top_widget)

        # Preview table
        table_group = QGroupBox(f"Results preview (first {self.PREVIEW_ROWS} rows)")
        table_layout = QVBoxLayout(table_group)
        self._table = QTableWidget(0, len(self.RESULT_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(self.RESULT_COLUMNS))
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        table_layout.addWidget(self._table)
        splitter.addWidget(table_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Sub-class hooks
    # ------------------------------------------------------------------

    def _build_config_group(self) -> QGroupBox:  # noqa: D102
        raise NotImplementedError

    def _run_inference(self) -> Tuple[dict, pd.DataFrame]:  # noqa: D102
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared file-chooser helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_file_row(
        label_text: str,
        default: str,
        caption: str,
        file_filter: str,
    ) -> Tuple[QWidget, QLineEdit, QPushButton]:
        """Build a labelled text-field + Browse button row."""
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setFixedWidth(110)
        hbox.addWidget(lbl)
        edit = QLineEdit(default)
        hbox.addWidget(edit, 1)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(90)
        hbox.addWidget(btn)

        def _browse():
            path, _ = QFileDialog.getOpenFileName(
                container, caption, edit.text(), file_filter
            )
            if path:
                edit.setText(path)

        btn.clicked.connect(_browse)
        return container, edit, btn

    # ------------------------------------------------------------------
    # Run / callbacks
    # ------------------------------------------------------------------

    def _on_run(self):
        self._run_btn.setEnabled(False)
        self._log.clear()
        self._metrics_label.setText("Running…")
        self._table.setRowCount(0)

        self._worker = _Worker(self._run_inference, parent=self)
        self._worker.log_msg.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, metrics: dict, result_df: pd.DataFrame):
        self._run_btn.setEnabled(True)
        self._show_metrics(metrics)
        self._populate_table(result_df)

    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._metrics_label.setText("❌ Error — see log.")
        self._append_log(f"\n{'='*60}\n{msg}")
        QMessageBox.critical(self, "Error", msg[:500])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_log(self, text: str):
        self._log.appendPlainText(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _show_metrics(self, metrics: dict):
        lines = []
        for key in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
            if key in metrics:
                lines.append(f"{key:<12}: {metrics[key]:.4f}")
        lines.append(f"{'threshold':<12}: {metrics.get('threshold', '—')}")
        lines.append(f"{'test_size':<12}: {metrics.get('test_size', '—')}")
        lines.append(f"{'positives':<12}: {metrics.get('positives', '—')}")
        self._metrics_label.setText("\n".join(lines))

    def _populate_table(self, df: pd.DataFrame):
        cols = [c for c in self.RESULT_COLUMNS if c in df.columns]
        rows = min(len(df), self.PREVIEW_ROWS)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(rows)
        for r in range(rows):
            for c_idx, col in enumerate(cols):
                val = df.iloc[r][col]
                if isinstance(val, float):
                    text = f"{val:.4f}"
                else:
                    text = str(val)
                self._table.setItem(r, c_idx, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
