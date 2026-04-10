"""Main application window with tab-based navigation."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from iot.ui.gas_tab import GasTab
from iot.ui.intrusion_tab import IntrusionTab
from iot.ui.leak_tab import LeakTab
from iot.ui.smoke_tab import SmokeTab


class MainWindow(QMainWindow):
    """IoT Emergency System — Model Testing UI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IoT Emergency System — Model Testing")
        self.resize(1100, 750)
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(4)

        # Header
        header = QLabel("🔬  IoT Emergency System — Model Testing")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 6px; "
            "border-bottom: 1px solid #ccc;"
        )
        layout.addWidget(header)

        # Tab widget
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setDocumentMode(True)

        self._smoke_tab = SmokeTab()
        self._leak_tab = LeakTab()
        self._gas_tab = GasTab()
        self._intrusion_tab = IntrusionTab()

        tabs.addTab(self._smoke_tab, "🔥  Smoke / Fire")
        tabs.addTab(self._leak_tab, "💧  Water Leak")
        tabs.addTab(self._gas_tab, "⚗️  Gas Leak")
        tabs.addTab(self._intrusion_tab, "🚪  Intrusion")

        layout.addWidget(tabs, 1)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Select a tab and click ▶ Run test to start."
        )
