"""Entry point for the IoT Model Testing UI.

Usage:
    python app.py
"""
import sys

from PyQt5.QtWidgets import QApplication

from iot.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
