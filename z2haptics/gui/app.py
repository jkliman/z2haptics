"""GUI entry point: tray icon plus settings window."""

from __future__ import annotations

import argparse
import logging
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from .config import AppConfig
from .controller import EngineController
from .tray import TrayIcon, make_icon
from .window import SettingsWindow

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="z2haptics-gui")
    ap.add_argument("--minimised", "--minimized", action="store_true",
                    dest="minimised", help="start hidden in the tray")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("z2haptics")
    app.setWindowIcon(make_icon(True))

    # Closing the settings window hides it to tray, so Qt must not treat that as
    # the end of the program.
    app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, "z2haptics",
                             "No system tray is available on this desktop.")
        return 1

    config = AppConfig.load()
    controller = EngineController(config)
    window = SettingsWindow(controller, config)
    tray = TrayIcon(controller, config, window, app)
    tray.show()

    controller.profileChanged.connect(lambda _: tray.rebuild_profile_menu())

    def startup() -> None:
        if config.start_engine_on_launch:
            if controller.start():
                tray.notify(f"Haptics running — {config.active_profile}")
            else:
                tray.show_window()
                tray.notify("Could not start — see Settings")

    QTimer.singleShot(300, startup)

    if not (args.minimised or config.start_minimised):
        window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
