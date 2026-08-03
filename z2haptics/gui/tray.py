"""System tray icon and menu."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

ACCENT = QColor(255, 212, 0)
IDLE = QColor(120, 120, 120)


def make_icon(active: bool, size: int = 64) -> QIcon:
    """Draw the tray icon: a mouse outline with radiating waves when active.

    Generated rather than shipped as a file so it stays crisp at any DPI and the
    active/idle states cannot drift out of sync with each other.
    """
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)

    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    colour = ACCENT if active else IDLE

    # Mouse body
    p.setPen(QPen(colour, max(2, size // 16)))
    body = QRectF(size * 0.30, size * 0.22, size * 0.40, size * 0.56)
    p.drawRoundedRect(body, size * 0.18, size * 0.18)
    p.drawLine(int(size * 0.50), int(size * 0.24), int(size * 0.50), int(size * 0.44))

    if active:
        # Vibration waves either side
        p.setPen(QPen(colour, max(1, size // 22)))
        for i, r in enumerate((0.10, 0.18)):
            span = int(size * r)
            for direction in (-1, 1):
                cx = size * (0.5 + direction * 0.30)
                rect = QRectF(cx - span / 2, size * 0.5 - span / 2, span, span)
                start = 60 * 16 if direction > 0 else 240 * 16
                p.drawArc(rect, start, 60 * 16)
    p.end()
    return QIcon(pix)


class TrayIcon(QSystemTrayIcon):
    def __init__(self, controller, config, window, app, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.config = config
        self.window = window
        self.app = app

        self.setIcon(make_icon(False))
        self.setToolTip("z2haptics — stopped")

        self.menu = QMenu()

        self.toggle_action = QAction("Start haptics", self.menu)
        self.toggle_action.triggered.connect(self._toggle)
        self.menu.addAction(self.toggle_action)

        self.menu.addSeparator()
        self.profile_menu = self.menu.addMenu("Profile")
        self._profile_group = QActionGroup(self.menu)
        self._profile_group.setExclusive(True)

        self.menu.addSeparator()
        test = QAction("Test rumble", self.menu)
        test.triggered.connect(lambda: self.controller.test_pulse(110, 90))
        self.menu.addAction(test)

        settings = QAction("Settings...", self.menu)
        settings.triggered.connect(self.show_window)
        self.menu.addAction(settings)

        self.menu.addSeparator()
        quit_action = QAction("Quit", self.menu)
        quit_action.triggered.connect(self._quit)
        self.menu.addAction(quit_action)

        self.setContextMenu(self.menu)
        self.activated.connect(self._on_activated)

        controller.statusChanged.connect(self._on_status)
        controller.profileChanged.connect(self._on_profile_changed)

        self.rebuild_profile_menu()

    # -- menu -----------------------------------------------------------------

    def rebuild_profile_menu(self) -> None:
        self.profile_menu.clear()
        for action in self._profile_group.actions():
            self._profile_group.removeAction(action)

        for name in self.controller.profiles:
            action = QAction(name, self.profile_menu, checkable=True)
            action.setChecked(name == self.config.active_profile)
            action.triggered.connect(lambda _, n=name: self.controller.set_profile(n))
            self._profile_group.addAction(action)
            self.profile_menu.addAction(action)

    def _on_profile_changed(self, name: str) -> None:
        for action in self._profile_group.actions():
            action.setChecked(action.text() == name)
        self._refresh_tooltip()

    def _on_status(self, message: str, running: bool) -> None:
        self.setIcon(make_icon(running))
        self.toggle_action.setText("Stop haptics" if running else "Start haptics")
        self._status_message = message
        self._refresh_tooltip()

    def _refresh_tooltip(self) -> None:
        running = self.controller.running
        state = "running" if running else "stopped"
        profile = self.config.active_profile
        self.setToolTip(f"z2haptics — {state}\nProfile: {profile}")

    def _toggle(self) -> None:
        if self.controller.running:
            self.controller.stop()
            self.notify("Haptics stopped")
        else:
            if self.controller.start():
                self.notify(f"Haptics running — {self.config.active_profile}")

    def notify(self, message: str) -> None:
        if self.config.show_notifications and self.supportsMessages():
            self.showMessage("z2haptics", message, make_icon(True), 2500)

    def _on_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_window()

    def show_window(self) -> None:
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def _quit(self) -> None:
        self.controller.shutdown()
        self.hide()
        self.app.quit()
