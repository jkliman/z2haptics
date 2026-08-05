"""
Drag-to-select overlay for choosing the HUD region.

A full-screen translucent window over the live desktop, so the region is chosen
against the actual game HUD rather than by guessing pixel coordinates.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .hud import Region


class _Picker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

        geo = QRect()
        for screen in QGuiApplication.screens():
            geo = geo.united(screen.geometry())
        self.setGeometry(geo)
        self._origin_offset = geo.topLeft()

        self.start = None
        self.end = None
        self.result: Region | None = None

    def mousePressEvent(self, event):
        self.start = event.position().toPoint()
        self.end = self.start
        self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is None:
            return
        self.end = event.position().toPoint()
        rect = QRect(self.start, self.end).normalized()

        if rect.width() >= 8 and rect.height() >= 8:
            # Convert to physical pixels: screen capture works in device pixels
            # while Qt reports logical ones, and they differ under display
            # scaling -- which would otherwise grab the wrong part of the screen.
            ratio = self.devicePixelRatioF()
            self.result = Region(
                left=int((rect.left() + self._origin_offset.x()) * ratio),
                top=int((rect.top() + self._origin_offset.y()) * ratio),
                width=int(rect.width() * ratio),
                height=int(rect.height() * ratio),
            )
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.result = None
            self.close()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))

        if self.start is not None and self.end is not None:
            rect = QRect(self.start, self.end).normalized()
            # Punch the selection clear so the HUD underneath stays readable.
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(rect, Qt.GlobalColor.transparent)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            p.setPen(QPen(QColor(255, 212, 0), 2))
            p.drawRect(rect)
            p.setPen(QColor(255, 255, 255))
            p.drawText(rect.left(), max(rect.top() - 6, 12),
                       f"{rect.width()} x {rect.height()}")
        else:
            p.setPen(QColor(255, 255, 255))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Drag a box around the weapon name on the HUD.\n"
                       "Esc to cancel.")
        p.end()


def pick_region(app) -> Region | None:
    picker = _Picker()
    picker.show()
    picker.raise_()
    picker.activateWindow()
    app.exec()
    return picker.result
