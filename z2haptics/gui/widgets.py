"""Reusable widgets: live band meters and the per-band tuning editor."""

from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..analysis import Band

ACCENT = QColor(255, 212, 0)       # Swiftpoint yellow
ACCENT_DIM = QColor(120, 100, 0)
GATE_LINE = QColor(200, 90, 90)
FLASH = QColor(255, 255, 255)


class BandMeter(QWidget):
    """A level bar for one band, with a gate marker and an onset flash.

    Levels are drawn on a dB scale. Linear looks almost dead for quiet content
    because band level is diluted across the band's bins, making it useless for
    judging whether a gate is set sensibly -- which is the whole point of this
    widget.
    """

    FLOOR_DB = -80.0

    def __init__(self, band_name: str, parent=None):
        super().__init__(parent)
        self.band_name = band_name
        self.level = 0.0
        self.gate = 0.0
        self.is_open = False
        self._flash_until = 0.0
        self._flash_strength = 0
        self.setMinimumHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, level: float, gate: float, is_open: bool) -> None:
        self.level, self.gate, self.is_open = level, gate, is_open
        self.update()

    def flash(self, strength: int) -> None:
        self._flash_until = time.time() + 0.22
        self._flash_strength = strength
        self.update()

    @staticmethod
    def _to_db(value: float) -> float:
        return 20.0 * math.log10(max(value, 1e-9))

    def _fraction(self, value: float) -> float:
        db = self._to_db(value)
        return max(0.0, min(1.0, (db - self.FLOOR_DB) / (0.0 - self.FLOOR_DB)))

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()

        bar_x, bar_w = 96, max(1, w - 96 - 52)
        bar_y, bar_h = 5, h - 10

        p.fillRect(0, 0, w, h, self.palette().window().color())
        p.setPen(self.palette().text().color())
        p.drawText(0, 0, 90, h, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.band_name)

        track = self.palette().alternateBase().color()
        p.fillRect(bar_x, bar_y, bar_w, bar_h, track)

        filled = int(bar_w * self._fraction(self.level))
        if filled > 0:
            p.fillRect(bar_x, bar_y, filled, bar_h, ACCENT if self.is_open else ACCENT_DIM)

        if self.gate > 0:
            gx = bar_x + int(bar_w * self._fraction(self.gate))
            p.setPen(QPen(GATE_LINE, 2))
            p.drawLine(gx, bar_y - 1, gx, bar_y + bar_h + 1)

        if time.time() < self._flash_until:
            p.fillRect(bar_x, bar_y, bar_w, bar_h, QColor(255, 255, 255, 60))
            p.setPen(FLASH)
            p.drawText(bar_x + bar_w + 6, 0, 46, h,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"{self._flash_strength}")
        else:
            p.setPen(self.palette().mid().color())
            p.drawText(bar_x + bar_w + 6, 0, 46, h,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"{self._to_db(self.level):.0f}dB")
        p.end()


class BandEditor(QGroupBox):
    """Editor for one band, with its live meter and rejection readout inline.

    The activity display sits with the controls rather than on a separate tab so
    a threshold change and its effect are visible at the same time. Tuning five
    interacting checks by adjusting one, switching tabs, watching, and switching
    back does not work.
    """

    changed = Signal()
    removed = Signal(object)

    def __init__(self, band: Band, parent=None):
        super().__init__(band.name, parent)
        self.band = band
        self.setCheckable(True)
        self.setChecked(band.enabled)

        outer = QVBoxLayout(self)

        # -- live activity, above the controls that shape it
        self.meter = BandMeter(band.name)
        outer.addWidget(self.meter)

        self.readout = QLabel("waiting for audio...")
        self.readout.setStyleSheet("font-family: Consolas, monospace; color: gray;")
        outer.addWidget(self.readout)

        self.advice = QLabel("")
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet("color: #b48ead;")
        outer.addWidget(self.advice)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._suspend = True

        # -- frequency range
        row = QHBoxLayout()
        self.low = QDoubleSpinBox(); self.low.setRange(10, 20000); self.low.setSuffix(" Hz")
        self.high = QDoubleSpinBox(); self.high.setRange(10, 22000); self.high.setSuffix(" Hz")
        self.low.setValue(band.low_hz); self.high.setValue(band.high_hz)
        row.addWidget(self.low); row.addWidget(QLabel("to")); row.addWidget(self.high)
        form.addRow("Frequency", self._wrap(row))

        # -- detection
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(1.0, 10.0); self.sensitivity.setSingleStep(0.05)
        self.sensitivity.setValue(band.sensitivity)
        self.sensitivity.setToolTip(
            "Flux must exceed this multiple of the rolling median to count as an "
            "onset. Higher = fewer, more certain detections.")
        form.addRow("Sensitivity", self.sensitivity)

        self.gate = QDoubleSpinBox()
        self.gate.setRange(0.0, 1.0); self.gate.setDecimals(5)
        self.gate.setSingleStep(0.0005); self.gate.setValue(band.gate)
        self.gate.setToolTip("Absolute level floor. Below this the band is treated as silent.")
        form.addRow("Gate", self.gate)

        self.refractory = QDoubleSpinBox()
        self.refractory.setRange(0, 2000); self.refractory.setSuffix(" ms")
        self.refractory.setValue(band.refractory_ms)
        self.refractory.setToolTip("Minimum spacing between onsets in this band.")
        form.addRow("Refractory", self.refractory)

        self.min_share = QDoubleSpinBox()
        self.min_share.setRange(0.0, 1.0); self.min_share.setSingleStep(0.05)
        self.min_share.setValue(band.min_share)
        self.min_share.setToolTip(
            "Minimum share of the frame's total flux. Raise this if a sharp sound "
            "elsewhere keeps falsely triggering this band. 0 disables the check.")
        form.addRow("Min share", self.min_share)

        self.min_flatness = QDoubleSpinBox()
        self.min_flatness.setRange(0.0, 1.0); self.min_flatness.setSingleStep(0.05)
        self.min_flatness.setValue(band.min_flatness)
        self.min_flatness.setToolTip(
            "Reject tonal content. Spectral flatness is ~1 for noise-like sounds "
            "(gunfire, explosions, impacts) and near 0 for played notes. Measured "
            "on real material: gunshots ~0.90, music ~0.11.\n\n"
            "Raise this when music keeps triggering the band. Set it to 0 for "
            "bands that SHOULD react to tonal content, like a music profile's "
            "kick and snare, or engine note in a racing profile.\n\n"
            "0.45 is a good starting point for gunfire.")
        form.addRow("Min flatness", self.min_flatness)

        self.max_rate = QDoubleSpinBox()
        self.max_rate.setRange(0.0, 60.0); self.max_rate.setSingleStep(1.0)
        self.max_rate.setValue(band.max_rate)
        self.max_rate.setToolTip(
            "Cap on how often this band may win, in pulses/sec. 0 = unlimited.\n\n"
            "The motor is a single actuator, so a constantly-firing band starves "
            "everything else. Capping a busy band hands those frames to quieter, "
            "more informative ones.")
        form.addRow("Max rate", self.max_rate)

        self.background_subtraction = QDoubleSpinBox()
        self.background_subtraction.setRange(0.0, 1.0)
        self.background_subtraction.setSingleStep(0.1)
        self.background_subtraction.setValue(band.background_subtraction)
        self.background_subtraction.setToolTip(
            "Remove the slowly-learned steady background before measuring. Helps "
            "against constant drones (engine note, ambience).\n\n"
            "Measured as unhelpful for music in shooters once the adaptive "
            "threshold was fixed, so the FPS profiles leave it off.")
        form.addRow("Background sub.", self.background_subtraction)

        # -- pulse shaping
        self.duration = QSpinBox()
        self.duration.setRange(1, 2000); self.duration.setSuffix(" ms")
        self.duration.setValue(band.duration_ms)
        self.duration.setToolTip("How long the motor runs for this band's pulses.")
        form.addRow("Duration", self.duration)

        row = QHBoxLayout()
        self.smin = QSpinBox(); self.smin.setRange(0, 100); self.smin.setValue(band.strength_min)
        self.smax = QSpinBox(); self.smax.setRange(0, 100); self.smax.setValue(band.strength_max)
        row.addWidget(self.smin); row.addWidget(QLabel("to")); row.addWidget(self.smax)
        form.addRow("Strength", self._wrap(row))

        row = QHBoxLayout()
        self.floor_db = QDoubleSpinBox()
        self.floor_db.setRange(-140, 0); self.floor_db.setSuffix(" dB")
        self.floor_db.setValue(band.level_floor_db)
        self.ceil_db = QDoubleSpinBox()
        self.ceil_db.setRange(-140, 0); self.ceil_db.setSuffix(" dB")
        self.ceil_db.setValue(band.level_ceil_db)
        row.addWidget(self.floor_db); row.addWidget(QLabel("to")); row.addWidget(self.ceil_db)
        w = self._wrap(row)
        w.setToolTip(
            "Loudness window mapped onto the strength range. A quiet event at the "
            "floor pulses at minimum strength; one at the ceiling pulses at maximum. "
            "Widen it if everything feels the same.")
        form.addRow("Loudness window", w)

        self.priority = QSpinBox()
        self.priority.setRange(0, 10); self.priority.setValue(band.priority)
        self.priority.setToolTip(
            "Nudges which band wins when several fire together. Deliberately only a "
            "thumb on the scale -- it cannot override a much stronger detection.")
        form.addRow("Priority", self.priority)

        outer.addLayout(form)

        btns = QHBoxLayout()
        btns.addStretch(1)
        remove = QPushButton("Remove band")
        remove.clicked.connect(lambda: self.removed.emit(self))
        btns.addWidget(remove)
        outer.addLayout(btns)

        for widget in (self.low, self.high, self.sensitivity, self.gate, self.refractory,
                       self.min_share, self.min_flatness, self.max_rate,
                       self.background_subtraction, self.duration, self.smin, self.smax,
                       self.floor_db, self.ceil_db, self.priority):
            widget.valueChanged.connect(self._on_change)
        self.toggled.connect(self._on_change)

        self._suspend = False

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    # -- live feedback --------------------------------------------------------

    def set_activity(self, activity: dict) -> None:
        """Update the inline meter and readout from one poll of engine state."""
        level = activity.get("level", 0.0)
        self.meter.set_level(level, self.band.gate, activity.get("open", False))

        rate = activity.get("rate", 0.0)
        flatness = activity.get("flatness", 0.0)
        sent = activity.get("sent", 0)
        rej = activity.get("rejections") or {}

        self.readout.setText(
            f"{rate:5.1f} onsets/s   sent {sent:<5} "
            f"flatness {flatness:4.2f}   fired {rej.get('accepted', 0)}"
        )
        self.advice.setText(self._advice(rate, rej, activity))

    def _advice(self, rate: float, rej: dict, activity: dict) -> str:
        """Say which knob to reach for, based on what is actually happening."""
        if not rej or not rej.get("frames"):
            return ""

        if rej.get("accepted", 0) == 0:
            if activity.get("peak", 0.0) < self.band.gate:
                return "Never reached the gate — lower Gate, or this band covers " \
                       "frequencies the audio does not contain."
            worst, _ = activity.get("dominant", ("threshold", 0))
            return {
                "threshold": "Blocked by Sensitivity — lower it.",
                "refractory": "Blocked by Refractory — events are closer together "
                              "than the spacing allows.",
                "share": "Blocked by Min share — another band dominates the frame.",
                "flatness": "Blocked by Min flatness — this content is too tonal.",
            }.get(worst, "Nothing fired.")

        if rate > 8:
            return "Firing very often — the motor will blur these together. " \
                   "Raise Sensitivity, Min flatness, or Refractory."
        if rate > 4:
            return "Busy. Consider a Max rate cap so this band cannot starve others."
        return ""

    def reset_meter(self) -> None:
        self.readout.setText("waiting for audio...")
        self.advice.setText("")

    def flash(self, strength: int) -> None:
        self.meter.flash(strength)

    def tick(self) -> None:
        self.meter.update()

    def _on_change(self, *_) -> None:
        if self._suspend:
            return
        self.apply_to_band()
        self.changed.emit()

    def apply_to_band(self) -> None:
        b = self.band
        b.low_hz = self.low.value()
        b.high_hz = max(self.high.value(), self.low.value() + 1)
        b.sensitivity = self.sensitivity.value()
        b.gate = self.gate.value()
        b.refractory_ms = self.refractory.value()
        b.min_share = self.min_share.value()
        b.min_flatness = self.min_flatness.value()
        b.max_rate = self.max_rate.value()
        b.background_subtraction = self.background_subtraction.value()
        b.duration_ms = self.duration.value()
        b.strength_min = min(self.smin.value(), self.smax.value())
        b.strength_max = max(self.smin.value(), self.smax.value())
        b.level_floor_db = min(self.floor_db.value(), self.ceil_db.value())
        b.level_ceil_db = max(self.floor_db.value(), self.ceil_db.value())
        b.priority = self.priority.value()
        b.enabled = self.isChecked()
