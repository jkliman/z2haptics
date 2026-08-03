"""The settings window: live status, audio device, profile tuning, testing, general."""

from __future__ import annotations

import copy
import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis import Band
from ..audio import list_devices
from ..profiles import USER_DIR, save_profile
from .config import get_autostart, set_autostart
from .widgets import BandEditor, BandMeter

log = logging.getLogger(__name__)


class SettingsWindow(QWidget):
    def __init__(self, controller, config, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.config = config
        self._meters: dict[str, BandMeter] = {}
        self._band_editors: list[BandEditor] = []
        self._dirty = False

        self.setWindowTitle("z2haptics")
        self.resize(760, 720)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_status_tab(), "Status")
        self.tabs.addTab(self._build_profile_tab(), "Profile")
        self.tabs.addTab(self._build_audio_tab(), "Audio")
        self.tabs.addTab(self._build_test_tab(), "Test")
        self.tabs.addTab(self._build_general_tab(), "General")
        layout.addWidget(self.tabs)

        controller.statusChanged.connect(self._on_status)
        controller.statsUpdated.connect(self._on_stats)
        controller.levelsUpdated.connect(self._on_levels)
        controller.onsetDetected.connect(self._on_onset)
        controller.profileChanged.connect(self._on_profile_changed)
        controller.errorRaised.connect(self._on_error)
        controller.testResult.connect(self._on_test_result)
        controller.apiStatus.connect(self._on_api_status)
        controller.x1ProfilesLoaded.connect(self._on_x1_profiles)
        controller.bandActivity.connect(self._on_band_activity)

        # Repaint meters so onset flashes decay even when no audio is arriving.
        self._repaint = QTimer(self)
        self._repaint.setInterval(60)
        self._repaint.timeout.connect(self._tick_meters)
        self._repaint.start()

        self._rebuild_meters()
        self._on_status("Stopped", controller.running)

    # -- Status ---------------------------------------------------------------

    def _build_status_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        self.status_label = QLabel("Stopped")
        f = self.status_label.font(); f.setPointSize(f.pointSize() + 2); f.setBold(True)
        self.status_label.setFont(f)
        v.addWidget(self.status_label)

        row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._toggle_engine)
        row.addWidget(self.start_btn)

        self.profile_combo = QComboBox()
        self.profile_combo.currentTextChanged.connect(self._on_profile_selected)
        row.addWidget(QLabel("Profile:"))
        row.addWidget(self.profile_combo, 1)
        v.addLayout(row)

        meters_box = QGroupBox("Band activity")
        self.meters_layout = QVBoxLayout(meters_box)
        self.meters_hint = QLabel(
            "Bar = band level, red line = gate, flash = onset fired.\n"
            "If a band never crosses its gate it can never trigger; if it sits "
            "above the gate constantly, raise the gate."
        )
        self.meters_hint.setWordWrap(True)
        self.meters_hint.setStyleSheet("color: gray;")
        self.meters_layout.addWidget(self.meters_hint)
        v.addWidget(meters_box)

        per_band_box = QGroupBox("Per band")
        pb = QVBoxLayout(per_band_box)
        self.per_band_label = QLabel("(not running)")
        self.per_band_label.setStyleSheet("font-family: Consolas, monospace;")
        pb.addWidget(self.per_band_label)
        hint = QLabel(
            "detected = onsets found | won = beat other bands | lost = outranked\n"
            "capped = blocked by this band's max rate | sent = reached the motor\n\n"
            "If an event never feels right, this says which stage lost it: low "
            "'detected' is a gate or sensitivity problem, high 'lost' means "
            "another band is winning, high 'capped' means its own rate limit."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        pb.addWidget(hint)
        v.addWidget(per_band_box)

        stats_box = QGroupBox("Counters")
        sf = QFormLayout(stats_box)
        self.stat_labels = {}
        for key, label in [
            ("onsets", "Onsets detected"),
            ("pulses", "Pulses queued"),
            ("sent", "Pulses sent"),
            ("dropped_rate", "Dropped (rate limit)"),
            ("dropped_queue", "Dropped (queue full)"),
            ("switches", "Profile switches"),
            ("errors", "Errors"),
        ]:
            lbl = QLabel("0")
            self.stat_labels[key] = lbl
            sf.addRow(label, lbl)
        v.addWidget(stats_box)
        v.addStretch(1)
        return w

    def _rebuild_meters(self) -> None:
        for meter in self._meters.values():
            meter.setParent(None)
            meter.deleteLater()
        self._meters.clear()

        profile = self.controller.active_profile
        if profile is None:
            return
        for band in profile.bands:
            meter = BandMeter(band.name)
            self._meters[band.name] = meter
            self.meters_layout.addWidget(meter)

    def _tick_meters(self) -> None:
        for meter in self._meters.values():
            meter.update()
        for editor in self._band_editors:
            editor.tick()

    def _on_levels(self, levels: dict) -> None:
        for name, (level, gate, is_open) in levels.items():
            meter = self._meters.get(name)
            if meter is not None:
                meter.set_level(level, gate, is_open)

    def _on_onset(self, band: str, strength: int, _norm: float) -> None:
        meter = self._meters.get(band)
        if meter is not None:
            meter.flash(strength)
        for editor in self._band_editors:
            if editor.band.name == band:
                editor.flash(strength)

    def _on_band_activity(self, activity: dict) -> None:
        for editor in self._band_editors:
            data = activity.get(editor.band.name)
            if data:
                editor.set_activity(data)

    def _on_stats(self, stats: dict) -> None:
        for key, lbl in self.stat_labels.items():
            lbl.setText(str(stats.get(key, 0)))

        per_band = stats.get("per_band") or {}
        if not per_band:
            self.per_band_label.setText("(no onsets yet)")
            return
        header = f"{'band':<12}{'detected':>9}{'won':>7}{'lost':>7}{'capped':>8}{'sent':>7}"
        rows = [header, "-" * len(header)]
        for name, b in per_band.items():
            rows.append(
                f"{name:<12}{b['detected']:>9}{b['won']:>7}{b['lost']:>7}"
                f"{b['capped']:>8}{b['queued']:>7}"
            )
        self.per_band_label.setText("\n".join(rows))

    def _on_status(self, message: str, running: bool) -> None:
        self.status_label.setText(message)
        self.start_btn.setText("Stop" if running else "Start")

    def _on_error(self, message: str) -> None:
        self.status_label.setText(message)

    def _toggle_engine(self) -> None:
        if self.controller.running:
            self.controller.stop()
        else:
            self.controller.start()

    # -- Profile --------------------------------------------------------------

    def _build_profile_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        row = QHBoxLayout()
        self.profile_combo2 = QComboBox()
        self.profile_combo2.currentTextChanged.connect(self._on_profile_selected)
        row.addWidget(QLabel("Editing:"))
        row.addWidget(self.profile_combo2, 1)
        reset_btn = QPushButton("Reset counters")
        reset_btn.setToolTip(
            "Zero the onset tallies so the next change is judged on fresh numbers "
            "instead of totals dominated by the old settings.\n\n"
            "Detection state (noise floor, flux history) is kept, so the reading "
            "settles again immediately.")
        reset_btn.clicked.connect(self._reset_counters)
        row.addWidget(reset_btn)
        reload_btn = QPushButton("Reload from disk")
        reload_btn.clicked.connect(self._reload_profiles)
        row.addWidget(reload_btn)
        v.addLayout(row)

        live_hint = QLabel(
            "Each band below shows its own live meter: bar = level, red line = "
            "gate, flash = onset. Edits apply immediately, so change a value and "
            "watch that band react. Hit Reset counters after each change."
        )
        live_hint.setWordWrap(True)
        live_hint.setStyleSheet("color: gray;")
        v.addWidget(live_hint)

        master = QGroupBox("Master")
        mf = QFormLayout(master)

        self.strength_scale = QDoubleSpinBox()
        self.strength_scale.setRange(0.0, 2.0); self.strength_scale.setSingleStep(0.05)
        self.strength_scale.setToolTip("Multiplies every pulse strength in this profile.")
        self.strength_scale.valueChanged.connect(self._on_profile_edited)
        mf.addRow("Strength scale", self.strength_scale)

        self.min_gap = QDoubleSpinBox()
        self.min_gap.setRange(0, 1000); self.min_gap.setSuffix(" ms")
        self.min_gap.setToolTip("Hard refractory period between any two pulses.")
        self.min_gap.valueChanged.connect(self._on_profile_edited)
        mf.addRow("Min gap", self.min_gap)

        self.max_pulses = QDoubleSpinBox()
        self.max_pulses.setRange(1, 60); self.max_pulses.setSuffix(" /sec")
        self.max_pulses.setToolTip("Sliding-window cap on pulses per second.")
        self.max_pulses.valueChanged.connect(self._on_profile_edited)
        mf.addRow("Max pulse rate", self.max_pulses)

        self.max_duty = QDoubleSpinBox()
        self.max_duty.setRange(0.05, 1.0); self.max_duty.setSingleStep(0.05)
        self.max_duty.setToolTip(
            "Maximum fraction of wall-clock the motor may be driven. The main "
            "protection against cooking the actuator during sustained action.")
        self.max_duty.valueChanged.connect(self._on_profile_edited)
        mf.addRow("Max duty cycle", self.max_duty)

        self.processes = QPlainTextEdit()
        self.processes.setMaximumHeight(60)
        self.processes.setPlaceholderText("bf6.exe, cs2.exe")
        self.processes.setToolTip("Executables that auto-select this profile, comma or newline separated.")
        self.processes.textChanged.connect(self._on_profile_edited)
        mf.addRow("Processes", self.processes)

        self.x1_profile = QComboBox()
        self.x1_profile.setEditable(True)
        self.x1_profile.setToolTip(
            "Optionally switch the Control Panel's own profile too, so button "
            "mappings follow the game.")
        self.x1_profile.currentTextChanged.connect(self._on_profile_edited)
        mf.addRow("X1 profile", self.x1_profile)

        v.addWidget(master)

        bands_box = QGroupBox("Bands")
        bl = QVBoxLayout(bands_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.bands_container = QWidget()
        self.bands_layout = QVBoxLayout(self.bands_container)
        self.bands_layout.addStretch(1)
        scroll.setWidget(self.bands_container)
        bl.addWidget(scroll)

        add_row = QHBoxLayout()
        add_btn = QPushButton("Add band")
        add_btn.clicked.connect(self._add_band)
        add_row.addWidget(add_btn)
        add_row.addStretch(1)
        bl.addLayout(add_row)
        v.addWidget(bands_box, 1)

        save_row = QHBoxLayout()
        self.dirty_label = QLabel("")
        self.dirty_label.setStyleSheet("color: #d08770;")
        save_row.addWidget(self.dirty_label, 1)
        save_as = QPushButton("Save as new...")
        save_as.clicked.connect(self._save_profile_as)
        save_row.addWidget(save_as)
        save_btn = QPushButton("Save profile")
        save_btn.clicked.connect(self._save_profile)
        save_row.addWidget(save_btn)
        v.addLayout(save_row)
        return w

    def _refresh_profile_combos(self) -> None:
        names = list(self.controller.profiles)
        for combo in (self.profile_combo, self.profile_combo2):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            idx = combo.findText(self.config.active_profile)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _load_profile_into_form(self) -> None:
        profile = self.controller.active_profile
        if profile is None:
            return

        for widget in (self.strength_scale, self.min_gap, self.max_pulses,
                       self.max_duty, self.processes, self.x1_profile):
            widget.blockSignals(True)

        self.strength_scale.setValue(profile.strength_scale)
        self.min_gap.setValue(profile.limits.min_gap_ms)
        self.max_pulses.setValue(profile.limits.max_pulses_sec)
        self.max_duty.setValue(profile.limits.max_duty)
        self.processes.setPlainText(", ".join(profile.processes))

        # The Control Panel profile list is fetched asynchronously; keep whatever
        # we already have so the combo does not flicker empty on every reload.
        existing = [self.x1_profile.itemText(i) for i in range(self.x1_profile.count())]
        if not existing:
            self.x1_profile.addItem("")
            self.controller.check_api()
        self.x1_profile.setCurrentText(profile.x1_profile or "")

        for widget in (self.strength_scale, self.min_gap, self.max_pulses,
                       self.max_duty, self.processes, self.x1_profile):
            widget.blockSignals(False)

        for editor in self._band_editors:
            editor.setParent(None)
            editor.deleteLater()
        self._band_editors.clear()

        for band in profile.bands:
            self._add_band_editor(band)

        self._set_dirty(False)

    def _add_band_editor(self, band: Band) -> None:
        editor = BandEditor(band)
        editor.changed.connect(self._on_band_edited)
        editor.removed.connect(self._remove_band)
        self.bands_layout.insertWidget(self.bands_layout.count() - 1, editor)
        self._band_editors.append(editor)

    def _add_band(self) -> None:
        profile = self.controller.active_profile
        if profile is None:
            return
        name, ok = QInputDialog.getText(self, "Add band", "Band name:")
        if not ok or not name.strip():
            return
        band = Band(name=name.strip(), low_hz=100, high_hz=1000)
        profile.bands.append(band)
        self._add_band_editor(band)
        self._on_profile_edited()

    def _remove_band(self, editor: BandEditor) -> None:
        profile = self.controller.active_profile
        if profile is None or len(profile.bands) <= 1:
            QMessageBox.information(self, "z2haptics", "A profile needs at least one band.")
            return
        if editor.band in profile.bands:
            profile.bands.remove(editor.band)
        self._band_editors.remove(editor)
        editor.setParent(None)
        editor.deleteLater()
        self._on_profile_edited()

    def _on_band_edited(self) -> None:
        # Counts accumulated under the old setting say nothing about the new one,
        # so clear them the moment a threshold moves. The rate readout re-settles
        # within a second or so of releasing the control.
        self.controller.reset_counters()
        self._on_profile_edited()

    def _on_profile_edited(self) -> None:
        profile = self.controller.active_profile
        if profile is None:
            return
        profile.strength_scale = self.strength_scale.value()
        profile.limits.min_gap_ms = self.min_gap.value()
        profile.limits.max_pulses_sec = self.max_pulses.value()
        profile.limits.max_duty = self.max_duty.value()
        raw = self.processes.toPlainText().replace("\n", ",")
        profile.processes = [p.strip() for p in raw.split(",") if p.strip()]
        profile.x1_profile = self.x1_profile.currentText().strip() or None

        self.controller.apply_live_edits(profile)
        self._rebuild_meters()
        self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self.dirty_label.setText(
            "Unsaved changes (already live) — Save to keep them." if dirty else "")

    def _save_profile(self) -> None:
        profile = self.controller.active_profile
        if profile is None:
            return
        try:
            path = save_profile(profile)
        except Exception as e:
            QMessageBox.warning(self, "z2haptics", f"Could not save: {e}")
            return
        self._set_dirty(False)
        QMessageBox.information(self, "z2haptics", f"Saved to\n{path}")

    def _save_profile_as(self) -> None:
        profile = self.controller.active_profile
        if profile is None:
            return
        name, ok = QInputDialog.getText(self, "Save as", "New profile name:")
        if not ok or not name.strip():
            return
        clone = copy.deepcopy(profile)
        clone.name = name.strip()
        clone.source = None
        try:
            path = save_profile(clone)
        except Exception as e:
            QMessageBox.warning(self, "z2haptics", f"Could not save: {e}")
            return
        self.controller.reload_profiles()
        self.config.active_profile = clone.name
        self.config.save()
        self._refresh_profile_combos()
        self._load_profile_into_form()
        self._rebuild_meters()
        QMessageBox.information(self, "z2haptics", f"Saved to\n{path}")

    def _reset_counters(self) -> None:
        self.controller.reset_counters()
        for editor in self._band_editors:
            editor.reset_meter()

    def _reload_profiles(self) -> None:
        self.controller.reload_profiles()
        self._refresh_profile_combos()
        self._load_profile_into_form()
        self._rebuild_meters()

    def _on_profile_selected(self, name: str) -> None:
        if not name or name == self.config.active_profile:
            return
        if self._dirty:
            reply = QMessageBox.question(
                self, "z2haptics",
                "You have unsaved changes to the current profile.\nSwitch anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._refresh_profile_combos()
                return
        self.controller.set_profile(name)
        self._refresh_profile_combos()
        self._load_profile_into_form()
        self._rebuild_meters()

    def _on_profile_changed(self, name: str) -> None:
        # Auto-switching can change the profile behind our back.
        if name != self.config.active_profile:
            self.config.active_profile = name
            self._refresh_profile_combos()
            self._load_profile_into_form()
            self._rebuild_meters()

    # -- Audio ----------------------------------------------------------------

    def _build_audio_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("Capture device")
        f = QFormLayout(box)

        self.device_combo = QComboBox()
        self.device_combo.setToolTip(
            "Which output to listen to. This is loopback capture -- it reads what "
            "is already playing and does not affect what you hear.")
        f.addRow("Device", self.device_combo)

        self.samplerate_combo = QComboBox()
        for sr in (44100, 48000, 96000):
            self.samplerate_combo.addItem(str(sr), sr)
        idx = self.samplerate_combo.findData(self.config.samplerate)
        self.samplerate_combo.setCurrentIndex(max(0, idx))
        f.addRow("Sample rate", self.samplerate_combo)

        v.addWidget(box)

        note = QLabel(
            "Changing the device or sample rate restarts audio capture.\n\n"
            "Pick the output you actually listen to. If you use a virtual mixer "
            "(Elgato Wave Link, VoiceMeeter), capture the device carrying game "
            "audio, not the one carrying your mic."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        v.addWidget(note)

        row = QHBoxLayout()
        refresh = QPushButton("Refresh list")
        refresh.clicked.connect(self._refresh_devices)
        row.addWidget(refresh)
        apply_btn = QPushButton("Apply and restart capture")
        apply_btn.clicked.connect(self._apply_audio)
        row.addWidget(apply_btn)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)

        self._refresh_devices()
        return w

    def _refresh_devices(self) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("System default output", "")
        try:
            for name, is_default in list_devices():
                label = f"{name}  (current default)" if is_default else name
                self.device_combo.addItem(label, name)
        except Exception as e:
            log.warning("could not list devices: %s", e)
        idx = self.device_combo.findData(self.config.device_name)
        self.device_combo.setCurrentIndex(max(0, idx))
        self.device_combo.blockSignals(False)

    def _apply_audio(self) -> None:
        self.config.device_name = self.device_combo.currentData() or ""
        self.config.samplerate = self.samplerate_combo.currentData()
        self.config.save()
        if self.controller.running:
            self.controller.restart()
        else:
            self.status_label.setText("Audio settings saved.")

    # -- Test -----------------------------------------------------------------

    def _build_test_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("Single pulse")
        f = QFormLayout(box)

        self.test_strength = QSlider(Qt.Orientation.Horizontal)
        self.test_strength.setRange(0, 100)
        self.test_strength.setValue(self.config.test_strength)
        self.test_strength_label = QLabel(str(self.config.test_strength))
        self.test_strength.valueChanged.connect(
            lambda v: self.test_strength_label.setText(str(v)))
        row = QHBoxLayout(); row.addWidget(self.test_strength, 1); row.addWidget(self.test_strength_label)
        f.addRow("Strength (0-100)", self._wrap(row))

        self.test_duration = QSlider(Qt.Orientation.Horizontal)
        self.test_duration.setRange(1, 1000)
        self.test_duration.setValue(self.config.test_duration_ms)
        self.test_duration_label = QLabel(f"{self.config.test_duration_ms} ms")
        self.test_duration.valueChanged.connect(
            lambda v: self.test_duration_label.setText(f"{v} ms"))
        row = QHBoxLayout(); row.addWidget(self.test_duration, 1); row.addWidget(self.test_duration_label)
        f.addRow("Duration", self._wrap(row))

        fire = QPushButton("Fire pulse")
        fire.clicked.connect(self._fire_test)
        f.addRow("", fire)
        v.addWidget(box)

        presets = QGroupBox("Presets")
        pl = QVBoxLayout(presets)
        for label, dur, strength in [
            ("Light tick (20ms, 30)", 20, 30),
            ("Gunshot (45ms, 70)", 45, 70),
            ("Explosion (110ms, 100)", 110, 100),
            ("Long rumble (400ms, 60)", 400, 60),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, d=dur, s=strength: self._fire(d, s))
            pl.addWidget(btn)

        burst = QPushButton("Burst: 5 shots, 90ms apart")
        burst.clicked.connect(self._fire_burst)
        pl.addWidget(burst)

        ramp = QPushButton("Strength ramp: 10 -> 100")
        ramp.clicked.connect(self._fire_ramp)
        pl.addWidget(ramp)
        v.addWidget(presets)

        self.test_result = QLabel("")
        self.test_result.setStyleSheet("color: gray;")
        v.addWidget(self.test_result)

        note = QLabel(
            "Test pulses bypass the rate limiter, since you asked for this exact "
            "pulse. A reply of 'OK' means the command was accepted -- it is not "
            "proof the motor moved, so trust your hand over the text."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        v.addWidget(note)
        v.addStretch(1)
        return w

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _fire(self, duration: int, strength: int) -> None:
        self.controller.test_pulse(duration, strength)
        self.test_result.setText(f"vibrate {duration} {strength}  ->  ...")

    def _on_test_result(self, text: str) -> None:
        self.test_result.setText(text)

    def _on_x1_profiles(self, names: list) -> None:
        current = self.x1_profile.currentText()
        self.x1_profile.blockSignals(True)
        self.x1_profile.clear()
        self.x1_profile.addItem("")
        self.x1_profile.addItems(names)
        self.x1_profile.setCurrentText(current)
        self.x1_profile.blockSignals(False)

    def _fire_test(self) -> None:
        d, s = self.test_duration.value(), self.test_strength.value()
        self.config.test_duration_ms, self.config.test_strength = d, s
        self.config.save()
        self._fire(d, s)

    def _fire_burst(self) -> None:
        from PySide6.QtCore import QTimer as _T
        for i in range(5):
            _T.singleShot(i * 90, lambda: self.controller.test_pulse(45, 70))
        self.test_result.setText("burst sent")

    def _fire_ramp(self) -> None:
        from PySide6.QtCore import QTimer as _T
        for i, s in enumerate(range(10, 101, 15)):
            _T.singleShot(i * 600, lambda st=s: self.controller.test_pulse(70, st))
        self.test_result.setText("ramp sent")

    # -- General --------------------------------------------------------------

    def _build_general_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        behaviour = QGroupBox("Behaviour")
        f = QFormLayout(behaviour)

        self.auto_switch_cb = QCheckBox("Follow the foreground app")
        self.auto_switch_cb.setChecked(self.config.auto_switch)
        self.auto_switch_cb.setToolTip(
            "Switch profiles automatically based on which window has focus.")
        self.auto_switch_cb.toggled.connect(self._on_general_changed)
        f.addRow(self.auto_switch_cb)

        self.start_on_launch_cb = QCheckBox("Start haptics when the app launches")
        self.start_on_launch_cb.setChecked(self.config.start_engine_on_launch)
        self.start_on_launch_cb.toggled.connect(self._on_general_changed)
        f.addRow(self.start_on_launch_cb)

        self.start_min_cb = QCheckBox("Start minimised to tray")
        self.start_min_cb.setChecked(self.config.start_minimised)
        self.start_min_cb.toggled.connect(self._on_general_changed)
        f.addRow(self.start_min_cb)

        self.autostart_cb = QCheckBox("Start with Windows")
        self.autostart_cb.setChecked(get_autostart())
        self.autostart_cb.toggled.connect(self._on_autostart_toggled)
        f.addRow(self.autostart_cb)

        self.notify_cb = QCheckBox("Show tray notifications")
        self.notify_cb.setChecked(self.config.show_notifications)
        self.notify_cb.toggled.connect(self._on_general_changed)
        f.addRow(self.notify_cb)

        v.addWidget(behaviour)

        api_box = QGroupBox("Swiftpoint X1 API")
        al = QVBoxLayout(api_box)
        self.api_status = QLabel("Checking...")
        self.api_status.setWordWrap(True)
        al.addWidget(self.api_status)

        row = QHBoxLayout()
        check = QPushButton("Re-check")
        check.clicked.connect(self._check_api)
        row.addWidget(check)
        enable = QPushButton("Enable API in settings.ini")
        enable.clicked.connect(self._enable_api)
        row.addWidget(enable)
        row.addStretch(1)
        al.addLayout(row)

        hint = QLabel(
            "The API is off by default. After enabling it you must FULLY quit the "
            "Swiftpoint X1 Control Panel from its tray icon and relaunch it -- the "
            "setting is only read at startup, and the Control Panel rewrites its "
            "settings file on a clean exit, which silently reverts edits made while "
            "it is running."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        al.addWidget(hint)
        v.addWidget(api_box)
        v.addStretch(1)

        QTimer.singleShot(0, self._check_api)
        return w

    def _on_general_changed(self) -> None:
        self.config.auto_switch = self.auto_switch_cb.isChecked()
        self.config.start_engine_on_launch = self.start_on_launch_cb.isChecked()
        self.config.start_minimised = self.start_min_cb.isChecked()
        self.config.show_notifications = self.notify_cb.isChecked()
        self.config.save()
        if self.controller.running:
            self.controller.restart()

    def _on_autostart_toggled(self, checked: bool) -> None:
        actual = set_autostart(checked)
        if actual != checked:
            self.autostart_cb.blockSignals(True)
            self.autostart_cb.setChecked(actual)
            self.autostart_cb.blockSignals(False)
            QMessageBox.warning(self, "z2haptics",
                                "Could not update the Windows startup entry.")
        self.config.autostart = actual
        self.config.save()

    def _check_api(self) -> None:
        self.api_status.setText("Checking...")
        self.api_status.setStyleSheet("color: gray;")
        self.controller.check_api()

    def _on_api_status(self, ok: bool, message: str) -> None:
        self.api_status.setText(("OK — " if ok else "Not available — ") + message)
        self.api_status.setStyleSheet("color: #a3be8c;" if ok else "color: #bf616a;")

    def _enable_api(self) -> None:
        from ..cli import SETTINGS_INI
        import shutil

        if not SETTINGS_INI.exists():
            QMessageBox.warning(self, "z2haptics", f"settings.ini not found at\n{SETTINGS_INI}")
            return
        text = SETTINGS_INI.read_text(encoding="utf-8", errors="replace")
        if "X1API=true" in text:
            QMessageBox.information(self, "z2haptics", "Already enabled.")
            return
        if "X1API=false" not in text:
            QMessageBox.warning(self, "z2haptics",
                                "No X1API key found — the Control Panel may be too old.")
            return
        try:
            shutil.copy2(SETTINGS_INI, SETTINGS_INI.with_suffix(".ini.bak"))
            SETTINGS_INI.write_text(text.replace("X1API=false", "X1API=true"),
                                    encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "z2haptics", f"Could not edit settings.ini: {e}")
            return
        QMessageBox.information(
            self, "z2haptics",
            "Enabled (a backup was saved alongside it).\n\n"
            "Now fully quit the Swiftpoint X1 Control Panel from its tray icon "
            "and relaunch it, then press Re-check.",
        )

    # -- window behaviour -----------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_profile_combos()
        self._load_profile_into_form()
        self._rebuild_meters()

    def closeEvent(self, event) -> None:
        # Closing hides to tray rather than quitting; the tray menu has Quit.
        event.ignore()
        self.hide()
