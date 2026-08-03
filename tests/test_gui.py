"""
GUI smoke tests.

Run against Qt's offscreen platform, so they exercise real widget construction,
signal wiring and the profile edit round-trip without needing a display. This is
where a typo in a signal name or a bad widget parent shows up -- silently, at
runtime, otherwise.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from z2haptics.gui.config import AppConfig  # noqa: E402
from z2haptics.gui.controller import EngineController  # noqa: E402
from z2haptics.gui.tray import make_icon  # noqa: E402
from z2haptics.gui.widgets import BandEditor, BandMeter  # noqa: E402
from z2haptics.gui.window import SettingsWindow  # noqa: E402
from z2haptics.analysis import Band  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr("z2haptics.gui.config.CONFIG_PATH", tmp_path / "config.json")
    return AppConfig()


@pytest.fixture
def controller(qapp, config):
    c = EngineController(config)
    yield c
    c.stop()


# -- config -------------------------------------------------------------------

def test_config_round_trips(tmp_path):
    path = tmp_path / "config.json"
    cfg = AppConfig(device_name="Headset", samplerate=44100, auto_switch=False)
    cfg.save(path)
    loaded = AppConfig.load(path)
    assert loaded.device_name == "Headset"
    assert loaded.samplerate == 44100
    assert loaded.auto_switch is False


def test_config_ignores_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"device_name": "X", "from_a_future_version": 1}', encoding="utf-8")
    assert AppConfig.load(path).device_name == "X"


def test_config_survives_corrupt_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    assert AppConfig.load(path).samplerate == 48000


# -- icon ---------------------------------------------------------------------

def test_icon_renders_in_both_states(qapp):
    for active in (True, False):
        icon = make_icon(active)
        assert not icon.isNull()
        assert not icon.pixmap(64, 64).isNull()


# -- widgets ------------------------------------------------------------------

def test_band_meter_accepts_updates(qapp):
    meter = BandMeter("impact")
    meter.set_level(0.01, 0.003, True)
    meter.flash(80)
    meter.resize(300, 26)
    meter.render(meter.grab())        # force a paint pass


def test_band_meter_handles_silence(qapp):
    """log10(0) would blow up if the floor clamp were missing."""
    meter = BandMeter("impact")
    meter.set_level(0.0, 0.0, False)
    meter.resize(300, 26)
    meter.render(meter.grab())


def test_band_editor_writes_back_to_the_band(qapp):
    band = Band(name="impact", low_hz=20, high_hz=90)
    editor = BandEditor(band)

    editor.duration.setValue(123)
    editor.smin.setValue(40)
    editor.smax.setValue(95)
    editor.priority.setValue(7)

    assert band.duration_ms == 123
    assert band.strength_min == 40
    assert band.strength_max == 95
    assert band.priority == 7


def test_band_editor_orders_inverted_ranges(qapp):
    """Typing min > max must not produce an inverted, un-triggerable range."""
    band = Band(name="x", low_hz=20, high_hz=90)
    editor = BandEditor(band)
    editor.smin.setValue(90)
    editor.smax.setValue(20)
    assert band.strength_min <= band.strength_max

    editor.floor_db.setValue(-10)
    editor.ceil_db.setValue(-90)
    assert band.level_floor_db <= band.level_ceil_db


def test_band_editor_toggle_sets_enabled(qapp):
    band = Band(name="x", low_hz=20, high_hz=90)
    editor = BandEditor(band)
    editor.setChecked(False)
    assert band.enabled is False


# -- controller ---------------------------------------------------------------

def test_controller_loads_profiles(controller):
    assert controller.profiles
    assert controller.active_profile is not None


def test_controller_switches_profile(controller):
    names = list(controller.profiles)
    target = names[-1]
    controller.set_profile(target)
    assert controller.config.active_profile == target


def test_profile_switch_does_not_block_the_gui_thread(controller, monkeypatch):
    """Regression: switching profiles froze the window.

    The Control Panel handoff writes configuration to the mouse and can take a
    long time. It used to run inline on the GUI thread. It must now be handed to
    the device worker, so set_profile returns immediately.
    """
    import threading
    import time

    slow_call_started = threading.Event()
    released = threading.Event()

    class SlowEngine:
        def __init__(self):
            self.profile = type("P", (), {"name": "before"})()

        def apply_profile(self, profile, switch_x1=True):
            assert switch_x1 is False, "GUI must not trigger the slow device call inline"
            self.profile = profile

        def switch_x1_profile(self, profile):
            slow_call_started.set()
            released.wait(timeout=5)

    controller.engine = SlowEngine()
    target = list(controller.profiles)[-1]

    start = time.perf_counter()
    controller.set_profile(target)
    elapsed = time.perf_counter() - start

    try:
        assert elapsed < 0.5, f"set_profile blocked for {elapsed:.2f}s"
        assert slow_call_started.wait(timeout=5), "device call never ran on the worker"
    finally:
        released.set()
        controller.engine = None


def test_test_pulse_returns_immediately(controller):
    """Test pulses must not block the GUI either; the result arrives by signal."""
    import time

    received = []
    controller.testResult.connect(received.append)

    start = time.perf_counter()
    controller.test_pulse(10, 5)
    assert time.perf_counter() - start < 0.5


def test_shutdown_is_safe_to_call_twice(controller):
    controller.shutdown()
    controller.shutdown()


def test_controller_is_not_running_before_start(controller):
    assert controller.running is False


# -- window -------------------------------------------------------------------

def test_settings_window_builds(controller, config):
    window = SettingsWindow(controller, config)
    assert window.tabs.count() == 5
    labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert labels == ["Status", "Profile", "Audio", "Test", "General"]
    window.close()


def test_window_shows_a_meter_per_band(controller, config):
    window = SettingsWindow(controller, config)
    window._refresh_profile_combos()
    window._load_profile_into_form()
    window._rebuild_meters()

    profile = controller.active_profile
    assert set(window._meters) == {b.name for b in profile.bands}
    window.close()


def test_editing_a_band_marks_the_profile_dirty(controller, config):
    window = SettingsWindow(controller, config)
    window._refresh_profile_combos()
    window._load_profile_into_form()
    assert window._dirty is False

    window._band_editors[0].duration.setValue(77)
    assert window._dirty is True
    assert controller.active_profile.bands[0].duration_ms == 77
    window.close()


def test_master_edits_reach_the_profile(controller, config):
    window = SettingsWindow(controller, config)
    window._refresh_profile_combos()
    window._load_profile_into_form()

    window.max_duty.setValue(0.42)
    window.min_gap.setValue(88)
    assert controller.active_profile.limits.max_duty == pytest.approx(0.42)
    assert controller.active_profile.limits.min_gap_ms == pytest.approx(88)
    window.close()


def test_process_list_is_parsed_from_free_text(controller, config):
    window = SettingsWindow(controller, config)
    window._refresh_profile_combos()
    window._load_profile_into_form()

    window.processes.setPlainText("a.exe, b.exe\nc.exe")
    assert controller.active_profile.processes == ["a.exe", "b.exe", "c.exe"]
    window.close()


def test_band_editor_shows_live_activity(qapp):
    """The meter and readout must live with the controls that shape them."""
    band = Band(name="gunfire", low_hz=90, high_hz=450, gate=0.003)
    editor = BandEditor(band)

    editor.set_activity({
        "level": 0.02, "peak": 0.05, "gate": 0.003, "open": True,
        "flatness": 0.72, "rate": 2.5, "sent": 14,
        "rejections": {"frames": 900, "accepted": 20, "threshold": 700,
                       "refractory": 100, "share": 40, "flatness": 40},
        "dominant": ("threshold", 700),
    })

    assert "2.5" in editor.readout.text()
    assert "0.72" in editor.readout.text()
    assert "14" in editor.readout.text()


def test_editor_advises_when_a_band_never_fires(qapp):
    band = Band(name="gunfire", low_hz=90, high_hz=450, gate=0.05)
    editor = BandEditor(band)

    editor.set_activity({
        "level": 0.001, "peak": 0.002, "gate": 0.05, "open": False,
        "flatness": 0.1, "rate": 0.0, "sent": 0,
        "rejections": {"frames": 900, "accepted": 0, "threshold": 0,
                       "refractory": 0, "share": 0, "flatness": 0},
        "dominant": ("threshold", 0),
    })
    assert "gate" in editor.advice.text().lower()


def test_editor_names_the_blocking_check(qapp):
    band = Band(name="gunfire", low_hz=90, high_hz=450, gate=0.001)
    editor = BandEditor(band)

    editor.set_activity({
        "level": 0.02, "peak": 0.05, "gate": 0.001, "open": True,
        "flatness": 0.1, "rate": 0.0, "sent": 0,
        "rejections": {"frames": 900, "accepted": 0, "threshold": 10,
                       "refractory": 0, "share": 0, "flatness": 800},
        "dominant": ("flatness", 800),
    })
    assert "flatness" in editor.advice.text().lower()


def test_editor_warns_when_firing_too_often(qapp):
    band = Band(name="gunfire", low_hz=90, high_hz=450, gate=0.001)
    editor = BandEditor(band)
    editor.set_activity({
        "level": 0.02, "peak": 0.05, "gate": 0.001, "open": True,
        "flatness": 0.8, "rate": 15.0, "sent": 300,
        "rejections": {"frames": 900, "accepted": 300, "threshold": 100,
                       "refractory": 0, "share": 0, "flatness": 0},
        "dominant": ("threshold", 100),
    })
    assert editor.advice.text(), "no warning at 15 onsets/sec"


def test_editing_a_band_resets_the_counters(controller, config):
    """Counts from the old setting would misrepresent the new one."""
    window = SettingsWindow(controller, config)
    window._refresh_profile_combos()
    window._load_profile_into_form()

    called = []
    controller.reset_counters = lambda: called.append(True)

    window._band_editors[0].sensitivity.setValue(3.3)
    assert called, "counters were not reset after a threshold change"
    window.close()


def test_reset_counters_is_safe_while_stopped(controller):
    controller.reset_counters()


def test_closing_the_window_hides_rather_than_quits(controller, config):
    window = SettingsWindow(controller, config)
    window.show()
    window.close()
    assert window.isVisible() is False


def test_saved_profile_round_trips(controller, config, tmp_path, monkeypatch):
    """Edits made in the GUI must survive a save/reload cycle."""
    monkeypatch.setattr("z2haptics.profiles.USER_DIR", tmp_path)

    from z2haptics.profiles import load_profile, save_profile

    profile = controller.active_profile
    profile.bands[0].duration_ms = 131
    profile.limits.max_duty = 0.37

    path = save_profile(profile)
    reloaded = load_profile(path)

    assert reloaded.bands[0].duration_ms == 131
    assert reloaded.limits.max_duty == pytest.approx(0.37)
    assert reloaded.name == profile.name
