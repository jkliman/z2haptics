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
