"""
Click-driven weapon simulation.

The point of this path is that a click is ground truth where audio was
inference, so the tests focus on the places that can still go wrong: firing
when the game is not listening, firing on an empty magazine, and automatic fire
running at the wrong rate.
"""

import time

import pytest

from z2haptics.firing import (
    DEFAULT_SPECS,
    FireController,
    WeaponSpec,
    spec_from_dict,
    spec_to_dict,
)


class FakeSink:
    def __init__(self):
        self.pulses = []

    def fire(self, pulse):
        self.pulses.append(pulse)
        return True


@pytest.fixture
def sink():
    return FakeSink()


def controller(sink, weapon, **kwargs):
    c = FireController(sink=sink, weapon=weapon, **kwargs)
    c.start()
    return c


# -- semi-automatic -----------------------------------------------------------

def test_one_click_fires_one_round_on_semi_auto(sink):
    spec = WeaponSpec(name="sniper", auto=False, magazine=5, duration_ms=130,
                      strength=100)
    c = controller(sink, spec)
    try:
        c.trigger_down()
        time.sleep(0.15)
        c.trigger_up()
        assert len(sink.pulses) == 1
        assert sink.pulses[0].duration_ms == 130
    finally:
        c.stop()


def test_holding_semi_auto_does_not_repeat(sink):
    spec = WeaponSpec(name="sniper", auto=False, magazine=5)
    c = controller(sink, spec)
    try:
        c.trigger_down()
        time.sleep(0.4)
        assert len(sink.pulses) == 1, "held semi-auto repeated"
    finally:
        c.stop()


# -- automatic ----------------------------------------------------------------

def test_automatic_fire_repeats_while_held(sink):
    spec = WeaponSpec(name="lmg", rpm=600, magazine=100)   # 10 rounds/sec
    c = controller(sink, spec)
    try:
        c.trigger_down()
        time.sleep(0.55)
        c.trigger_up()
        # ~5-6 rounds in 550ms; allow slack for scheduler jitter.
        assert 3 <= len(sink.pulses) <= 8, f"fired {len(sink.pulses)}"
    finally:
        c.stop()


def test_higher_rpm_fires_more_rounds(sink):
    def count(rpm):
        s = FakeSink()
        c = controller(s, WeaponSpec(name="w", rpm=rpm, magazine=200))
        try:
            c.trigger_down()
            time.sleep(0.4)
            c.trigger_up()
            return len(s.pulses)
        finally:
            c.stop()

    slow, fast = count(300), count(900)
    assert fast > slow, f"900rpm fired {fast}, 300rpm fired {slow}"


def test_release_stops_automatic_fire(sink):
    spec = WeaponSpec(name="lmg", rpm=900, magazine=100)
    c = controller(sink, spec)
    try:
        c.trigger_down()
        time.sleep(0.2)
        c.trigger_up()
        settled = len(sink.pulses)
        time.sleep(0.3)
        assert len(sink.pulses) == settled, "kept firing after release"
    finally:
        c.stop()


# -- magazine and reload ------------------------------------------------------

def test_firing_stops_when_the_magazine_empties(sink):
    spec = WeaponSpec(name="smg", rpm=1200, magazine=5)
    c = controller(sink, spec)
    try:
        c.trigger_down()
        time.sleep(0.8)
        assert len(sink.pulses) == 5, f"fired {len(sink.pulses)} from a 5-round magazine"
        assert c.stats()["ammo"] == 0
    finally:
        c.stop()


def test_reload_refills_and_suppresses_meanwhile(sink):
    spec = WeaponSpec(name="smg", rpm=1200, magazine=3, reload_s=0.3)
    c = controller(sink, spec)
    try:
        c.trigger_down()
        time.sleep(0.4)
        c.trigger_up()
        assert c.stats()["ammo"] == 0

        c.reload()
        assert c.stats()["reloading_s"] > 0
        fired_before = len(sink.pulses)

        c.trigger_down()
        time.sleep(0.15)              # still inside the reload
        assert len(sink.pulses) == fired_before, "fired during reload"

        c.refill()
        time.sleep(0.1)
        assert len(sink.pulses) > fired_before, "did not resume after reload"
    finally:
        c.stop()


def test_reload_on_a_full_magazine_is_ignored(sink):
    spec = WeaponSpec(name="ak", magazine=30, reload_s=2.0)
    c = controller(sink, spec)
    try:
        c.reload()
        assert c.stats()["reloading_s"] == 0, "reloaded a full magazine"
    finally:
        c.stop()


def test_dry_clicks_are_counted_not_pulsed(sink):
    spec = WeaponSpec(name="pistol", auto=False, magazine=1)
    c = controller(sink, spec)
    try:
        c.trigger_down(); time.sleep(0.1); c.trigger_up()
        assert len(sink.pulses) == 1

        for _ in range(3):
            c.trigger_down(); time.sleep(0.06); c.trigger_up()

        assert len(sink.pulses) == 1, "pulsed on an empty magazine"
        assert c.stats()["dry_clicks"] >= 1
    finally:
        c.stop()


# -- gating -------------------------------------------------------------------

def test_nothing_fires_when_the_game_is_not_active(sink):
    """Otherwise the mouse buzzes every time you click a link."""
    spec = WeaponSpec(name="ak", rpm=600, magazine=30)
    c = controller(sink, spec, is_active=lambda: False)
    try:
        c.trigger_down()
        time.sleep(0.3)
        assert sink.pulses == []
    finally:
        c.stop()


def test_no_weapon_means_no_pulses(sink):
    c = controller(sink, None)
    try:
        c.trigger_down()
        time.sleep(0.2)
        assert sink.pulses == []
    finally:
        c.stop()


# -- shaping ------------------------------------------------------------------

def test_first_round_can_hit_harder(sink):
    spec = WeaponSpec(name="ak", rpm=600, magazine=30, strength=50,
                      first_shot_bonus=30)
    c = controller(sink, spec)
    try:
        c.trigger_down()
        time.sleep(0.35)
        c.trigger_up()
        assert len(sink.pulses) >= 2
        assert sink.pulses[0].strength > sink.pulses[1].strength
        assert sink.pulses[0].label.endswith("!")
    finally:
        c.stop()


def test_strength_scale_is_applied_and_clamped(sink):
    spec = WeaponSpec(name="ak", auto=False, magazine=5, strength=80)
    c = controller(sink, spec, strength_scale=2.0)
    try:
        c.trigger_down(); time.sleep(0.1)
        assert sink.pulses[0].strength == 100
    finally:
        c.stop()


def test_swapping_weapon_resets_the_magazine(sink):
    a = WeaponSpec(name="a", auto=False, magazine=2)
    b = WeaponSpec(name="b", auto=False, magazine=7)
    c = controller(sink, a)
    try:
        c.trigger_down(); time.sleep(0.08); c.trigger_up()
        assert c.stats()["ammo"] == 1
        c.set_weapon(b)
        assert c.stats()["ammo"] == 7
        assert c.stats()["weapon"] == "b"
    finally:
        c.stop()


def test_external_ammo_sync_is_clamped(sink):
    spec = WeaponSpec(name="ak", magazine=30)
    c = controller(sink, spec)
    try:
        c.set_ammo(12)
        assert c.stats()["ammo"] == 12
        c.set_ammo(999)
        assert c.stats()["ammo"] == 30
        c.set_ammo(-5)
        assert c.stats()["ammo"] == 0
    finally:
        c.stop()


# -- specs --------------------------------------------------------------------

def test_spec_round_trips():
    spec = DEFAULT_SPECS["lmg"]
    assert spec_from_dict(spec_to_dict(spec)) == spec


def test_spec_ignores_unknown_keys():
    spec = spec_from_dict({"name": "x", "rpm": 500, "from_the_future": 1})
    assert spec.name == "x"
    assert spec.rpm == 500


def test_shot_interval_matches_rpm():
    assert WeaponSpec(name="x", rpm=600).shot_interval == pytest.approx(0.1)
    assert WeaponSpec(name="x", rpm=1200).shot_interval == pytest.approx(0.05)
