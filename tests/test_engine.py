"""Tests for winner selection, pulse shaping and motor-protective rate limiting."""

import time

import pytest

from z2haptics.analysis import Band, Onset
from z2haptics.api import DURATION_MAX, HapticSink, Pulse, X1Connection
from z2haptics.engine import PRIORITY_WEIGHT, HapticEngine
from z2haptics.profiles import MotorLimits, Profile


def band(name, priority, **kw):
    defaults = dict(
        low_hz=20, high_hz=100, duration_ms=50, strength_min=20, strength_max=100,
        priority=priority,
    )
    defaults.update(kw)
    return Band(name=name, **defaults)


def onset(name, strength):
    return Onset(band=name, strength=strength, level=0.01, level_db=-40.0,
                 flux=1.0, threshold=0.5, share=0.5)


# -- winner selection ---------------------------------------------------------

def rank(onsets, bands):
    """Mirror of HapticEngine's ranking, exercised without audio hardware."""
    by_name = {b.name: b for b in bands}

    def score(o):
        b = by_name.get(o.band)
        return o.strength + PRIORITY_WEIGHT * (b.priority if b else 0)

    return sorted(onsets, key=score, reverse=True)[0]


def test_priority_breaks_ties_between_equal_strengths():
    bands = [band("impact", 3), band("gunfire", 2)]
    winner = rank([onset("gunfire", 0.5), onset("impact", 0.5)], bands)
    assert winner.band == "impact"


def test_strong_low_priority_onset_beats_weak_high_priority_leakage():
    """The regression this weighting exists to prevent.

    A gunshot leaks a little energy into the low `impact` band. Ranking on
    priority first made that leakage win, so every gunshot was shaped like an
    explosion. Strength must dominate.
    """
    bands = [band("impact", 3), band("gunfire", 2)]
    winner = rank([onset("impact", 0.15), onset("gunfire", 0.56)], bands)
    assert winner.band == "gunfire"


def test_genuine_explosion_still_wins_over_its_own_leakage():
    bands = [band("impact", 3), band("gunfire", 2)]
    winner = rank([onset("impact", 0.86), onset("gunfire", 0.40)], bands)
    assert winner.band == "impact"


def test_unknown_band_does_not_crash_ranking():
    bands = [band("impact", 3)]
    assert rank([onset("ghost", 0.9), onset("impact", 0.2)], bands).band == "ghost"


# -- pulse shaping ------------------------------------------------------------

@pytest.mark.parametrize("strength,expected", [(0.0, 20), (0.5, 60), (1.0, 100)])
def test_strength_maps_across_band_window(strength, expected):
    b = band("x", 1, strength_min=20, strength_max=100)
    got = round(b.strength_min + strength * (b.strength_max - b.strength_min))
    assert got == expected


def test_profile_scale_is_applied_and_clamped():
    b = band("x", 1, strength_min=20, strength_max=100)
    raw = b.strength_min + 1.0 * (b.strength_max - b.strength_min)
    assert max(0, min(100, int(round(raw * 1.5)))) == 100
    assert max(0, min(100, int(round(raw * 0.5)))) == 50


# -- HapticSink limits --------------------------------------------------------

def test_duration_and_strength_are_clamped():
    conn = X1Connection()
    sent = []
    conn.command = lambda cmd, expect_response=True: sent.append(cmd) or "OK"

    conn.vibrate(999_999, 250)
    conn.vibrate(-50, -10)
    assert sent == [f"vibrate {DURATION_MAX} 100", "vibrate 0 0"]


def test_min_gap_blocks_rapid_pulses():
    sink = HapticSink(min_gap_ms=100.0, max_pulses_sec=1000, max_duty=1.0, dry_run=True)
    now = time.perf_counter()
    sink._last_fire = now
    assert not sink._budget_allows(now + 0.01, 0.02)
    assert sink._budget_allows(now + 0.20, 0.02)


def test_max_pulses_per_second_is_enforced():
    sink = HapticSink(min_gap_ms=0.0, max_pulses_sec=5, max_duty=1.0, dry_run=True)
    now = time.perf_counter()
    sink._recent = [(now, 0.001) for _ in range(5)]
    assert not sink._budget_allows(now, 0.001)


def test_duty_cycle_caps_sustained_drive():
    """Long pulses must be refused even when the pulse *count* is legal."""
    sink = HapticSink(min_gap_ms=0.0, max_pulses_sec=100, max_duty=0.5, dry_run=True)
    now = time.perf_counter()
    sink._recent = [(now, 0.45)]
    assert not sink._budget_allows(now, 0.20)
    assert sink._budget_allows(now, 0.02)


def test_dry_run_never_touches_the_device():
    sink = HapticSink(dry_run=True)
    sink.start()
    try:
        assert sink.fire(Pulse(50, 80, "x"))
        time.sleep(0.3)
        assert sink.sent >= 1
        assert not sink.conn.connected
    finally:
        sink.stop()


def test_full_queue_drops_the_weakest_pulse():
    sink = HapticSink(queue_size=2, dry_run=True)
    assert sink.fire(Pulse(50, 10, "weak"))
    assert sink.fire(Pulse(50, 20, "mid"))
    assert sink.fire(Pulse(50, 90, "strong"))   # evicts "weak"
    assert sink.dropped_queue == 1


def test_full_queue_rejects_a_weaker_newcomer():
    sink = HapticSink(queue_size=2, dry_run=True)
    sink.fire(Pulse(50, 80, "a"))
    sink.fire(Pulse(50, 90, "b"))
    assert not sink.fire(Pulse(50, 5, "weak"))


# -- profile plumbing ---------------------------------------------------------

def test_apply_profile_updates_limits(monkeypatch):
    monkeypatch.setattr("z2haptics.engine.LoopbackCapture", lambda **kw: type(
        "FakeCapture", (), {"start": lambda s: None, "stop": lambda s: None,
                            "resolved_name": "fake"})())

    p1 = Profile(name="A", bands=[band("x", 1)], limits=MotorLimits(50, 10, 0.5))
    p2 = Profile(name="B", bands=[band("y", 2)], limits=MotorLimits(80, 6, 0.3))

    engine = HapticEngine(profile=p1, dry_run=True)
    engine.apply_profile(p2)

    assert engine.profile.name == "B"
    assert engine.sink.min_gap_ms == 80
    assert engine.sink.max_pulses_sec == 6
    assert engine.sink.max_duty == 0.3
    assert engine.stats.profile_switches == 1


def test_reapplying_the_same_profile_is_a_noop(monkeypatch):
    monkeypatch.setattr("z2haptics.engine.LoopbackCapture", lambda **kw: type(
        "FakeCapture", (), {"start": lambda s: None, "stop": lambda s: None,
                            "resolved_name": "fake"})())
    p = Profile(name="A", bands=[band("x", 1)])
    engine = HapticEngine(profile=p, dry_run=True)
    engine.apply_profile(p)
    assert engine.stats.profile_switches == 0
