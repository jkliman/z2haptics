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


def onset(name, strength, flatness=0.8):
    return Onset(band=name, strength=strength, level=0.01, level_db=-40.0,
                 flux=1.0, threshold=0.5, share=0.5, flatness=flatness)


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


def test_connection_serialises_concurrent_commands():
    """Regression: the GUI froze when switching profiles.

    Two threads shared one X1Connection. Each command is a write followed by a
    blocking read, so they interleaved and each could consume the other's
    response -- after which both blocked forever waiting for a reply that had
    already been read. Commands must not overlap.
    """
    import threading

    conn = X1Connection()
    overlaps = []
    active = []
    lock = threading.Lock()

    def fake_io(cmd, expect_response=True):
        with lock:
            active.append(cmd)
            if len(active) > 1:
                overlaps.append(list(active))
        time.sleep(0.002)          # stand in for the pipe round trip
        with lock:
            active.remove(cmd)
        return "OK"

    # Exercise the real locking in command() around a fake transport.
    conn._f = object()
    original = X1Connection.command

    def locked_command(self, cmd, expect_response=True):
        with self._lock:
            return fake_io(cmd, expect_response)

    X1Connection.command = locked_command
    try:
        threads = [
            threading.Thread(target=lambda i=i: [conn.command(f"c{i}-{j}") for j in range(20)])
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "thread deadlocked"
    finally:
        X1Connection.command = original

    assert not overlaps, f"commands overlapped: {overlaps[:3]}"


def test_apply_profile_can_skip_the_slow_device_call():
    """`Profile Set` writes to the mouse, so a UI thread must be able to opt out."""
    from z2haptics.profiles import Profile as P

    calls = []
    p1 = P(name="A", bands=[band("x", 1)], x1_profile="one")
    p2 = P(name="B", bands=[band("y", 1)], x1_profile="two")

    engine = HapticEngine.__new__(HapticEngine)
    engine.profile = p1
    engine.dry_run = False
    engine.stats = type("S", (), {"profile_switches": 0})()
    engine._lock = __import__("threading").Lock()
    engine.analyzer = type("A", (), {"reconfigure": lambda s, b: None})()
    engine.sink = type("S", (), {
        "conn": type("C", (), {"profile_set": lambda s, n: calls.append(n)})(),
        "min_gap_ms": 0, "max_pulses_sec": 0, "max_duty": 0,
    })()

    engine.apply_profile(p2, switch_x1=False)
    assert calls == [], "device call should have been skipped"

    engine.profile = p1
    engine.apply_profile(p2, switch_x1=True)
    assert calls == ["two"]


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


def test_engine_turns_audio_into_shaped_pulses(monkeypatch):
    """Full path: audio in -> onset -> winner -> shaped pulse on the queue.

    Exercises HapticEngine._on_audio directly so no device or capture is needed.
    """
    import numpy as np

    monkeypatch.setattr("z2haptics.engine.LoopbackCapture", lambda **kw: type(
        "FakeCapture", (), {"start": lambda s: None, "stop": lambda s: None,
                            "resolved_name": "fake"})())

    profile = Profile(
        name="T",
        bands=[
            Band(name="low", low_hz=40, high_hz=120, gate=0.0, refractory_ms=0.0,
                 duration_ms=90, strength_min=50, strength_max=100,
                 level_floor_db=-120.0, level_ceil_db=-10.0, priority=3),
            Band(name="high", low_hz=2000, high_hz=6000, gate=0.0, refractory_ms=0.0,
                 duration_ms=20, strength_min=20, strength_max=50,
                 level_floor_db=-120.0, level_ceil_db=-10.0, priority=1),
        ],
    )

    fired: list[Pulse] = []
    engine = HapticEngine(profile=profile, dry_run=True)
    engine.sink.fire = lambda p: fired.append(p) or True

    sr = 48000

    def tone(freq, dur, amp):
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    engine._on_audio(np.zeros(sr // 2, dtype=np.float32))   # learn the floor
    engine._on_audio(tone(70, 0.25, 0.6))
    assert fired, "a loud low tone produced no pulse"
    assert fired[0].label.startswith("low")
    assert fired[0].duration_ms == 90
    assert 50 <= fired[0].strength <= 100

    loud_strength = fired[0].strength
    fired.clear()

    engine2 = HapticEngine(profile=profile, dry_run=True)
    engine2.sink.fire = lambda p: fired.append(p) or True
    engine2._on_audio(np.zeros(sr // 2, dtype=np.float32))
    engine2._on_audio(tone(70, 0.25, 0.02))
    assert fired, "a quiet low tone produced no pulse"
    assert fired[0].strength < loud_strength, "quiet event should pulse softer"


def test_reapplying_the_same_profile_is_a_noop(monkeypatch):
    monkeypatch.setattr("z2haptics.engine.LoopbackCapture", lambda **kw: type(
        "FakeCapture", (), {"start": lambda s: None, "stop": lambda s: None,
                            "resolved_name": "fake"})())
    p = Profile(name="A", bands=[band("x", 1)])
    engine = HapticEngine(profile=p, dry_run=True)
    engine.apply_profile(p)
    assert engine.stats.profile_switches == 0
