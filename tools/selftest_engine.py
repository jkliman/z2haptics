"""
End-to-end detection self-test.

Plays a synthetic sequence of events out the default speaker -- low thumps,
mid-band bursts and high clicks, separated by silence -- captures them back over
loopback, and reports which bands fired. This exercises the whole chain
(capture -> mono -> FFT -> flux -> threshold -> onset) without needing a game
running or a human to judge whether detection worked.

    python tools/selftest_engine.py [--profile FPS] [--haptics]

--haptics also drives the motor, so you can feel whether the pulse shaping
matches the events.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from z2haptics.analysis import BandAnalyzer  # noqa: E402
from z2haptics.api import HapticSink, Pulse  # noqa: E402
from z2haptics.audio import LoopbackCapture  # noqa: E402
from z2haptics.profiles import discover  # noqa: E402

SR = 48000


def thump(dur=0.18, f0=55.0):
    """Low-frequency impact: a decaying sine, like an explosion."""
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    env = np.exp(-t * 22)
    return 0.7 * env * np.sin(2 * np.pi * f0 * t)


def burst(dur=0.09, lo=150, hi=450):
    """Mid-band noise burst: stands in for a weapon report."""
    n = int(SR * dur)
    noise = np.random.default_rng(0).normal(0, 1, n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < lo) | (freqs > hi)] = 0
    sig = np.fft.irfft(spec, n)
    env = np.exp(-np.linspace(0, dur, n) * 32)
    return 0.75 * env * sig / (np.max(np.abs(sig)) or 1)


def click(dur=0.03, lo=2500, hi=6500):
    """High-band tick: stands in for a supersonic crack."""
    n = int(SR * dur)
    noise = np.random.default_rng(1).normal(0, 1, n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < lo) | (freqs > hi)] = 0
    sig = np.fft.irfft(spec, n)
    env = np.exp(-np.linspace(0, dur, n) * 90)
    return 0.6 * env * sig / (np.max(np.abs(sig)) or 1)


def silence(dur):
    return np.zeros(int(SR * dur))


def build_sequence():
    """Return (audio, expected) where expected is a list of (time_s, label)."""
    parts, expected, t = [], [], 0.0

    def add(sig, label=None):
        nonlocal t
        if label:
            expected.append((t, label))
        parts.append(sig)
        t += len(sig) / SR

    add(silence(1.0))                       # let the detector learn the noise floor
    for _ in range(3):
        add(thump(), "low")
        add(silence(0.45))
    for _ in range(4):
        add(burst(), "mid")
        add(silence(0.30))
    for _ in range(4):
        add(click(), "high")
        add(silence(0.28))
    add(thump(0.25, 45), "low")             # one heavy finisher
    add(silence(0.8))

    return np.concatenate(parts), expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="FPS")
    ap.add_argument("--haptics", action="store_true", help="also drive the motor")
    args = ap.parse_args()

    profiles = discover()
    if args.profile not in profiles:
        print(f"no profile {args.profile!r}; have {', '.join(profiles)}")
        return 1
    profile = profiles[args.profile]

    audio, expected = build_sequence()
    duration = len(audio) / SR
    print(f"profile: {profile.name}")
    print(f"bands:   {', '.join(f'{b.name}({b.low_hz:.0f}-{b.high_hz:.0f}Hz)' for b in profile.bands)}")
    print(f"sequence: {duration:.1f}s, {len(expected)} events "
          f"({sum(1 for _, l in expected if l == 'low')} low, "
          f"{sum(1 for _, l in expected if l == 'mid')} mid, "
          f"{sum(1 for _, l in expected if l == 'high')} high)\n")

    analyzer = BandAnalyzer(profile.bands, samplerate=SR)
    detected: list[tuple[float, str, float]] = []
    t0 = None

    sink = None
    if args.haptics:
        sink = HapticSink(
            min_gap_ms=profile.limits.min_gap_ms,
            max_pulses_sec=profile.limits.max_pulses_sec,
            max_duty=profile.limits.max_duty,
        )
        sink.start()

    def on_audio(mono):
        nonlocal t0
        if t0 is None:
            t0 = time.time()
        for onset in analyzer.push(mono):
            detected.append((time.time() - t0, onset.band, onset.strength))
            if sink:
                band = next(b for b in profile.bands if b.name == onset.band)
                span = band.strength_max - band.strength_min
                strength = int(band.strength_min + onset.strength * span)
                sink.fire(Pulse(band.duration_ms, strength, onset.band, band.priority))

    cap = LoopbackCapture(on_audio, samplerate=SR, blocksize=512)
    cap.start()
    print(f"capturing from: {cap.resolved_name}\n")
    time.sleep(0.4)

    spk = sc.default_speaker()
    player = threading.Thread(
        target=lambda: spk.play(np.column_stack([audio, audio]), samplerate=SR),
        daemon=True,
    )
    player.start()
    time.sleep(duration + 0.8)
    cap.stop()
    if sink:
        sink.stop()

    print(f"=== detected {len(detected)} onsets ===")
    for ts, band, strength in detected:
        print(f"  {ts:6.2f}s  {band:<9} strength={strength:.2f}")

    by_band: dict[str, int] = {}
    for _, band, _ in detected:
        by_band[band] = by_band.get(band, 0) + 1

    print(f"\n=== per band ===")
    for b in profile.bands:
        print(f"  {b.name:<9} {by_band.get(b.name, 0)} onsets")

    expected_counts = {"low": 4, "mid": 4, "high": 4}
    print(f"\n=== expected roughly ===")
    print(f"  low-frequency events : {expected_counts['low']}")
    print(f"  mid-band events      : {expected_counts['mid']}")
    print(f"  high-band events     : {expected_counts['high']}")

    if sink:
        print(f"\nhaptics: {sink.stats()}")

    total = len(detected)
    if total == 0:
        print("\nFAIL: nothing detected. Is the default speaker the device being captured?")
        return 1
    print(f"\n{total} onsets detected across {len(by_band)} band(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
