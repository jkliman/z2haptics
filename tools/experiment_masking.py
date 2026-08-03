"""
Measure how well gunshots survive a dense mix, with and without background
subtraction.

Masking does not come from loudness alone. The detection threshold is a rolling
median of spectral flux, and a steady tone produces almost no flux -- so it
raises nothing. What actually masks is *busy* content: music with moving notes,
overlapping explosions, continuous spectral change. That keeps median flux high,
and a gunshot then has to clear a bar the mix itself raised.

This script synthesises a realistic dense mix and reports the detection rate of
known gunshot events, so changes to the detector can be judged on numbers rather
than vibes.

    python tools/experiment_masking.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from z2haptics.analysis import Band, BandAnalyzer  # noqa: E402

SR = 48000
RNG = np.random.default_rng(7)


def note(freq, dur, amp):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    env = np.minimum(1.0, np.exp(-t * 3.0) + 0.35)
    return amp * env * np.sin(2 * np.pi * freq * t)


def music(duration, amp=0.22):
    """A moving bassline plus pad -- busy, so it keeps median flux high."""
    out = []
    scale = [55, 62, 65, 73, 82, 98, 110, 131, 147, 165]
    t = 0.0
    while t < duration:
        d = float(RNG.choice([0.18, 0.22, 0.3]))
        f = float(RNG.choice(scale))
        chunk = note(f, d, amp) + note(f * 2, d, amp * 0.5) + note(f * 3, d, amp * 0.25)
        out.append(chunk)
        t += d
    sig = np.concatenate(out)[:int(SR * duration)]
    return sig


def explosion(amp=0.8, dur=0.5):
    n = int(SR * dur)
    noise = RNG.normal(0, 1, n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[freqs > 160] *= 0.05
    sig = np.fft.irfft(spec, n)
    sig /= np.max(np.abs(sig)) or 1
    return amp * np.exp(-np.linspace(0, dur, n) * 6) * sig


def gunshot(amp=0.5, dur=0.10):
    """Mid-band crack: the event we must not lose."""
    n = int(SR * dur)
    noise = RNG.normal(0, 1, n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < 110) | (freqs > 480)] *= 0.04
    sig = np.fft.irfft(spec, n)
    sig /= np.max(np.abs(sig)) or 1
    return amp * np.exp(-np.linspace(0, dur, n) * 38) * sig


def build_scene(duration=24.0, with_music=True, with_explosions=True):
    """Return (mix, shot_times) for a scene with known gunshot positions."""
    mix = np.zeros(int(SR * duration), dtype=np.float64)

    if with_music:
        m = music(duration)
        mix[:len(m)] += m

    if with_explosions:
        t = 1.4
        while t < duration:
            e = explosion()
            i = int(t * SR)
            mix[i:i + len(e)] += e[:len(mix) - i]
            t += float(RNG.uniform(1.1, 2.0))

    shot_times = []
    t = 2.0
    while t < duration - 0.5:
        g = gunshot()
        i = int(t * SR)
        mix[i:i + len(g)] += g[:len(mix) - i]
        shot_times.append(t)
        t += float(RNG.uniform(0.55, 0.95))

    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix *= 0.95 / peak
    return mix.astype(np.float32), shot_times


def bands(subtraction: float, flatness: float = 0.0, sensitivity: float = 1.5,
          refractory: float = 60.0):
    return [
        Band(name="impact", low_hz=20, high_hz=85, sensitivity=sensitivity + 0.2,
             gate=0.0035, refractory_ms=130, background_subtraction=subtraction,
             min_flatness=flatness * 0.5,   # impacts are flat too, but less so
             duration_ms=105, priority=2,
             level_floor_db=-47, level_ceil_db=-21),
        Band(name="gunfire", low_hz=90, high_hz=500, sensitivity=sensitivity,
             gate=0.0028, refractory_ms=refractory, min_share=0.20,
             background_subtraction=subtraction, min_flatness=flatness,
             duration_ms=42, priority=2,
             level_floor_db=-51, level_ceil_db=-25),
    ]


def run(mix, shot_times, subtraction, flatness=0.0, sensitivity=1.5,
        refractory=60.0, tolerance=0.16):
    analyzer = BandAnalyzer(
        bands(subtraction, flatness, sensitivity, refractory), samplerate=SR)

    detections = {"impact": [], "gunfire": []}
    pos = 0
    hop = 512
    clock = 0.0
    while pos < len(mix):
        chunk = mix[pos:pos + hop]
        for onset in analyzer.push(chunk):
            detections[onset.band].append(clock)
        pos += hop
        clock += len(chunk) / SR

    hits = 0
    for t in shot_times:
        if any(abs(d - t) <= tolerance for d in detections["gunfire"]):
            hits += 1

    false_pos = len(detections["gunfire"]) - hits
    return {
        "shots": len(shot_times),
        "hits": hits,
        "recall": hits / max(len(shot_times), 1),
        "gunfire_events": len(detections["gunfire"]),
        "false_pos": max(false_pos, 0),
        "impact_events": len(detections["impact"]),
    }


def sweep(title, mix, shots, configs):
    print(f"\n=== {title} ===")
    print(f"  {'config':<34}{'recall':>8}{'fired':>7}{'false+':>8}{'precision':>11}")
    for label, kwargs in configs:
        r = run(mix, shots, **kwargs)
        precision = r["hits"] / max(r["gunfire_events"], 1)
        print(f"  {label:<34}{r['recall']:>7.0%}{r['gunfire_events']:>7}"
              f"{r['false_pos']:>8}{precision:>10.0%}")


def main():
    print("Gunshot detection in a dense mix.")
    print("recall    = fraction of real gunshots that fired")
    print("precision = fraction of fired pulses that were real gunshots")
    print("Precision is what matters: the motor is a single actuator, so false")
    print("pulses are exactly what drowns out the real ones.")

    mix, shots = build_scene(with_music=True, with_explosions=True)

    sweep("flatness alone", mix, shots, [
        (f"flatness {f}", dict(subtraction=0.0, flatness=f))
        for f in (0.0, 0.25, 0.35, 0.45, 0.55)
    ])

    sweep("sensitivity alone", mix, shots, [
        (f"sensitivity {s}", dict(subtraction=0.0, flatness=0.0, sensitivity=s))
        for s in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    ])

    sweep("flatness 0.45 + sensitivity", mix, shots, [
        (f"flat 0.45 + sens {s}", dict(subtraction=0.0, flatness=0.45, sensitivity=s))
        for s in (1.5, 2.0, 2.5, 3.0, 4.0)
    ])

    sweep("best combos + refractory", mix, shots, [
        ("flat .45 sens 3.0 refr 60", dict(subtraction=0.0, flatness=0.45,
                                           sensitivity=3.0, refractory=60)),
        ("flat .45 sens 3.0 refr 120", dict(subtraction=0.0, flatness=0.45,
                                            sensitivity=3.0, refractory=120)),
        ("flat .45 sens 3.0 refr 200", dict(subtraction=0.0, flatness=0.45,
                                            sensitivity=3.0, refractory=200)),
        ("+ subtraction", dict(subtraction=1.0, flatness=0.45,
                               sensitivity=3.0, refractory=120)),
        ("flat .55 sens 4.0 refr 120", dict(subtraction=0.0, flatness=0.55,
                                            sensitivity=4.0, refractory=120)),
    ])


if __name__ == "__main__":
    main()
