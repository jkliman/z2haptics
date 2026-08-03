"""
Find what actually separates a gunshot from music in the same frequency band.

The masking experiment showed recall is already 100% -- gunshots are detected.
The problem is that music fires the same band ~10x more often, so the motor
buzzes continuously and real events stop standing out. That makes this a
false-positive problem, and the question becomes: what property distinguishes
the two when they overlap in frequency?

Candidates measured here, all cheap enough for the audio hot path:

  spectral flatness  geometric/arithmetic mean of the band spectrum. Noise-like
                     content (a gunshot) is flat; a tonal bass note concentrates
                     energy in a few harmonics and is far from flat.
  attack slope       how fast band energy rises. Percussive events jump within a
                     frame or two; played notes ramp.
  crest factor       peak/RMS in the time domain.

    python tools/experiment_discriminators.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_masking import SR, build_scene, gunshot, music  # noqa: E402

NFFT = 2048
HOP = 512
LOW, HIGH = 90.0, 500.0


def band_bins(nfft=NFFT):
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    lo = int(np.searchsorted(freqs, LOW, "left"))
    hi = int(np.searchsorted(freqs, HIGH, "right"))
    return lo, max(hi, lo + 1)


def spectral_flatness(mag: np.ndarray) -> float:
    """Geometric mean / arithmetic mean. 1.0 = white noise, ->0 = pure tone."""
    m = np.maximum(mag, 1e-12)
    return float(np.exp(np.mean(np.log(m))) / np.mean(m))


def frame_features(frame: np.ndarray, prev_energy: float):
    window = np.hanning(len(frame))
    mag = np.abs(np.fft.rfft(frame * window))
    lo, hi = band_bins(len(frame))
    band = mag[lo:hi]
    energy = float(band.sum())
    flat = spectral_flatness(band)
    attack = energy / max(prev_energy, 1e-9)
    crest = float(np.max(np.abs(frame)) / (np.sqrt(np.mean(frame ** 2)) + 1e-12))
    return energy, flat, attack, crest


def collect(signal: np.ndarray, at_times=None, label=""):
    """Feature stats over frames, optionally only near given event times."""
    rows = []
    prev = 1e-9
    pos = 0
    clock = 0.0
    while pos + NFFT <= len(signal):
        frame = signal[pos:pos + NFFT]
        energy, flat, attack, crest = frame_features(frame, prev)
        prev = energy
        # Frame centre, since the window straddles the hop.
        centre = clock + (NFFT / 2) / SR
        if at_times is None:
            rows.append((flat, attack, crest, energy))
        else:
            if any(abs(centre - t) <= 0.05 for t in at_times):
                rows.append((flat, attack, crest, energy))
        pos += HOP
        clock += HOP / SR
    return np.array(rows) if rows else np.zeros((0, 4))


def describe(name, rows):
    if not len(rows):
        print(f"  {name:<26} (no frames)")
        return
    flat, attack, crest, energy = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    print(f"  {name:<26}"
          f"flatness {np.median(flat):>6.3f} [{np.percentile(flat, 10):.3f}-{np.percentile(flat, 90):.3f}]   "
          f"attack {np.median(attack):>7.2f}   crest {np.median(crest):>5.2f}")


def main():
    print("Frame features in the gunfire band (90-500Hz).\n")

    # Isolated sources, so the measurements are unambiguous.
    shots = np.zeros(int(SR * 12), dtype=np.float32)
    shot_times = []
    t = 0.5
    while t < 11.0:
        g = gunshot()
        i = int(t * SR)
        shots[i:i + len(g)] += g[:len(shots) - i]
        shot_times.append(t)
        t += 0.9

    mus = music(12.0).astype(np.float32)

    print("isolated sources:")
    describe("gunshot frames", collect(shots, shot_times))
    describe("music frames", collect(mus))

    # Separability: pick a flatness threshold and see how each side falls.
    g_rows = collect(shots, shot_times)
    m_rows = collect(mus)
    print("\nflatness threshold sweep (keep frames with flatness >= t):")
    print(f"  {'t':>6}{'gunshot kept':>15}{'music kept':>13}")
    for thresh in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        gk = float(np.mean(g_rows[:, 0] >= thresh)) if len(g_rows) else 0
        mk = float(np.mean(m_rows[:, 0] >= thresh)) if len(m_rows) else 0
        print(f"  {thresh:>6.2f}{gk:>14.0%}{mk:>13.0%}")

    # And in a realistic combined scene.
    mix, times = build_scene(with_music=True, with_explosions=True)
    print("\nrealistic mix:")
    describe("frames at gunshots", collect(mix, times))
    describe("all frames", collect(mix))


if __name__ == "__main__":
    main()
