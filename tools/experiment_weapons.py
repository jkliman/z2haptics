"""
Measure whether the weapon classifier can actually tell guns apart.

Builds a set of synthetic weapons with distinct but overlapping character --
the way real guns differ, not the way textbook test tones differ -- then scores
classification accuracy under increasingly unfair conditions:

  clean          isolated shots, the best case
  varied         shot-to-shot variation, as real weapons have
  distant        quieter and duller, as the same gun sounds across the map
  over music     the dense-mix case the rest of this project fights

Accuracy alone is the wrong measure for haptics: confidently firing the wrong
weapon's pulse feels worse than falling back to a generic one. So the confusion
between "wrong" and "declined to guess" is reported separately.

    python tools/experiment_weapons.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from z2haptics.analysis import spectral_feature  # noqa: E402
from z2haptics.weapons import WeaponSet, build_template, separability  # noqa: E402

SR = 48000
NFFT = 4096


def shot(rng, body_hz, crack_hz, body_q, decay, bright, amp=0.7, dur=0.16):
    """A synthetic gunshot: low body thump plus a filtered noise crack.

    Weapons differ mainly in where the body sits, how bright the crack is, and
    how fast it decays -- so those are the parameters varied here.
    """
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)

    body = np.sin(2 * np.pi * body_hz * t) * np.exp(-t * body_q)

    noise = rng.normal(0, 1, n)
    spec = np.fft.rfft(noise)
    fr = np.fft.rfftfreq(n, 1 / SR)
    shape = np.exp(-((fr - crack_hz) ** 2) / (2 * (crack_hz * bright) ** 2))
    spec *= shape
    crack = np.fft.irfft(spec, n)
    crack /= np.max(np.abs(crack)) or 1

    sig = 0.6 * body + 0.9 * crack
    sig *= np.exp(-t * decay)
    sig /= np.max(np.abs(sig)) or 1
    return (amp * sig).astype(np.float32)


# body_hz, crack_hz, body_q, decay, bright
WEAPONS = {
    "ak74":    dict(body_hz=95,  crack_hz=1800, body_q=40, decay=26, bright=0.55),
    "m4":      dict(body_hz=130, crack_hz=2600, body_q=45, decay=30, bright=0.50),
    "lmg":     dict(body_hz=80,  crack_hz=1400, body_q=32, decay=20, bright=0.65),
    "sniper":  dict(body_hz=65,  crack_hz=900,  body_q=22, decay=12, bright=0.75),
    "shotgun": dict(body_hz=110, crack_hz=700,  body_q=28, decay=16, bright=0.90),
    "pistol":  dict(body_hz=160, crack_hz=3200, body_q=55, decay=38, bright=0.45),
}


def jitter(params, rng, amount):
    """Shot-to-shot variation. No two pulls of a trigger sound identical."""
    out = dict(params)
    for key in ("body_hz", "crack_hz", "decay", "bright"):
        out[key] = params[key] * (1.0 + rng.normal(0, amount))
    return out


def feature_of(sig):
    if sig.size < NFFT:
        sig = np.pad(sig, (0, NFFT - sig.size))
    frame = sig[:NFFT] * np.hanning(NFFT)
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    return spectral_feature(spectrum, freqs)


def music_bed(n, rng, amp=0.25):
    out = np.zeros(n, dtype=np.float32)
    t = 0
    while t < n:
        d = int(SR * float(rng.choice([0.14, 0.2, 0.26])))
        f = float(rng.choice([110, 131, 165, 196, 220]))
        tt = np.linspace(0, d / SR, d, endpoint=False)
        seg = amp * (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(4 * np.pi * f * tt))
        seg[:int(SR * 0.02)] *= np.linspace(0, 1, int(SR * 0.02))
        out[t:t + len(seg)] += seg[:max(0, n - t)].astype(np.float32)
        t += d
    return out


def build_set(rng, n_train=6, train_jitter=0.05):
    templates = []
    for name, params in WEAPONS.items():
        feats = [feature_of(shot(rng, **jitter(params, rng, train_jitter)))
                 for _ in range(n_train)]
        templates.append(build_template(name, feats))
    return WeaponSet(name="synthetic", templates=templates)


def evaluate(ws, rng, n_test=25, test_jitter=0.05, amp=0.7, dull=0.0, music=0.0):
    right = wrong = declined = 0
    for name, params in WEAPONS.items():
        for _ in range(n_test):
            p = jitter(params, rng, test_jitter)
            if dull:
                # Distance rolls off the top end and softens the crack.
                p["crack_hz"] *= (1.0 - dull)
                p["bright"] *= (1.0 + dull)
            sig = shot(rng, amp=amp, **p)
            if music:
                bed = music_bed(len(sig), rng, amp=music)
                sig = (sig + bed).astype(np.float32)
            match, _score = ws.classify(feature_of(sig))
            if match is None:
                declined += 1
            elif match.name == name:
                right += 1
            else:
                wrong += 1
    total = right + wrong + declined
    return right / total, wrong / total, declined / total


def main():
    rng = np.random.default_rng(4)
    ws = build_set(rng)

    print("Template consistency (1.00 = every capture identical):")
    for t in ws.templates:
        print(f"  {t.name:<9} {t.spread:.3f}")

    print("\nMost confusable pairs (1.00 = indistinguishable):")
    for a, b, sim in separability(ws)[:5]:
        print(f"  {a:<9} vs {b:<9} {sim:5.2f}")

    conditions = [
        ("clean", {}),
        ("varied (2x jitter)", dict(test_jitter=0.10)),
        ("quiet (0.25 amp)", dict(amp=0.25)),
        ("distant (dulled)", dict(dull=0.25)),
        ("over music", dict(music=0.25)),
        ("distant + music", dict(dull=0.25, music=0.25)),
    ]

    def table(header):
        print(f"\n{header}")
        print(f"{'condition':<26}{'correct':>9}{'WRONG':>8}{'declined':>10}")
        print("-" * 53)
        worst_wrong = 0.0
        for label, kwargs in conditions:
            right, wrong, declined = evaluate(ws, rng, **kwargs)
            worst_wrong = max(worst_wrong, wrong)
            print(f"  {label:<24}{right:>8.0%}{wrong:>8.0%}{declined:>10.0%}")
        return worst_wrong

    table(f"min_confidence={ws.min_confidence}  min_margin={ws.min_margin}")

    # For haptics, a confidently wrong weapon feels worse than a generic hit, so
    # thresholds are chosen to bound the error rate rather than to maximise
    # accuracy. Sweep to find the loosest pair that keeps errors acceptable.
    print("\n\n=== threshold sweep (worst-case error across all conditions) ===")
    print(f"  {'confidence':>11}{'margin':>9}{'worst WRONG':>13}{'clean correct':>15}")
    best = None
    for conf in (0.55, 0.65, 0.75, 0.80, 0.85):
        for margin in (0.04, 0.10, 0.15, 0.20):
            ws.min_confidence, ws.min_margin = conf, margin
            worst = max(evaluate(ws, rng, **kw)[1] for _, kw in conditions)
            clean = evaluate(ws, rng)[0]
            print(f"  {conf:>11.2f}{margin:>9.2f}{worst:>12.0%}{clean:>15.0%}")
            if worst <= 0.06 and (best is None or clean > best[2]):
                best = (conf, margin, clean)

    if best:
        ws.min_confidence, ws.min_margin, _ = best
        print(f"\nLoosest pair keeping worst-case error <=6%: "
              f"min_confidence={best[0]}, min_margin={best[1]}")
        table(f"chosen: min_confidence={best[0]}  min_margin={best[1]}")
    else:
        print("\nNo threshold pair kept worst-case error at or below 6%.")

    print("\n'declined' means no template matched confidently, so the band's own")
    print("pulse shape is used. That is the safe outcome -- a confidently wrong")
    print("weapon feels worse than a generic hit.")


if __name__ == "__main__":
    main()
