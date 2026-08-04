"""
Leave-one-out cross-validation of a weapon set against its own captures.

Building a template from every sample and then testing on those same samples
would flatter the classifier badly -- each probe would be compared against a
template it helped create. Holding each sample out in turn and rebuilding
without it gives an honest estimate of how it will behave on the next shot it
has never seen.

    python tools/validate_weapons.py bf6
"""

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from z2haptics.analysis import spectral_feature  # noqa: E402
from z2haptics.learn import MIN_USABLE_PEAK, SESSIONS_DIR, _peak_spectrum  # noqa: E402
from z2haptics.weapons import WeaponSet, build_template  # noqa: E402


def load_features(session_dir: Path, nfft: int = 4096):
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    sr = meta["samplerate"]
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)

    by_label: dict[str, list[np.ndarray]] = {}
    for s in meta["samples"]:
        if s["label"] == "ambient":
            continue
        path = session_dir / s["filename"]
        if not path.exists():
            continue
        with wave.open(str(path), "rb") as w:
            raw = w.readframes(w.getnframes())
        seg = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        if seg.size == 0 or float(np.max(np.abs(seg))) < MIN_USABLE_PEAK:
            continue
        by_label.setdefault(s["label"], []).append(
            spectral_feature(_peak_spectrum(seg, nfft), freqs))
    return by_label


def cross_validate(by_label, min_confidence, min_margin):
    labels = sorted(by_label)
    confusion = {a: {b: 0 for b in labels + ["(declined)"]} for a in labels}

    for held_label in labels:
        for i in range(len(by_label[held_label])):
            probe = by_label[held_label][i]

            templates = []
            for label in labels:
                feats = ([f for j, f in enumerate(by_label[label]) if j != i]
                         if label == held_label else by_label[label])
                if feats:
                    templates.append(build_template(label, feats))

            ws = WeaponSet(name="cv", templates=templates,
                           min_confidence=min_confidence, min_margin=min_margin)
            match, _ = ws.classify(probe)
            confusion[held_label]["(declined)" if match is None else match.name] += 1

    return labels, confusion


def report(labels, confusion):
    cols = labels + ["(declined)"]
    print(f"\n{'actual \\ called':<16}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (16 + 12 * len(cols)))

    right = wrong = declined = 0
    for a in labels:
        row = confusion[a]
        print(f"{a:<16}" + "".join(f"{row[c]:>12}" for c in cols))
        for c in cols:
            if c == "(declined)":
                declined += row[c]
            elif c == a:
                right += row[c]
            else:
                wrong += row[c]

    total = right + wrong + declined
    if not total:
        return 0.0, 0.0, 0.0
    return right / total, wrong / total, declined / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", default="bf6")
    args = ap.parse_args()

    session_dir = (Path(args.session) if Path(args.session).is_dir()
                   else SESSIONS_DIR / args.session)
    by_label = load_features(session_dir)
    if not by_label:
        print("no usable samples", file=sys.stderr)
        return 1

    print(f"Session: {session_dir}")
    for label, feats in sorted(by_label.items()):
        print(f"  {label:<10} {len(feats)} usable samples")

    print("\n=== leave-one-out cross-validation, current thresholds ===")
    labels, confusion = cross_validate(by_label, 0.75, 0.20)
    right, wrong, declined = report(labels, confusion)
    print(f"\n  correct {right:.0%}   WRONG {wrong:.0%}   declined {declined:.0%}")

    print("\n=== threshold sweep ===")
    print(f"  {'confidence':>11}{'margin':>8}{'correct':>10}{'WRONG':>8}{'declined':>10}")
    best = None
    for conf in (0.0, 0.2, 0.4, 0.55, 0.65, 0.75, 0.85):
        for margin in (0.0, 0.05, 0.10, 0.20, 0.30):
            _, cm = cross_validate(by_label, conf, margin)
            r, w, d = report_quiet(labels, cm)
            print(f"  {conf:>11.2f}{margin:>8.2f}{r:>9.0%}{w:>8.0%}{d:>10.0%}")
            if w <= 0.05 and (best is None or r > best[2]):
                best = (conf, margin, r, w, d)

    if best:
        print(f"\nBest with error <=5%: confidence {best[0]}, margin {best[1]} "
              f"-> {best[2]:.0%} correct, {best[3]:.0%} wrong, {best[4]:.0%} declined")
    return 0


def report_quiet(labels, confusion):
    right = wrong = declined = 0
    for a in labels:
        for c, n in confusion[a].items():
            if c == "(declined)":
                declined += n
            elif c == a:
                right += n
            else:
                wrong += n
    total = right + wrong + declined or 1
    return right / total, wrong / total, declined / total


if __name__ == "__main__":
    sys.exit(main())
