"""
Find which loopback device actually carries game audio.

A machine with a virtual mixer (Elgato Wave Link, VoiceMeeter) exposes many
outputs, and the Windows default is often not the one a game plays through.
Capturing the wrong one yields files that look valid but contain only ambience.

Listens to every loopback device at once and reports peak level and crest factor
(peak / median), which is what separates "gunfire is happening here" from
"something is quietly playing here".

    python tools/find_game_audio.py --seconds 12

Fire continuously while it runs.
"""

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import soundcard as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 48000
BLOCK = 2048


def listen(mic, seconds, results, lock):
    slices = []
    try:
        with mic.recorder(samplerate=SR, channels=2, blocksize=BLOCK) as rec:
            n_blocks = int(seconds * SR / BLOCK)
            for _ in range(n_blocks):
                data = rec.record(numframes=BLOCK)
                mono = data.mean(axis=1) if data.ndim > 1 else data
                slices.append(float(np.sqrt(np.mean(mono ** 2))))
    except Exception as e:
        with lock:
            results[mic.name] = ("error", str(e)[:40], 0.0, 0.0)
        return

    if not slices:
        with lock:
            results[mic.name] = ("no data", "", 0.0, 0.0)
        return

    arr = np.array(slices)
    peak = float(arr.max())
    median = float(np.median(arr))
    crest = peak / max(median, 1e-9)
    with lock:
        results[mic.name] = ("ok", "", peak, crest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    mics = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
    if not mics:
        print("no loopback devices found")
        return 1

    try:
        default = sc.default_speaker().name
    except Exception:
        default = ""

    print(f"Listening to {len(mics)} loopback devices for {args.seconds:.0f}s.")
    print("FIRE CONTINUOUSLY NOW.\n")

    results, lock = {}, threading.Lock()
    threads = [threading.Thread(target=listen, args=(m, args.seconds, results, lock),
                                daemon=True) for m in mics]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.seconds + 10)

    print(f"{'device':<52}{'peak':>9}{'crest':>8}   verdict")
    print("-" * 88)

    rows = []
    for mic in mics:
        status, err, peak, crest = results.get(mic.name, ("no result", "", 0.0, 0.0))
        rows.append((mic.name, status, err, peak, crest))

    rows.sort(key=lambda r: (r[3] * min(r[4], 50)), reverse=True)

    for name, status, err, peak, crest in rows:
        tag = " (default)" if name == default else ""
        label = (name + tag)[:50]
        if status != "ok":
            print(f"{label:<52}{'-':>9}{'-':>8}   {status} {err}")
            continue
        if peak < 0.002:
            verdict = "silent"
        elif crest > 6:
            verdict = "<<< GAME AUDIO (sharp transients)"
        elif crest > 3:
            verdict = "some transients"
        else:
            verdict = "steady sound only"
        print(f"{label:<52}{peak:>9.4f}{crest:>8.1f}   {verdict}")

    best = next((r for r in rows if r[1] == "ok" and r[3] >= 0.002 and r[4] > 6), None)
    if best:
        print(f"\nUse this device:\n  --device \"{best[0]}\"")
    else:
        print("\nNo device showed sharp transients. Either nothing was firing during")
        print("the test, or game audio is routed somewhere loopback cannot see.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
