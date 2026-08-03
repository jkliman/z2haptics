"""
Build a synthetic learn session so `z2haptics analyze` can be exercised without
a game running. Useful for demos and for checking report formatting changes.

    python tools/make_demo_session.py
    python -m z2haptics.cli analyze demo --write-profile
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from z2haptics.learn import SESSIONS_DIR, LearnSession  # noqa: E402

SR = 48000


def narrowband(centre, dur, amp=0.5, width=120.0, seed=0):
    n = int(SR * dur)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < centre - width) | (freqs > centre + width)] = 0
    sig = np.fft.irfft(spec, n)
    sig = sig / (np.max(np.abs(sig)) or 1.0)
    env = np.exp(-np.linspace(0, dur, n) * 14)
    return (amp * env * sig).astype(np.float32)


def room_tone(dur, seed=99):
    """Broadband low-level ambience, so contrast has a realistic reference."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.0025, int(SR * dur))).astype(np.float32)


def feed(session, audio, block=512):
    for i in range(0, len(audio), block):
        session.on_audio(audio[i:i + block])


def main():
    session = LearnSession(
        name="demo", labels=["gunshot", "laser", "explosion"],
        samplerate=SR, pre_roll_s=0.5, post_roll_s=0.15, buffer_s=3.0,
    )

    events = [
        ("explosion", lambda i: narrowband(65, 0.30, amp=0.85, width=45, seed=i)),
        ("gunshot",   lambda i: narrowband(280, 0.16, amp=0.70, width=180, seed=i + 50)),
        ("laser",     lambda i: narrowband(3600, 0.10, amp=0.55, width=320, seed=i + 90)),
    ]

    for label, gen in events:
        for i in range(4):
            feed(session, room_tone(0.55, seed=i))
            feed(session, gen(i) + room_tone(0.30 + 0.0, seed=i + 7)[:len(gen(i))])
            session.mark(label)
            feed(session, room_tone(0.35, seed=i + 3))

    for i in range(3):
        feed(session, room_tone(0.8, seed=i + 200))
        session.mark("ambient")
        feed(session, room_tone(0.4, seed=i + 300))

    session.flush()
    print(f"wrote {sum(session.counts.values())} samples to {session.dir}")
    print(f"counts: {session.counts}")
    print(f"\nnow run:  python -m z2haptics.cli analyze demo --write-profile")


if __name__ == "__main__":
    main()
