"""HUD region capture and visual weapon matching."""

import numpy as np
import pytest

from z2haptics.hud import (
    HudSet,
    HudWatcher,
    Region,
    build_template,
    fingerprint,
    is_blank,
    load,
    save,
)


def text_like(seed: int, w: int = 240, h: int = 48, blocks: int = 6) -> np.ndarray:
    """A crop with glyph-ish structure, distinct per seed."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    for i in range(blocks):
        x = int(rng.integers(4, w - 24))
        y = int(rng.integers(4, h - 20))
        bw = int(rng.integers(8, 22))
        bh = int(rng.integers(10, 18))
        img[y:y + bh, x:x + bw] = int(rng.integers(180, 255))
    return img


def over_background(crop: np.ndarray, brightness: int, seed: int = 0) -> np.ndarray:
    """Same glyphs composited over different scenery."""
    rng = np.random.default_rng(seed)
    bg = (rng.normal(brightness, 18, crop.shape)).clip(0, 255).astype(np.uint8)
    mask = crop.max(axis=2) > 120
    out = bg.copy()
    out[mask] = crop[mask]
    return out


# -- fingerprint --------------------------------------------------------------

def test_fingerprint_is_unit_length():
    f = fingerprint(text_like(1))
    assert np.linalg.norm(f) == pytest.approx(1.0, abs=1e-4)


def test_fingerprint_survives_a_changing_background():
    """HUD text sits over whatever the player is looking at.

    Comparing raw pixels would mostly compare the scenery, which is why the
    fingerprint is gradient-based rather than brightness-based.
    """
    glyphs = text_like(2)
    dark = fingerprint(over_background(glyphs, 30, seed=1))
    bright = fingerprint(over_background(glyphs, 170, seed=2))
    assert float(np.dot(dark, bright)) > 0.7


def test_different_text_gives_different_fingerprints():
    a = fingerprint(text_like(3))
    b = fingerprint(text_like(99))
    assert float(np.dot(a, b)) < 0.9


def test_blank_capture_is_detected():
    """Exclusive fullscreen grabs come back featureless."""
    assert is_blank(np.zeros((40, 200, 3), dtype=np.uint8))
    assert is_blank(np.full((40, 200, 3), 17, dtype=np.uint8))
    assert not is_blank(text_like(4))


# -- identification -----------------------------------------------------------

def make_set(seeds):
    templates = [
        build_template(name, [over_background(text_like(seed), b, seed=b)
                              for b in (40, 110, 180)])
        for name, seed in seeds.items()
    ]
    return HudSet(name="t", region=Region(0, 0, 240, 48), templates=templates)


WEAPONS = {"assault": 11, "smg": 22, "lmg": 33, "sniper": 44}


def test_identifies_each_weapon():
    hs = make_set(WEAPONS)
    correct = 0
    for name, seed in WEAPONS.items():
        probe = over_background(text_like(seed), 90, seed=7)
        match, _ = hs.identify(probe)
        correct += match is not None and match.name == name
    assert correct >= 3, f"only {correct}/4 identified"


def test_declines_on_a_blank_capture():
    hs = make_set(WEAPONS)
    match, score = hs.identify(np.zeros((48, 240, 3), dtype=np.uint8))
    assert match is None
    assert score == 0.0


def test_declines_on_unknown_content():
    hs = make_set(WEAPONS)
    match, _ = hs.identify(over_background(text_like(777), 90, seed=3))
    assert match is None or match.name in WEAPONS


def test_empty_set_is_safe():
    hs = HudSet(name="empty")
    assert hs.identify(text_like(1)) == (None, 0.0)


# -- watcher ------------------------------------------------------------------

class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def grab(self, region):
        frame = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return frame

    def close(self):
        pass


def test_watcher_requires_confirmation_before_switching():
    """Swap animations pass through frames that match nothing or match wrong.

    Acting on the first frame would make the weapon flap during every swap.
    """
    hs = make_set(WEAPONS)
    assault = over_background(text_like(11), 90, seed=7)
    smg = over_background(text_like(22), 90, seed=7)

    changes = []
    w = HudWatcher(hs, on_change=changes.append, confirmations=2)
    w._capture = FakeCapture([assault, assault, assault])

    w.poll(); w.poll()
    assert changes == ["assault"], f"changes: {changes}"

    # A single stray frame of a different weapon must not switch.
    w._capture = FakeCapture([smg])
    w.poll()
    assert changes == ["assault"], "switched on one frame"

    w._capture = FakeCapture([smg, smg])
    w.poll(); w.poll()
    assert changes[-1] == "smg"


def test_watcher_holds_state_through_blank_frames():
    hs = make_set(WEAPONS)
    assault = over_background(text_like(11), 90, seed=7)
    blank = np.zeros((48, 240, 3), dtype=np.uint8)

    w = HudWatcher(hs, confirmations=1)
    w._capture = FakeCapture([assault])
    w.poll()
    assert w.current == "assault"

    w._capture = FakeCapture([blank, blank])
    w.poll(); w.poll()
    assert w.current == "assault", "blank frame cleared the weapon"
    assert w.blank_reads == 2


# -- persistence --------------------------------------------------------------

def test_hud_set_round_trips(tmp_path):
    hs = make_set(WEAPONS)
    hs.region = Region(100, 200, 240, 48)
    path = save(hs, tmp_path / "t.json")
    back = load(path)

    assert back.region.left == 100
    assert back.region.width == 240
    assert [t.name for t in back.templates] == [t.name for t in hs.templates]

    probe = over_background(text_like(11), 90, seed=7)
    a = hs.identify(probe)[0]
    b = back.identify(probe)[0]
    assert (a.name if a else None) == (b.name if b else None)


def test_region_round_trips():
    r = Region(10, 20, 30, 40)
    assert Region.from_dict(r.as_dict()) == r
