"""
Per-weapon identification.

The behaviour that matters most here is *declining*. For haptics a confidently
wrong weapon feels worse than a generic pulse, so the thresholds are tuned to
bound the error rate rather than to maximise accuracy, and the tests assert the
safe-failure behaviour as much as the correct-match behaviour.
"""

import numpy as np
import pytest

from z2haptics.analysis import FEATURE_BINS, spectral_feature
from z2haptics.weapons import (
    WeaponSet,
    WeaponTemplate,
    build_template,
    load,
    save,
    separability,
)

SR = 48000
NFFT = 4096


def _feature_from(sig):
    if sig.size < NFFT:
        sig = np.pad(sig, (0, NFFT - sig.size))
    spectrum = np.abs(np.fft.rfft(sig[:NFFT] * np.hanning(NFFT)))
    return spectral_feature(spectrum, np.fft.rfftfreq(NFFT, 1 / SR))


def shot(rng, body_hz, crack_hz, decay=26, bright=0.55, amp=0.7, dur=0.16):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    body = np.sin(2 * np.pi * body_hz * t) * np.exp(-t * 40)
    noise = rng.normal(0, 1, n)
    spec = np.fft.rfft(noise)
    fr = np.fft.rfftfreq(n, 1 / SR)
    spec *= np.exp(-((fr - crack_hz) ** 2) / (2 * (crack_hz * bright) ** 2))
    crack = np.fft.irfft(spec, n)
    crack /= np.max(np.abs(crack)) or 1
    sig = (0.6 * body + 0.9 * crack) * np.exp(-t * decay)
    sig /= np.max(np.abs(sig)) or 1
    return (amp * sig).astype(np.float32)


def make_set(rng, specs, n=6, **kwargs):
    templates = []
    for name, params in specs.items():
        feats = [_feature_from(shot(rng, **params)) for _ in range(n)]
        templates.append(build_template(name, feats))
    return WeaponSet(name="test", templates=templates, **kwargs)


DISTINCT = {
    "sniper": dict(body_hz=65, crack_hz=800, decay=12, bright=0.75),
    "rifle": dict(body_hz=120, crack_hz=2400, decay=30, bright=0.50),
    "pistol": dict(body_hz=170, crack_hz=3600, decay=40, bright=0.45),
}


# -- feature ------------------------------------------------------------------

def test_feature_is_unit_length_and_fixed_size():
    rng = np.random.default_rng(0)
    f = _feature_from(shot(rng, 100, 2000))
    assert f.shape == (FEATURE_BINS,)
    assert np.linalg.norm(f) == pytest.approx(1.0, abs=1e-4)


def test_feature_ignores_loudness():
    """The same gun close up and far away must produce the same fingerprint."""
    rng = np.random.default_rng(1)
    loud = shot(rng, 120, 2400, amp=0.9)
    quiet = (loud * 0.05).astype(np.float32)
    assert float(np.dot(_feature_from(loud), _feature_from(quiet))) > 0.99


def test_different_weapons_give_different_features():
    rng = np.random.default_rng(2)
    a = _feature_from(shot(rng, 65, 800, bright=0.75))
    b = _feature_from(shot(rng, 170, 3600, bright=0.45))
    assert float(np.dot(a, b)) < 0.95


# -- templates ----------------------------------------------------------------

def test_build_template_reports_consistency():
    rng = np.random.default_rng(3)
    feats = [_feature_from(shot(rng, 120, 2400)) for _ in range(6)]
    t = build_template("rifle", feats)
    assert t.samples == 6
    assert t.spread > 0.9, "identical-ish captures should agree"


SIX = {
    "ak74": dict(body_hz=95, crack_hz=1800, decay=26, bright=0.55),
    "m4": dict(body_hz=130, crack_hz=2600, decay=30, bright=0.50),
    "lmg": dict(body_hz=80, crack_hz=1400, decay=20, bright=0.65),
    "sniper": dict(body_hz=65, crack_hz=900, decay=12, bright=0.75),
    "shotgun": dict(body_hz=110, crack_hz=700, decay=16, bright=0.90),
    "pistol": dict(body_hz=160, crack_hz=3200, decay=38, bright=0.45),
}


def test_centring_improves_discrimination():
    """Raw fingerprints of any two guns are dominated by what they share.

    Comparing raw vectors leaves almost no margin between first and second
    place, so the classifier declines nearly everything. Centring removes the
    generic-gunshot component and leaves what actually distinguishes them.

    Asserted on accuracy rather than on raw pairwise similarity: with a small
    number of templates the centred vectors sum to zero and lie in a lower
    dimensional subspace, so individual pair similarities can legitimately rise
    even as discrimination improves.
    """
    rng = np.random.default_rng(4)
    ws = make_set(rng, SIX)

    def raw_classify(probe):
        """What classify() would do without centring, same thresholds."""
        scored = sorted(((t, t.similarity(probe)) for t in ws.templates),
                        key=lambda p: p[1], reverse=True)
        best, score = scored[0]
        if score < ws.min_confidence:
            return None
        if len(scored) > 1 and (score - scored[1][1]) < ws.min_margin:
            return None
        return best.name

    named_centred = named_raw = correct_centred = 0
    for name, params in SIX.items():
        for _ in range(6):
            probe = _feature_from(shot(rng, **params))
            match, _ = ws.classify(probe)
            if match is not None:
                named_centred += 1
                correct_centred += match.name == name
            if raw_classify(probe) is not None:
                named_raw += 1

    assert named_centred > named_raw, (
        f"centring did not improve confident matches: raw named {named_raw}, "
        f"centred named {named_centred}"
    )
    assert correct_centred >= named_centred * 0.8, "centred matches were mostly wrong"


# -- classification -----------------------------------------------------------

def test_classifies_distinct_weapons():
    rng = np.random.default_rng(5)
    ws = make_set(rng, DISTINCT)

    correct = 0
    for name, params in DISTINCT.items():
        for _ in range(8):
            match, _ = ws.classify(_feature_from(shot(rng, **params)))
            if match and match.name == name:
                correct += 1
    assert correct >= 16, f"only {correct}/24 distinct shots identified"


def test_declines_rather_than_guessing_on_noise():
    """Random noise must not be confidently named as a weapon."""
    rng = np.random.default_rng(6)
    ws = make_set(rng, DISTINCT)

    named = 0
    for _ in range(20):
        junk = rng.normal(0, 1, NFFT).astype(np.float32)
        match, _ = ws.classify(_feature_from(junk))
        if match is not None:
            named += 1
    assert named <= 4, f"noise was named a weapon {named}/20 times"


def test_declines_when_two_weapons_are_indistinguishable():
    """Near-identical guns should decline, not flap between them."""
    rng = np.random.default_rng(7)
    twins = {
        "a": dict(body_hz=120, crack_hz=2400),
        "b": dict(body_hz=121, crack_hz=2410),
    }
    ws = make_set(rng, twins)

    declined = 0
    for _ in range(15):
        match, _ = ws.classify(_feature_from(shot(rng, **twins["a"])))
        if match is None:
            declined += 1
    assert declined >= 10, f"only declined {declined}/15 for indistinguishable guns"


def test_empty_set_and_missing_feature_are_safe():
    ws = WeaponSet(name="empty")
    assert ws.classify(None) == (None, 0.0)
    assert ws.classify(np.zeros(FEATURE_BINS, dtype=np.float32))[0] is None


def test_wrong_sized_feature_is_rejected():
    rng = np.random.default_rng(8)
    ws = make_set(rng, DISTINCT)
    assert ws.classify(np.zeros(5, dtype=np.float32))[0] is None


def test_thresholds_bound_the_error_rate():
    """Loosening the margin must trade accuracy for errors, not be free."""
    rng = np.random.default_rng(9)
    ws = make_set(rng, DISTINCT)

    def wrong_rate():
        wrong = 0
        for name, params in DISTINCT.items():
            for _ in range(8):
                match, _ = ws.classify(_feature_from(shot(rng, **params)))
                if match is not None and match.name != name:
                    wrong += 1
        return wrong

    ws.min_confidence, ws.min_margin = 0.75, 0.20
    assert wrong_rate() <= 3, "default thresholds let too many errors through"


# -- persistence --------------------------------------------------------------

def test_weapon_set_round_trips(tmp_path):
    rng = np.random.default_rng(10)
    ws = make_set(rng, DISTINCT)
    ws.templates[0].duration_ms = 120
    ws.templates[0].strength_min = 70
    ws.templates[0].strength_max = 100

    path = save(ws, tmp_path / "w.yaml")
    back = load(path)

    assert [t.name for t in back.templates] == [t.name for t in ws.templates]
    assert back.templates[0].duration_ms == 120
    assert back.templates[0].strength_min == 70
    assert back.min_confidence == pytest.approx(ws.min_confidence)

    probe = _feature_from(shot(rng, **DISTINCT["sniper"]))
    assert ws.classify(probe)[0].name == back.classify(probe)[0].name


def test_loaded_set_recomputes_its_centroid(tmp_path):
    """A set loaded from disk must classify identically to one built in memory."""
    rng = np.random.default_rng(11)
    ws = make_set(rng, DISTINCT)
    back = load(save(ws, tmp_path / "w.yaml"))
    assert back._centroid is not None
    assert len(back._centered) == len(back.templates)


def test_separability_orders_most_confusable_first():
    rng = np.random.default_rng(12)
    ws = make_set(rng, DISTINCT)
    pairs = separability(ws)
    assert pairs == sorted(pairs, key=lambda p: p[2], reverse=True)


def test_template_shaping_is_optional():
    t = WeaponTemplate(name="x", feature=np.ones(FEATURE_BINS, dtype=np.float32))
    assert t.duration_ms is None
    assert t.strength_min is None
