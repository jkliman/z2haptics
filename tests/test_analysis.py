"""Tests for band analysis and onset strength mapping."""

import numpy as np
import pytest

from z2haptics.analysis import Band, BandAnalyzer, to_mono

SR = 48000


def make_band(**kw) -> Band:
    # A deliberately wide dB window. Band level is diluted across the band's bins,
    # so a narrow window clamps quiet test tones to 0 and hides ordering bugs.
    # Real profiles anchor the floor to their gate instead -- see retune_profiles.
    defaults = dict(
        name="test", low_hz=40, high_hz=200, sensitivity=1.5, gate=0.0,
        refractory_ms=0.0, level_floor_db=-120.0, level_ceil_db=-10.0,
    )
    defaults.update(kw)
    return Band(**defaults)


def tone(freq: float, dur: float, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(dur: float) -> np.ndarray:
    return np.zeros(int(SR * dur), dtype=np.float32)


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [0.5, 0.5, 0.5])


def test_to_mono_passes_through_mono():
    mono = np.array([0.1, 0.2], dtype=np.float32)
    assert np.allclose(to_mono(mono), [0.1, 0.2])


def test_silence_produces_no_onsets():
    a = BandAnalyzer([make_band(gate=0.001)], samplerate=SR)
    assert a.push(silence(1.0)) == []


def test_transient_produces_an_onset():
    a = BandAnalyzer([make_band()], samplerate=SR)
    a.push(silence(0.5))                 # learn the floor
    onsets = a.push(tone(100, 0.3))
    assert len(onsets) >= 1
    assert onsets[0].band == "test"


def test_gate_suppresses_quiet_events():
    """A signal below the gate must never trigger, however sharp its onset."""
    loud = BandAnalyzer([make_band(gate=0.0)], samplerate=SR)
    loud.push(silence(0.5))
    assert len(loud.push(tone(100, 0.3, amp=0.001))) >= 1

    gated = BandAnalyzer([make_band(gate=0.5)], samplerate=SR)
    gated.push(silence(0.5))
    assert gated.push(tone(100, 0.3, amp=0.001)) == []


def test_refractory_limits_onset_rate():
    """Two hits inside the refractory window collapse to one onset."""
    a = BandAnalyzer([make_band(refractory_ms=500.0)], samplerate=SR)
    a.push(silence(0.5))
    n = 0
    for _ in range(4):
        n += len(a.push(tone(100, 0.05)))
        n += len(a.push(silence(0.05)))
    assert n <= 1


def test_louder_event_yields_higher_strength():
    """The core property the dB strength model exists to provide.

    The earlier overshoot-ratio model saturated at 1.0 for every event because
    the adaptive threshold collapses toward zero in quiet passages.
    """
    def strength_for(amp: float) -> float:
        a = BandAnalyzer([make_band()], samplerate=SR)
        a.push(silence(0.5))
        onsets = a.push(tone(100, 0.3, amp=amp))
        assert onsets, f"no onset at amp={amp}"
        return onsets[0].strength

    quiet, mid, loud = strength_for(0.02), strength_for(0.15), strength_for(0.8)
    assert quiet < mid < loud
    assert quiet < 0.99, "quiet event should not saturate"


def test_strength_is_clamped_to_unit_range():
    a = BandAnalyzer([make_band(level_floor_db=-10.0, level_ceil_db=-9.0)], samplerate=SR)
    a.push(silence(0.5))
    for onset in a.push(tone(100, 0.3, amp=0.9)):
        assert 0.0 <= onset.strength <= 1.0


def test_min_share_rejects_leaked_energy():
    """A band demanding most of the frame's flux ignores spectral leakage."""
    bands = [
        make_band(name="low", low_hz=40, high_hz=90, min_share=0.9),
        make_band(name="mid", low_hz=90, high_hz=400),
    ]
    a = BandAnalyzer(bands, samplerate=SR)
    a.push(silence(0.5))
    fired = {o.band for o in a.push(tone(200, 0.3))}
    assert "mid" in fired
    assert "low" not in fired


def test_reconfigure_swaps_bands():
    a = BandAnalyzer([make_band(name="a")], samplerate=SR)
    a.reconfigure([make_band(name="b", low_hz=40, high_hz=200)])
    a.push(silence(0.5))
    onsets = a.push(tone(100, 0.3))
    assert all(o.band == "b" for o in onsets)


def test_band_bins_cover_requested_range():
    freqs = np.fft.rfftfreq(2048, 1 / SR)
    lo, hi = make_band(low_hz=100, high_hz=500).bins(freqs)
    assert freqs[lo] >= 100 - (SR / 2048)
    assert freqs[hi - 1] <= 500 + (SR / 2048)
    assert hi > lo


@pytest.mark.parametrize("hop", [256, 512, 1024])
def test_various_hop_sizes_detect(hop):
    a = BandAnalyzer([make_band()], samplerate=SR, hop_size=hop)
    a.push(silence(0.5))
    assert len(a.push(tone(100, 0.3))) >= 1
