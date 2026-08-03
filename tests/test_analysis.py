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


def _burst(freq, dur, amp, sr=SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    env = np.exp(-t * 30)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise_burst(lo, hi, dur, amp, sr=SR, seed=0):
    """Broadband burst -- spectrally flat, like a gunshot."""
    n = int(sr * dur)
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec[(freqs < lo) | (freqs > hi)] *= 0.02
    sig = np.fft.irfft(spec, n)
    sig /= np.max(np.abs(sig)) or 1
    env = np.exp(-np.linspace(0, dur, n) * 35)
    return (amp * env * sig).astype(np.float32)


def test_sensitivity_actually_changes_detection_count():
    """Regression: the adaptive threshold was inert.

    It used to be median(flux) * sensitivity. Flux is max(0, energy - prev), so
    roughly half of all frames are exactly zero and the median sits at zero --
    the threshold collapsed to its 1e-9 floor and sensitivity had no effect at
    any value between 1.5 and 5.0. Every scrap of positive flux became an onset,
    which is what buried real events under music.
    """
    rng = np.random.default_rng(3)
    busy = np.zeros(int(SR * 6), dtype=np.float32)
    for i in range(60):
        f = float(rng.choice([110, 130, 165, 196, 220]))
        seg = tone(f, 0.09, amp=0.25)
        start = int(i * 0.095 * SR)
        busy[start:start + len(seg)] += seg[:len(busy) - start]

    def count(sensitivity):
        band = make_band(low_hz=90, high_hz=400, gate=0.0, sensitivity=sensitivity)
        return len(BandAnalyzer([band], samplerate=SR).push(busy))

    low, high = count(1.5), count(5.0)
    assert low > high, f"sensitivity had no effect: {low} onsets at 1.5, {high} at 5.0"


def _busy_music(duration, sr=SR, amp=0.30, seed=5):
    """Continuously moving notes -- what actually floods the detector.

    Not a single sustained tone: a steady tone produces almost no flux and
    triggers nothing. It is note-to-note movement that keeps firing onsets.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(int(sr * duration), dtype=np.float32)
    scale = [110, 131, 147, 165, 196, 220, 247]
    t = 0.0
    while t < duration - 0.15:
        d = float(rng.choice([0.12, 0.16, 0.2]))
        f = float(rng.choice(scale))
        seg = tone(f, d, amp) + tone(f * 2, d, amp * 0.5)
        # Real instruments ramp in over a few milliseconds. An instantaneous
        # attack is a discontinuity, which reads as broadband and would make
        # synthetic music look far more gunshot-like than the real thing.
        ramp = np.linspace(0.0, 1.0, int(sr * 0.02), dtype=np.float32)
        seg[:len(ramp)] *= ramp
        i = int(t * sr)
        out[i:i + len(seg)] += seg[:len(out) - i].astype(np.float32)
        t += d
    return out


def test_min_flatness_suppresses_musical_onsets():
    """Music is tonal, gunfire is broadband. Flatness separates them.

    Measured on continuous material, which is how it is actually used -- see the
    limitation note on Band.min_flatness about attacks from silence.

    The effect is real but moderate. Once the adaptive threshold was fixed,
    flatness accounts for roughly a 20-25% cut in false positives on realistic
    material (tools/experiment_masking.py), not the order of magnitude it
    appeared to give while the threshold was inert.
    """
    music = _busy_music(6.0)

    def count(min_flatness):
        band = make_band(low_hz=90, high_hz=450, gate=0.0, sensitivity=1.5,
                         min_flatness=min_flatness)
        return len(BandAnalyzer([band], samplerate=SR).push(music))

    unfiltered, filtered = count(0.0), count(0.45)
    assert unfiltered > 0, "premise: music triggers onsets without the filter"
    assert filtered < unfiltered * 0.9, (
        f"flatness barely helped: {unfiltered} onsets -> {filtered}"
    )


def test_broadband_events_survive_the_flatness_filter():
    """The filter must not cost us the events it is protecting."""
    music = _busy_music(4.0)
    shot = _noise_burst(120, 400, 0.12, amp=0.6)
    scene = music.copy()
    at = int(2.5 * SR)
    scene[at:at + len(shot)] += shot[:len(scene) - at]

    band = make_band(low_hz=90, high_hz=450, gate=0.0, sensitivity=1.5,
                     min_flatness=0.45, refractory_ms=0.0)
    analyzer = BandAnalyzer([band], samplerate=SR)

    fired_near_shot = False
    pos, clock = 0, 0.0
    while pos < len(scene):
        for _ in analyzer.push(scene[pos:pos + 512]):
            if abs(clock - 2.5) < 0.15:
                fired_near_shot = True
        pos += 512
        clock += 512 / SR

    assert fired_near_shot, "broadband burst was rejected by the flatness filter"


def test_background_subtraction_still_detects_in_silence():
    """It must not cost us detections when there is nothing to subtract."""
    band = make_band(gate=0.0, background_subtraction=1.0)
    a = BandAnalyzer([band], samplerate=SR)
    a.push(silence(0.5))
    assert a.push(tone(100, 0.3, amp=0.5))


def test_background_absorbs_sustained_content_not_transients():
    """The asymmetric time constants: steady sound is learned, hits are not."""
    band = make_band(gate=0.0, background_subtraction=1.0)

    sustained = BandAnalyzer([band], samplerate=SR)
    sustained.push(tone(100, 2.5, amp=0.5))
    learned = float(sustained._background.max())

    transient = BandAnalyzer([band], samplerate=SR)
    transient.push(silence(0.6))
    transient.push(_burst(100, 0.1, amp=0.5))
    absorbed = float(transient._background.max())

    assert absorbed < learned * 0.5, (
        f"a brief transient inflated the background as much as sustained tone "
        f"({absorbed:.4f} vs {learned:.4f})"
    )


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
