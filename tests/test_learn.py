"""Tests for the event-capture and spectral-profiling path.

These exercise the whole learn -> analyze -> suggest chain against synthetic
events with known frequency content, so a regression in band derivation shows up
without needing a game running.
"""

import numpy as np
import pytest

from z2haptics.learn import (
    LearnSession,
    RingBuffer,
    analyze_session,
    suggest_bands,
)

SR = 48000


def narrowband(centre: float, dur: float, sr: int = SR, amp: float = 0.5, width: float = 60.0):
    """A decaying burst of energy concentrated near `centre` Hz."""
    n = int(sr * dur)
    rng = np.random.default_rng(int(centre))
    noise = rng.normal(0, 1, n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec[(freqs < centre - width) | (freqs > centre + width)] = 0
    sig = np.fft.irfft(spec, n)
    sig = sig / (np.max(np.abs(sig)) or 1.0)
    env = np.exp(-np.linspace(0, dur, n) * 12)
    return (amp * env * sig).astype(np.float32)


def quiet(dur: float, sr: int = SR):
    rng = np.random.default_rng(99)
    return (rng.normal(0, 1e-4, int(sr * dur))).astype(np.float32)


# -- RingBuffer ---------------------------------------------------------------

def test_ring_buffer_retains_most_recent_audio():
    rb = RingBuffer(SR, 1.0)
    rb.write(np.ones(SR // 2, dtype=np.float32))
    rb.write(np.full(SR // 2, 2.0, dtype=np.float32))
    tail = rb.extract(end_offset=0, length=SR // 2)
    assert np.allclose(tail, 2.0)


def test_ring_buffer_handles_oversized_write():
    rb = RingBuffer(SR, 0.5)
    big = np.arange(SR, dtype=np.float32)
    rb.write(big)
    tail = rb.extract(end_offset=0, length=10)
    assert np.allclose(tail, big[-10:])


def test_ring_buffer_extract_with_offset_looks_backwards():
    rb = RingBuffer(SR, 1.0)
    rb.write(np.ones(SR // 2, dtype=np.float32))
    rb.write(np.zeros(SR // 2, dtype=np.float32))
    seg = rb.extract(end_offset=SR // 2, length=100)
    assert np.allclose(seg, 1.0)


# -- capture + analysis round trip -------------------------------------------

def _run_session(tmp_path, events, labels, blocks_of=512):
    """Feed audio through a LearnSession, marking at the right moments."""
    session = LearnSession(
        name="t", labels=labels, samplerate=SR,
        pre_roll_s=0.4, post_roll_s=0.1, buffer_s=3.0, root=tmp_path,
    )

    for label, audio in events:
        # Lead-in so the pre-roll has context, then the event, then the mark.
        for chunk in np.array_split(quiet(0.5), max(1, int(SR * 0.5) // blocks_of)):
            session.on_audio(chunk)
        for chunk in np.array_split(audio, max(1, len(audio) // blocks_of)):
            session.on_audio(chunk)
        if label:
            session.mark(label)
        for chunk in np.array_split(quiet(0.3), max(1, int(SR * 0.3) // blocks_of)):
            session.on_audio(chunk)

    session.flush()
    return session


def test_marks_produce_saved_samples(tmp_path):
    events = [("gunshot", narrowband(300, 0.15)) for _ in range(3)]
    session = _run_session(tmp_path, events, ["gunshot"])

    assert session.counts["gunshot"] == 3
    assert (session.dir / "session.json").exists()
    assert (session.dir / "gunshot_001.wav").exists()
    assert len(session.samples) == 3


def test_unmarked_audio_is_not_saved(tmp_path):
    session = _run_session(tmp_path, [(None, narrowband(300, 0.15))], ["gunshot"])
    assert session.counts["gunshot"] == 0


def test_reopening_a_session_does_not_overwrite_captures(tmp_path):
    """Capturing a weapon set takes several sittings.

    Restarting numbering at 001 would silently destroy the previous sitting's
    work, and the loss would only surface when the templates came out wrong.
    """
    events = [("gunshot", narrowband(300, 0.15)) for _ in range(3)]
    first = _run_session(tmp_path, events, ["gunshot"])
    assert first.counts["gunshot"] == 3

    second = LearnSession(
        name="t", labels=["gunshot"], samplerate=SR,
        pre_roll_s=0.4, post_roll_s=0.1, buffer_s=3.0, root=tmp_path,
    )
    assert second.counts["gunshot"] == 3, "reopened session restarted numbering"

    for chunk in np.array_split(quiet(0.5), 40):
        second.on_audio(chunk)
    for chunk in np.array_split(narrowband(300, 0.15), 10):
        second.on_audio(chunk)
    second.mark("gunshot")
    second.flush()

    files = sorted(p.name for p in second.dir.glob("gunshot_*.wav"))
    assert len(files) == 4, f"expected 4 captures, found {files}"
    assert "gunshot_004.wav" in files


def test_resume_recovers_from_a_killed_session(tmp_path):
    """A session killed rather than stopped leaves WAVs but stale metadata."""
    events = [("gunshot", narrowband(300, 0.15)) for _ in range(2)]
    first = _run_session(tmp_path, events, ["gunshot"])
    (first.dir / "session.json").unlink()

    second = LearnSession(
        name="t", labels=["gunshot"], samplerate=SR, root=tmp_path,
    )
    assert second.counts["gunshot"] == 2, "filenames on disk were ignored"


def test_analysis_recovers_each_label(tmp_path):
    events = (
        [("laser", narrowband(3000, 0.12)) for _ in range(3)]
        + [("gunshot", narrowband(220, 0.15)) for _ in range(3)]
    )
    session = _run_session(tmp_path, events, ["laser", "gunshot"])
    result = analyze_session(session.dir)

    assert set(result["spectra"]) == {"laser", "gunshot"}
    assert result["spectra"]["laser"].count == 3
    assert result["spectra"]["gunshot"].count == 3


def test_spectral_peak_matches_the_event_frequency(tmp_path):
    """The measurement that everything downstream depends on."""
    events = [("laser", narrowband(3000, 0.12)) for _ in range(3)]
    session = _run_session(tmp_path, events, ["laser"])
    spec = analyze_session(session.dir)["spectra"]["laser"]
    assert 2400 < spec.peak_hz < 3600, f"peak at {spec.peak_hz:.0f}Hz, expected ~3000"


def test_two_events_are_spectrally_distinguishable(tmp_path):
    """A low event and a high event must land in different suggested bands."""
    events = (
        [("thump", narrowband(70, 0.2, width=30)) for _ in range(3)]
        + [("laser", narrowband(4000, 0.12, width=200)) for _ in range(3)]
    )
    session = _run_session(tmp_path, events, ["thump", "laser"])
    spectra = analyze_session(session.dir)["spectra"]

    thump = suggest_bands(spectra["thump"], None, max_bands=1, contrast_db=4.0)
    laser = suggest_bands(spectra["laser"], None, max_bands=1, contrast_db=4.0)

    assert thump and laser
    assert thump[0]["high_hz"] < laser[0]["low_hz"], (
        f"bands overlap: thump {thump[0]}, laser {laser[0]}"
    )


def test_suggested_band_brackets_the_event(tmp_path):
    events = [("laser", narrowband(3000, 0.12, width=150)) for _ in range(3)]
    session = _run_session(tmp_path, events, ["laser"])
    spec = analyze_session(session.dir)["spectra"]["laser"]

    bands = suggest_bands(spec, None, max_bands=1, contrast_db=4.0)
    assert bands
    assert bands[0]["low_hz"] <= 3000 <= bands[0]["high_hz"]


def test_ambient_is_used_as_the_contrast_reference(tmp_path):
    events = (
        [("ambient", quiet(0.4)) for _ in range(2)]
        + [("laser", narrowband(3000, 0.12)) for _ in range(3)]
    )
    session = _run_session(tmp_path, events, ["laser"])
    spectra = analyze_session(session.dir)["spectra"]
    assert "ambient" in spectra

    bands = suggest_bands(spectra["laser"], spectra["ambient"].mean_db,
                          max_bands=1, contrast_db=4.0)
    assert bands
    assert bands[0]["low_hz"] <= 3000 <= bands[0]["high_hz"]


def test_suggestions_stay_inside_the_usable_range(tmp_path):
    events = [("laser", narrowband(3000, 0.12)) for _ in range(2)]
    session = _run_session(tmp_path, events, ["laser"])
    spec = analyze_session(session.dir)["spectra"]["laser"]
    for b in suggest_bands(spec, None, max_bands=4, contrast_db=2.0):
        assert b["low_hz"] >= 20
        assert b["high_hz"] <= 14100


def test_high_contrast_threshold_yields_nothing(tmp_path):
    events = [("laser", narrowband(3000, 0.12)) for _ in range(2)]
    session = _run_session(tmp_path, events, ["laser"])
    spec = analyze_session(session.dir)["spectra"]["laser"]
    assert suggest_bands(spec, None, contrast_db=500.0) == []


def test_analyze_rejects_a_missing_session(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_session(tmp_path / "nope")
