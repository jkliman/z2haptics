"""
Frequency-band energy tracking and per-band onset detection.

The engine is onset-triggered rather than amplitude-following: we detect the
*attack* of an event (a gunshot, an explosion, a footstep) and fire one shaped
pulse for it. That keeps discrete events feeling discrete, leaves the motor idle
between hits, and matches what a small ERM/LRA actuator can actually reproduce.

Detection is per-band spectral flux against an adaptive threshold:

  1. Window each frame, take the real FFT, get magnitudes.
  2. Sum magnitudes across each band's bins to get band energy.
  3. Flux = the positive change in band energy since the previous frame.
     Positive-only, so decays never trigger.
  4. Compare flux against a rolling median of recent flux, scaled by the band's
     sensitivity. A median tracks the noise floor without being dragged around
     by the very transients we are trying to detect, which a mean would be.
  5. Require an absolute level gate and a per-band refractory period.

Onset strength is reported as how far the flux overshot its threshold, which is
what the engine maps onto vibration strength.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Band:
    """One frequency band and its detection tuning."""

    name: str
    low_hz: float
    high_hz: float

    # Detection
    sensitivity: float = 1.6      # flux must exceed sensitivity * rolling median
    gate: float = 0.002           # absolute RMS floor; below this the band is silent
    refractory_ms: float = 90.0   # minimum spacing between onsets in this band
    min_share: float = 0.0        # min fraction of the frame's total flux (0 = off)

    # Pulse shaping. Strength comes from how LOUD the event is, measured in dB
    # against this band's own window, not from how far the flux overshot its
    # threshold. Overshoot is near-useless as a strength signal: the adaptive
    # threshold collapses toward zero during quiet passages, so the ratio
    # saturates and every event reads as maximum intensity.
    duration_ms: int = 60         # motor on-time for this band's pulses
    strength_min: int = 25        # strength at level_floor_db
    strength_max: int = 100       # strength at level_ceil_db
    level_floor_db: float = -55.0  # quiet events map here
    level_ceil_db: float = -20.0   # loud events map here
    priority: int = 0             # higher wins when the queue is contended

    enabled: bool = True

    def bins(self, freqs: np.ndarray) -> tuple[int, int]:
        lo = int(np.searchsorted(freqs, self.low_hz, side="left"))
        hi = int(np.searchsorted(freqs, self.high_hz, side="right"))
        return lo, max(hi, lo + 1)


@dataclass
class Onset:
    """A detected transient in one band."""

    band: str
    strength: float        # 0..1, normalised loudness between the band's dB window
    level: float           # band RMS at detection time
    level_db: float        # the same level in dBFS
    flux: float
    threshold: float
    share: float           # this band's fraction of the frame's total flux


@dataclass
class BandState:
    """Rolling detector state for a single band."""

    prev_energy: float = 0.0
    flux_history: deque = field(default_factory=lambda: deque(maxlen=43))  # ~0.5s at 512 hop
    last_onset_s: float = -1e9
    level: float = 0.0
    last_flux: float = 0.0
    last_threshold: float = 0.0


class BandAnalyzer:
    """Sliding-window FFT analyzer that emits onsets per configured band.

    Frames overlap: `frame_size` samples of context advanced by `hop_size` each
    call. 2048/512 at 48kHz gives ~23Hz resolution with a 10.7ms hop -- fine
    enough to separate bass bins, fast enough that onsets do not feel late.
    """

    def __init__(
        self,
        bands: list[Band],
        samplerate: int = 48000,
        frame_size: int = 2048,
        hop_size: int = 512,
    ):
        self.bands = bands
        self.samplerate = samplerate
        self.frame_size = frame_size
        self.hop_size = hop_size

        self.window = np.hanning(frame_size).astype(np.float32)
        self.freqs = np.fft.rfftfreq(frame_size, 1.0 / samplerate)
        self._bin_ranges = {b.name: b.bins(self.freqs) for b in bands}

        self._buf = np.zeros(frame_size, dtype=np.float32)
        self._filled = 0
        self._state = {b.name: BandState() for b in bands}
        self._clock = 0.0  # seconds of audio consumed

    @property
    def state(self) -> dict[str, BandState]:
        return self._state

    def reconfigure(self, bands: list[Band]) -> None:
        """Swap the band set (on profile change) without dropping the audio buffer."""
        self.bands = bands
        self._bin_ranges = {b.name: b.bins(self.freqs) for b in bands}
        for b in bands:
            self._state.setdefault(b.name, BandState())

    def push(self, mono: np.ndarray) -> list[Onset]:
        """Feed mono samples, return any onsets detected in them."""
        onsets: list[Onset] = []
        pos = 0
        n = len(mono)

        while pos < n:
            need = self.hop_size - self._filled if self._filled < self.hop_size else 0
            take = min(need if need else self.hop_size, n - pos)

            # Slide the frame left by `take` and append new samples at the end.
            self._buf = np.roll(self._buf, -take)
            self._buf[-take:] = mono[pos:pos + take]
            pos += take
            self._filled += take

            if self._filled >= self.hop_size:
                self._filled = 0
                self._clock += self.hop_size / self.samplerate
                onsets.extend(self._analyze_frame())

        return onsets

    def _analyze_frame(self) -> list[Onset]:
        spectrum = np.abs(np.fft.rfft(self._buf * self.window))
        onsets: list[Onset] = []

        # First pass: flux per band, so `min_share` can compare each band against
        # the frame total and reject energy that merely leaked in from elsewhere.
        fluxes: dict[str, float] = {}
        for band in self.bands:
            if not band.enabled:
                continue
            lo, hi = self._bin_ranges[band.name]
            slice_ = spectrum[lo:hi]
            if slice_.size == 0:
                fluxes[band.name] = 0.0
                continue
            fluxes[band.name] = max(0.0, float(slice_.sum()) - self._state[band.name].prev_energy)
        total_flux = sum(fluxes.values()) or 1e-12

        for band in self.bands:
            if not band.enabled:
                continue
            st = self._state[band.name]
            lo, hi = self._bin_ranges[band.name]
            slice_ = spectrum[lo:hi]
            if slice_.size == 0:
                continue

            energy = float(slice_.sum())
            # Band RMS, normalised by bin count so bands of different widths
            # sit on a comparable scale.
            st.level = float(np.sqrt(np.mean(slice_ ** 2))) / self.frame_size * 2.0

            flux = max(0.0, energy - st.prev_energy)
            st.prev_energy = energy
            st.last_flux = flux

            hist = st.flux_history
            if len(hist) >= 8:
                median = float(np.median(hist))
                threshold = max(median * band.sensitivity, 1e-9)
            else:
                threshold = float("inf")  # still learning the floor
            st.last_threshold = threshold
            hist.append(flux)

            if st.level < band.gate:
                continue
            if flux <= threshold:
                continue
            if (self._clock - st.last_onset_s) * 1000.0 < band.refractory_ms:
                continue

            share = flux / total_flux
            if band.min_share > 0.0 and share < band.min_share:
                continue

            st.last_onset_s = self._clock

            # Strength from loudness, mapped across the band's dB window.
            level_db = 20.0 * np.log10(max(st.level, 1e-9))
            span_db = max(band.level_ceil_db - band.level_floor_db, 1e-6)
            norm = (level_db - band.level_floor_db) / span_db

            onsets.append(
                Onset(
                    band=band.name,
                    strength=float(np.clip(norm, 0.0, 1.0)),
                    level=st.level,
                    level_db=float(level_db),
                    flux=flux,
                    threshold=threshold,
                    share=float(share),
                )
            )

        return onsets


def to_mono(data: np.ndarray) -> np.ndarray:
    """Downmix an (n, channels) capture block to mono float32."""
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    return data.mean(axis=1).astype(np.float32, copy=False)
