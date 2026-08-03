"""
The haptic engine: loopback audio in, shaped motor pulses out.

Flow per capture block (~10.7ms at the default hop):

    loopback block -> mono -> BandAnalyzer -> onsets -> pulse shaping -> HapticSink

Pulse shaping maps an onset's normalised strength onto the band's configured
[strength_min, strength_max] window, scaled by the profile master. When several
bands fire on the same frame only the highest-priority one is sent -- the motor
is a single actuator, so stacking pulses would just blur two events into one
longer buzz.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .analysis import BandAnalyzer
from .api import HapticSink, Pulse, X1Connection
from .audio import LoopbackCapture
from .foreground import foreground_process
from .profiles import Profile, match_profile

log = logging.getLogger(__name__)

# How much one step of band priority is worth when choosing the frame's winner,
# expressed in units of normalised onset strength (0..1). Small on purpose: it
# should settle ties between similar-strength detections without letting a weak
# leaked onset in a high-priority band beat a strong genuine one.
PRIORITY_WEIGHT = 0.05


@dataclass
class BandStats:
    """Per-band accounting, so it is visible *where* an event was lost.

    "Gunshots do not come through" has three very different causes -- never
    detected, detected but outranked, or detected and rate-limited -- and they
    need opposite fixes. Counting each separately makes that answerable.
    """

    detected: int = 0      # onsets the analyzer reported
    won: int = 0           # frames this band won
    lost: int = 0          # detected but another band won the frame
    capped: int = 0        # excluded because the band hit its own max_rate
    queued: int = 0        # pulses accepted by the sink


@dataclass
class EngineStats:
    onsets: int = 0
    pulses: int = 0
    suppressed_stacking: int = 0
    profile_switches: int = 0
    started_at: float = field(default_factory=time.time)
    last_onsets: list = field(default_factory=list)  # recent (band, strength) for the UI
    bands: dict = field(default_factory=dict)        # band name -> BandStats

    def band(self, name: str) -> BandStats:
        stats = self.bands.get(name)
        if stats is None:
            stats = self.bands[name] = BandStats()
        return stats


class HapticEngine:
    """Drives haptics from system audio according to the active profile."""

    def __init__(
        self,
        profile: Profile,
        profiles: dict[str, Profile] | None = None,
        device_name: str | None = None,
        samplerate: int = 48000,
        frame_size: int = 2048,
        hop_size: int = 512,
        dry_run: bool = False,
        auto_switch: bool = False,
        fallback_profile: Profile | None = None,
        on_event=None,
    ):
        self.profile = profile
        self.profiles = profiles or {}
        self.fallback_profile = fallback_profile
        self.auto_switch = auto_switch
        self.dry_run = dry_run
        self.on_event = on_event

        self.stats = EngineStats()
        self._lock = threading.Lock()

        self.analyzer = BandAnalyzer(
            bands=profile.bands,
            samplerate=samplerate,
            frame_size=frame_size,
            hop_size=hop_size,
        )
        self.sink = HapticSink(
            connection=X1Connection(),
            min_gap_ms=profile.limits.min_gap_ms,
            max_pulses_sec=profile.limits.max_pulses_sec,
            max_duty=profile.limits.max_duty,
            dry_run=dry_run,
        )
        self.capture = LoopbackCapture(
            callback=self._on_audio,
            device_name=device_name,
            samplerate=samplerate,
            blocksize=hop_size,
        )

        self._stop = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._current_process = ""
        self._band_wins: dict[str, list[float]] = {}   # band -> recent win timestamps

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self.sink.start()
        self.capture.start()
        if self.auto_switch:
            self._stop.clear()
            self._watch_thread = threading.Thread(
                target=self._watch_foreground, name="fg-watch", daemon=True
            )
            self._watch_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        self.sink.stop()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=2.0)
            self._watch_thread = None

    def __enter__(self) -> "HapticEngine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- profile switching ----------------------------------------------------

    def apply_profile(self, profile: Profile, switch_x1: bool = True) -> None:
        """Switch the active profile.

        `switch_x1` controls whether the Control Panel's own profile is changed
        too. That call writes configuration to the mouse and can take a long
        time, so callers on a UI thread must pass False and perform the switch
        on a worker instead.
        """
        with self._lock:
            if profile.name == self.profile.name:
                return
            log.info("switching profile: %s -> %s", self.profile.name, profile.name)
            self.profile = profile
            self.analyzer.reconfigure(profile.bands)
            self.sink.min_gap_ms = profile.limits.min_gap_ms
            self.sink.max_pulses_sec = profile.limits.max_pulses_sec
            self.sink.max_duty = profile.limits.max_duty
            self.stats.profile_switches += 1

        if switch_x1:
            self.switch_x1_profile(profile)

    def switch_x1_profile(self, profile: Profile) -> None:
        """Drive the Control Panel's own profile. Blocking -- never call from a UI thread."""
        if not profile.x1_profile or self.dry_run:
            return
        try:
            self.sink.conn.profile_set(profile.x1_profile)
        except Exception as e:
            log.warning("could not set X1 profile %r: %s", profile.x1_profile, e)

    def _watch_foreground(self) -> None:
        while not self._stop.is_set():
            proc = foreground_process()
            if proc and proc != self._current_process:
                self._current_process = proc
                target = match_profile(self.profiles, proc)
                if target is None and self.fallback_profile is not None:
                    target = self.fallback_profile
                if target is not None:
                    self.apply_profile(target)
            self._stop.wait(1.0)

    def _band_rate_exceeded(self, band, now: float) -> bool:
        """Has this band already used up its own pulses-per-second allowance?"""
        if band.max_rate <= 0:
            return False
        wins = self._band_wins.get(band.name)
        if not wins:
            return False
        cutoff = now - 1.0
        wins[:] = [t for t in wins if t >= cutoff]
        return len(wins) >= band.max_rate

    # -- hot path -------------------------------------------------------------

    def _on_audio(self, mono: np.ndarray) -> None:
        onsets = self.analyzer.push(mono)
        if not onsets:
            return

        with self._lock:
            profile = self.profile
            bands = {b.name: b for b in profile.bands}

        self.stats.onsets += len(onsets)
        for onset in onsets:
            self.stats.band(onset.band).detected += 1

        # A band that has hit its own max_rate is excluded from the running. The
        # motor is one actuator, so without this a constantly-firing band (an
        # explosion-heavy low band, a musical bassline) wins every contested
        # frame and starves rarer events that carry more information. Excluding
        # it hands the frame to the next band rather than dropping the frame.
        now = time.perf_counter()
        eligible = []
        for onset in onsets:
            band = bands.get(onset.band)
            if band is None:
                continue
            if self._band_rate_exceeded(band, now):
                self.stats.band(onset.band).capped += 1
                continue
            eligible.append(onset)

        if not eligible:
            return

        # One actuator, so pick a single winner per frame.
        #
        # Rank by strength with priority as a thumb on the scale, NOT as an
        # absolute override. A sharp transient leaks energy into neighbouring
        # bands, so a gunshot also trips the low `impact` band -- weakly. If
        # priority sorted first, that weak leakage would outrank the strong
        # real detection and every gunshot would be shaped like an explosion.
        # Weighting keeps priority decisive only between comparable strengths.
        def score(o):
            band = bands.get(o.band)
            return o.strength + PRIORITY_WEIGHT * (band.priority if band else 0)

        ranked = sorted(eligible, key=score, reverse=True)
        winner = ranked[0]
        self.stats.suppressed_stacking += len(ranked) - 1

        self.stats.band(winner.band).won += 1
        for onset in ranked[1:]:
            self.stats.band(onset.band).lost += 1

        band = bands.get(winner.band)
        if band is None:
            return

        self._band_wins.setdefault(winner.band, []).append(now)

        span = band.strength_max - band.strength_min
        strength = band.strength_min + winner.strength * span
        strength = int(round(strength * profile.strength_scale))
        strength = max(0, min(100, strength))
        if strength <= 0:
            return

        pulse = Pulse(
            duration_ms=band.duration_ms,
            strength=strength,
            label=f"{winner.band}@{winner.strength:.2f}",
            priority=band.priority,
        )
        if self.sink.fire(pulse):
            self.stats.pulses += 1
            self.stats.band(winner.band).queued += 1

        self.stats.last_onsets.append((time.time(), winner.band, strength))
        if len(self.stats.last_onsets) > 64:
            del self.stats.last_onsets[:-64]

        if self.on_event:
            try:
                self.on_event(winner, pulse)
            except Exception:
                log.exception("on_event handler raised")
