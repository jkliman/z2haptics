"""
Qt-facing wrapper around HapticEngine.

The engine runs its own capture and delivery threads, so its callbacks arrive on
those threads -- never the GUI thread. Qt widgets may only be touched from the
GUI thread, so everything crosses over as a Qt signal, which delivers queued
across thread boundaries.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, QTimer, Signal

from ..analysis import Onset
from ..api import Pulse, X1Connection, X1Error
from ..engine import HapticEngine
from ..profiles import Profile, discover

log = logging.getLogger(__name__)


class EngineController(QObject):
    """Owns the engine lifecycle and republishes its activity as Qt signals."""

    statusChanged = Signal(str, bool)          # message, is_running
    statsUpdated = Signal(dict)                # counters for the status panel
    onsetDetected = Signal(str, int, float)    # band, strength, onset strength 0..1
    levelsUpdated = Signal(dict)               # band name -> (level, gate, is_open)
    profileChanged = Signal(str)
    errorRaised = Signal(str)
    testResult = Signal(str)                   # response text from a test pulse
    x1ProfilesLoaded = Signal(list)            # Control Panel profile names
    apiStatus = Signal(bool, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.profiles: dict[str, Profile] = {}
        self.engine: HapticEngine | None = None

        # Every device command is a blocking pipe round trip, and `Profile Set`
        # writes config to the mouse and can take a noticeable amount of time.
        # Running any of that on the GUI thread freezes the window, so it all
        # goes through one worker -- single-threaded so commands stay ordered.
        self._devices = ThreadPoolExecutor(max_workers=1, thread_name_prefix="x1-cmd")

        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._emit_state)

        self.reload_profiles()

    def shutdown(self) -> None:
        self.stop()
        self._devices.shutdown(wait=False, cancel_futures=True)

    def _submit(self, fn, *args) -> None:
        """Run a blocking device call on the worker, logging anything it raises."""
        def wrapped():
            try:
                fn(*args)
            except Exception:
                log.exception("device command failed")

        try:
            self._devices.submit(wrapped)
        except RuntimeError:
            pass  # executor already shut down

    # -- profiles -------------------------------------------------------------

    def reload_profiles(self) -> None:
        self.profiles = discover()
        if self.config.active_profile not in self.profiles and self.profiles:
            self.config.active_profile = next(iter(self.profiles))

    @property
    def active_profile(self) -> Profile | None:
        return self.profiles.get(self.config.active_profile)

    def set_profile(self, name: str) -> None:
        profile = self.profiles.get(name)
        if profile is None:
            return
        self.config.active_profile = name
        self.config.save()

        engine = self.engine
        if engine is not None:
            # The in-memory swap is cheap, so do it inline for an instant switch.
            # The Control Panel handoff writes to the mouse and must not block
            # the GUI thread, so it goes to the worker.
            engine.apply_profile(profile, switch_x1=False)
            self._submit(engine.switch_x1_profile, profile)

        self.profileChanged.emit(name)

    def apply_live_edits(self, profile: Profile) -> None:
        """Push in-place profile edits to a running engine.

        apply_profile() short-circuits when the name is unchanged, which is what
        we want for auto-switching but not for live tuning -- so reconfigure the
        analyzer and limits directly.
        """
        self.profiles[profile.name] = profile
        engine = self.engine
        if engine is None or engine.profile.name != profile.name:
            return
        engine.profile = profile
        engine.analyzer.reconfigure(profile.bands)
        engine.sink.min_gap_ms = profile.limits.min_gap_ms
        engine.sink.max_pulses_sec = profile.limits.max_pulses_sec
        engine.sink.max_duty = profile.limits.max_duty

    # -- lifecycle ------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.engine is not None

    def start(self) -> bool:
        if self.engine is not None:
            return True
        profile = self.active_profile
        if profile is None:
            self.errorRaised.emit("No profiles available.")
            return False

        try:
            engine = HapticEngine(
                profile=profile,
                profiles=self.profiles,
                device_name=self.config.device_name or None,
                samplerate=self.config.samplerate,
                auto_switch=self.config.auto_switch,
                fallback_profile=self.profiles.get("Default"),
                on_event=self._on_event,
            )
            engine.start()
        except Exception as e:
            log.exception("engine failed to start")
            self.errorRaised.emit(str(e))
            self.statusChanged.emit(f"Failed to start: {e}", False)
            return False

        self.engine = engine
        self._poll.start()
        device = engine.capture.resolved_name or "default output"
        self.statusChanged.emit(f"Running on {device}", True)
        self.profileChanged.emit(engine.profile.name)
        return True

    def stop(self) -> None:
        self._poll.stop()
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                log.exception("error stopping engine")
            self.engine = None
        self.statusChanged.emit("Stopped", False)

    def restart(self) -> bool:
        """Needed after changes the running capture stream cannot absorb."""
        was_running = self.running
        self.stop()
        return self.start() if was_running else True

    # -- engine callbacks (audio thread) --------------------------------------

    def _on_event(self, onset: Onset, pulse: Pulse) -> None:
        self.onsetDetected.emit(onset.band, pulse.strength, onset.strength)

    # -- polling (GUI thread) -------------------------------------------------

    def _emit_state(self) -> None:
        engine = self.engine
        if engine is None:
            return

        # Auto-switching happens inside the engine on its foreground-watch
        # thread, so the GUI only finds out by noticing the change here.
        if engine.profile.name != self.config.active_profile:
            self.config.active_profile = engine.profile.name
            self.config.save()
            self.profileChanged.emit(engine.profile.name)

        stats = engine.stats
        sink = engine.sink.stats()
        self.statsUpdated.emit({
            "profile": engine.profile.name,
            "onsets": stats.onsets,
            "pulses": stats.pulses,
            "sent": sink["sent"],
            "dropped_rate": sink["dropped_rate"],
            "dropped_queue": sink["dropped_queue"],
            "errors": sink["errors"],
            "switches": stats.profile_switches,
            "per_band": {
                name: {
                    "detected": b.detected, "won": b.won, "lost": b.lost,
                    "capped": b.capped, "queued": b.queued,
                }
                for name, b in stats.bands.items()
            },
        })

        levels = {}
        for band in engine.profile.bands:
            state = engine.analyzer.state.get(band.name)
            if state is not None:
                levels[band.name] = (state.level, band.gate, state.level >= band.gate)
        self.levelsUpdated.emit(levels)

    # -- one-off device commands ---------------------------------------------

    def test_pulse(self, duration_ms: int, strength: int) -> None:
        """Fire a pulse, bypassing the rate limiter.

        Used by the test panel, where the user explicitly asked for this exact
        pulse and throttling it would be confusing. Runs on the device worker;
        the response arrives on `testResult`.
        """
        self._submit(self._do_test_pulse, duration_ms, strength)

    def _do_test_pulse(self, duration_ms: int, strength: int) -> None:
        try:
            if self.engine is not None:
                resp = self.engine.sink.conn.vibrate(duration_ms, strength)
            else:
                conn = X1Connection()
                try:
                    resp = conn.vibrate(duration_ms, strength)
                finally:
                    conn.close()
        except X1Error as e:
            self.errorRaised.emit(str(e))
            resp = f"ERROR: {e}"
        self.testResult.emit(f"vibrate {duration_ms} {strength}  ->  {resp}")

    def check_api(self) -> None:
        """Probe the API on the worker; the answer arrives on `apiStatus`."""
        self._submit(self._do_check_api)

    def _do_check_api(self) -> None:
        try:
            conn = X1Connection()
            conn.connect()
            try:
                active = conn.profile_get()
                pipe = conn.active_pipe
                names = conn.profile_list()
            finally:
                conn.close()
        except X1Error as e:
            self.apiStatus.emit(False, str(e))
            self.x1ProfilesLoaded.emit([])
            return
        except Exception as e:
            self.apiStatus.emit(False, str(e))
            self.x1ProfilesLoaded.emit([])
            return

        self.apiStatus.emit(True, f"Connected ({pipe}); X1 profile: {active}")
        self.x1ProfilesLoaded.emit(names)
