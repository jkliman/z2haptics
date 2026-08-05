"""
Click-driven haptics: HUD tells us the weapon, the mouse tells us when it fires.

This replaces audio inference entirely for weapon feedback. Audio has to work
out *that* a shot happened and *which* weapon it was from a mix full of music,
explosions and other players' fire; measured on real captures that topped out
around a third of shots identified. A mouse click is the event itself, and the
HUD states the weapon outright.

Audio is still the right tool for what has no input event behind it --
explosions, incoming fire, vehicles -- so the two can run together, with the
audio profile's own bands covering everything the trigger does not.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from .api import HapticSink, X1Connection
from .firing import DEFAULT_SPECS, FireController, WeaponSpec, spec_from_dict
from .foreground import foreground_process
from .hud import HudSet, HudWatcher
from .inputs import InputWatcher

log = logging.getLogger(__name__)


@dataclass
class WeaponEngineStats:
    weapon: str | None = None
    hud_score: float = 0.0
    swaps: int = 0
    suppressed_inactive: int = 0
    blank_captures: int = 0
    started_at: float = field(default_factory=time.time)


class WeaponEngine:
    """Drives haptics from mouse input, with the weapon identified by the HUD."""

    def __init__(
        self,
        hud_set: HudSet | None = None,
        specs: dict[str, WeaponSpec] | None = None,
        processes: list[str] | None = None,
        fire_key: str = "mouse_left",
        reload_key: str = "r",
        strength_scale: float = 1.0,
        sink: HapticSink | None = None,
        dry_run: bool = False,
        on_event=None,
    ):
        self.hud_set = hud_set
        self.specs = dict(specs or DEFAULT_SPECS)
        self.processes = [p.lower() for p in (processes or [])]
        self.fire_key = fire_key
        self.reload_key = reload_key
        self.on_event = on_event

        self.stats = WeaponEngineStats()

        self.sink = sink or HapticSink(
            connection=X1Connection(),
            # The trigger already limits the rate, so these only guard the motor
            # against a very high RPM weapon rather than shaping the feel.
            min_gap_ms=25.0,
            max_pulses_sec=22.0,
            max_duty=0.55,
            dry_run=dry_run,
        )
        self._owns_sink = sink is None

        self.fire = FireController(
            sink=self.sink,
            weapon=None,
            is_active=self._game_is_active,
            strength_scale=strength_scale,
        )

        self.inputs = InputWatcher(
            keys=[fire_key, reload_key],
            on_press=self._on_press,
            on_release=self._on_release,
        )

        self.hud_watcher = (
            HudWatcher(hud_set, on_change=self._on_weapon_change) if hud_set else None
        )
        self._hud_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- gating ---------------------------------------------------------------

    def _game_is_active(self) -> bool:
        """Only pulse while the game has focus.

        Without this the mouse buzzes every time you click a link, because the
        trigger has no idea the game is not listening.
        """
        if not self.processes:
            return True
        active = foreground_process().lower()
        ok = any(active == p for p in self.processes)
        if not ok:
            self.stats.suppressed_inactive += 1
        return ok

    # -- input ----------------------------------------------------------------

    def _on_press(self, key: str) -> None:
        if key == self.fire_key:
            if self._game_is_active():
                self.fire.trigger_down()
        elif key == self.reload_key:
            if self._game_is_active():
                self.fire.reload()

    def _on_release(self, key: str) -> None:
        if key == self.fire_key:
            self.fire.trigger_up()

    # -- HUD ------------------------------------------------------------------

    def _on_weapon_change(self, name: str | None) -> None:
        spec = self.specs.get(name) if name else None
        if name and spec is None:
            # Seen on the HUD but never configured: fall back to a sane default
            # rather than going silent.
            spec = DEFAULT_SPECS.get(name) or WeaponSpec(name=name)
            self.specs[name] = spec

        self.fire.set_weapon(spec)
        self.stats.weapon = name
        self.stats.swaps += 1
        log.info("weapon changed to %r", name)
        if self.on_event:
            try:
                self.on_event("weapon", name)
            except Exception:
                log.exception("on_event handler raised")

    def _hud_loop(self) -> None:
        while not self._stop.is_set():
            if self.hud_watcher is not None:
                self.hud_watcher.poll()
                self.stats.hud_score = self.hud_watcher.last_score
                self.stats.blank_captures = self.hud_watcher.blank_reads
            self._stop.wait(self.hud_watcher.interval if self.hud_watcher else 0.5)

    # -- lifecycle ------------------------------------------------------------

    def set_weapon(self, name: str | None) -> None:
        """Force the weapon, for use without a HUD set."""
        self._on_weapon_change(name)

    def start(self) -> None:
        if self._owns_sink:
            self.sink.start()
        self.fire.start()
        self.inputs.start()
        if self.hud_watcher is not None:
            self._stop.clear()
            self._hud_thread = threading.Thread(target=self._hud_loop,
                                                name="hud-watch", daemon=True)
            self._hud_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.inputs.stop()
        self.fire.stop()
        if self._hud_thread is not None:
            self._hud_thread.join(timeout=2.0)
            self._hud_thread = None
        if self.hud_watcher is not None:
            self.hud_watcher.close()
        if self._owns_sink:
            self.sink.stop()

    def __enter__(self) -> "WeaponEngine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def status(self) -> dict:
        s = self.fire.stats()
        s.update({
            "hud_score": round(self.stats.hud_score, 2),
            "swaps": self.stats.swaps,
            "blank_captures": self.stats.blank_captures,
            "sink": self.sink.stats(),
        })
        return s


def specs_from_yaml(raw: dict) -> dict[str, WeaponSpec]:
    out = dict(DEFAULT_SPECS)
    for name, d in (raw or {}).items():
        merged = {"name": name, **(d or {})}
        out[name] = spec_from_dict(merged)
    return out
