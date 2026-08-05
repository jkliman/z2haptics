"""
Click-driven weapon simulation.

Once the fire event comes from the mouse rather than from audio, the weapon can
be modelled instead of merely reacted to: automatic fire pulses at the weapon's
real rate while the button is held, stops when the magazine would be empty, and
stays quiet through a reload. That is a far more convincing feel than anything
triggered by sound, and it is only possible because a click is ground truth.

The honest cost of this approach: a click fires whether or not the game does.
Menus, an empty magazine, a weapon that has not finished its reload -- all
produce a phantom pulse unless modelled. Hence foreground gating, ammo counting
and reload suppression. Those exist to close the gap that audio never had,
because audio only ever reacted to sounds the game actually made.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from .api import Pulse

log = logging.getLogger(__name__)


@dataclass
class WeaponSpec:
    """How one weapon fires and how it should feel."""

    name: str

    # Fire behaviour
    rpm: float = 600.0            # rounds per minute for automatic fire
    magazine: int = 30            # rounds before a reload is needed
    reload_s: float = 2.5         # time from reload keypress to a full magazine
    auto: bool = True             # held trigger keeps firing
    burst: int = 0                # >0: fire this many rounds per click

    # Pulse shaping
    duration_ms: int = 40
    strength: int = 70

    # First round of a burst can hit harder, which reads as recoil onset.
    first_shot_bonus: int = 0

    @property
    def shot_interval(self) -> float:
        return 60.0 / max(self.rpm, 1.0)


@dataclass
class FireState:
    ammo: int = 0
    reloading_until: float = 0.0
    firing: bool = False
    next_shot_at: float = 0.0
    rounds_this_pull: int = 0
    shots_fired: int = 0
    dry_clicks: int = 0
    reloads: int = 0


class FireController:
    """Turns mouse input into shaped pulses for the current weapon.

    Runs its own timer thread so automatic fire keeps pulsing while the button
    is held, rather than only on the press edge.
    """

    def __init__(
        self,
        sink,
        weapon: WeaponSpec | None = None,
        is_active=None,
        strength_scale: float = 1.0,
    ):
        self.sink = sink
        self.weapon = weapon
        # Callable returning False when pulses should be suppressed -- normally
        # "is the game the foreground window", so clicking in a browser or a
        # menu does not buzz the mouse.
        self.is_active = is_active or (lambda: True)
        self.strength_scale = strength_scale

        self.state = FireState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        if weapon is not None:
            self.state.ammo = weapon.magazine

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fire-ctl", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "FireController":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- weapon ---------------------------------------------------------------

    def set_weapon(self, weapon: WeaponSpec | None, refill: bool = True) -> None:
        with self._lock:
            changed = weapon is not self.weapon
            self.weapon = weapon
            if weapon is not None and (refill or self.state.ammo > weapon.magazine):
                self.state.ammo = weapon.magazine
            if changed:
                self.state.firing = False
                self.state.rounds_this_pull = 0
                self.state.reloading_until = 0.0

    # -- input edges ----------------------------------------------------------

    def trigger_down(self) -> None:
        with self._lock:
            if self.weapon is None:
                return
            self.state.firing = True
            self.state.rounds_this_pull = 0
            self.state.next_shot_at = 0.0     # fire the first round immediately

    def trigger_up(self) -> None:
        with self._lock:
            self.state.firing = False
            self.state.rounds_this_pull = 0

    def reload(self) -> None:
        with self._lock:
            if self.weapon is None:
                return
            if self.state.ammo >= self.weapon.magazine:
                return                        # already full; game would ignore it too
            self.state.reloading_until = time.monotonic() + self.weapon.reload_s
            self.state.firing = False
            self.state.reloads += 1

    def refill(self) -> None:
        with self._lock:
            if self.weapon is not None:
                self.state.ammo = self.weapon.magazine
                self.state.reloading_until = 0.0

    def set_ammo(self, rounds: int) -> None:
        """Sync magazine state from an external source, e.g. the HUD counter."""
        with self._lock:
            if self.weapon is not None:
                self.state.ammo = max(0, min(rounds, self.weapon.magazine))

    # -- firing loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self._tick()
            # Fine enough to keep automatic fire even at high RPM: a 1000rpm
            # weapon needs a round every 60ms.
            time.sleep(0.004)

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            weapon = self.weapon
            if weapon is None or not self.state.firing:
                return
            if now < self.state.reloading_until:
                return
            if now < self.state.next_shot_at:
                return

            if self.state.ammo <= 0:
                # Out of ammo: one dry click's worth of nothing, then stop, so a
                # held trigger does not keep buzzing on an empty magazine.
                self.state.firing = False
                self.state.dry_clicks += 1
                return

            if weapon.burst and self.state.rounds_this_pull >= weapon.burst:
                self.state.firing = False
                return
            if not weapon.auto and self.state.rounds_this_pull >= 1:
                self.state.firing = False
                return

            first = self.state.rounds_this_pull == 0
            self.state.ammo -= 1
            self.state.rounds_this_pull += 1
            self.state.shots_fired += 1
            self.state.next_shot_at = now + weapon.shot_interval

            strength = weapon.strength + (weapon.first_shot_bonus if first else 0)
            strength = int(round(strength * self.strength_scale))
            strength = max(0, min(100, strength))
            duration = weapon.duration_ms
            label = f"{weapon.name}{'!' if first else ''}"

        if not self.is_active():
            return
        if strength > 0:
            self.sink.fire(Pulse(duration_ms=duration, strength=strength,
                                 label=label, priority=5))

    # -- introspection --------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            reloading = max(0.0, self.state.reloading_until - time.monotonic())
            return {
                "weapon": self.weapon.name if self.weapon else None,
                "ammo": self.state.ammo,
                "magazine": self.weapon.magazine if self.weapon else 0,
                "firing": self.state.firing,
                "reloading_s": round(reloading, 2),
                "shots_fired": self.state.shots_fired,
                "dry_clicks": self.state.dry_clicks,
                "reloads": self.state.reloads,
            }


# -- persistence --------------------------------------------------------------

def spec_to_dict(w: WeaponSpec) -> dict:
    return {
        "name": w.name,
        "rpm": round(w.rpm, 1),
        "magazine": int(w.magazine),
        "reload_s": round(w.reload_s, 2),
        "auto": bool(w.auto),
        "burst": int(w.burst),
        "duration_ms": int(w.duration_ms),
        "strength": int(w.strength),
        "first_shot_bonus": int(w.first_shot_bonus),
    }


def spec_from_dict(d: dict) -> WeaponSpec:
    known = {f for f in WeaponSpec.__dataclass_fields__}
    return WeaponSpec(**{k: v for k, v in d.items() if k in known})


# Starting points for the common Battlefield weapon classes. These are
# deliberately round numbers -- the exact RPM of a given gun matters far less
# than the class feeling distinct, and they are meant to be edited.
DEFAULT_SPECS = {
    "assault": WeaponSpec(name="assault", rpm=700, magazine=30, reload_s=2.4,
                          duration_ms=38, strength=68, first_shot_bonus=12),
    "smg": WeaponSpec(name="smg", rpm=850, magazine=32, reload_s=2.2,
                      duration_ms=30, strength=58, first_shot_bonus=10),
    "lmg": WeaponSpec(name="lmg", rpm=750, magazine=100, reload_s=6.0,
                      duration_ms=28, strength=62, first_shot_bonus=14),
    "sniper": WeaponSpec(name="sniper", rpm=50, magazine=5, reload_s=3.5,
                         auto=False, duration_ms=130, strength=100),
    "shotgun": WeaponSpec(name="shotgun", rpm=90, magazine=8, reload_s=4.0,
                          auto=False, duration_ms=95, strength=92),
    "pistol": WeaponSpec(name="pistol", rpm=380, magazine=15, reload_s=1.8,
                         auto=False, duration_ms=45, strength=60),
    "rifle": WeaponSpec(name="rifle", rpm=750, magazine=30, reload_s=2.3,
                        duration_ms=36, strength=65, first_shot_bonus=12),
}
