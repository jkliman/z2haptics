"""
Profile loading: per-game band layouts and motor tuning.

A profile is a YAML file describing which frequency bands matter for a game, how
sensitive each should be, and how the resulting pulses are shaped. Profiles also
declare which processes they apply to, so the engine can follow the foreground
window and switch itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .analysis import Band

log = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "profiles"
USER_DIR = Path.home() / ".z2haptics" / "profiles"


@dataclass
class MotorLimits:
    """Motor-protective limits, overridable per profile."""

    min_gap_ms: float = 45.0
    max_pulses_sec: float = 14.0
    max_duty: float = 0.55


@dataclass
class Profile:
    name: str
    description: str = ""
    processes: list[str] = field(default_factory=list)
    strength_scale: float = 1.0
    limits: MotorLimits = field(default_factory=MotorLimits)
    bands: list[Band] = field(default_factory=list)
    source: Path | None = None

    # Optionally drive the Control Panel's own profile switching alongside ours.
    x1_profile: str | None = None

    def matches(self, process_name: str) -> bool:
        p = process_name.lower()
        return any(p == proc.lower() for proc in self.processes)


def _band_from_dict(d: dict) -> Band:
    known = {
        "name", "low_hz", "high_hz", "sensitivity", "gate", "refractory_ms",
        "min_share", "duration_ms", "strength_min", "strength_max",
        "level_floor_db", "level_ceil_db", "priority", "enabled",
    }
    unknown = set(d) - known
    if unknown:
        log.warning("band %r: ignoring unknown keys %s", d.get("name"), sorted(unknown))
    return Band(**{k: v for k, v in d.items() if k in known})


def load_profile(path: Path) -> Profile:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    limits_raw = raw.get("limits", {}) or {}
    bands_raw = raw.get("bands", []) or []

    profile = Profile(
        name=raw.get("name") or path.stem,
        description=raw.get("description", ""),
        processes=list(raw.get("processes", []) or []),
        strength_scale=float(raw.get("strength_scale", 1.0)),
        limits=MotorLimits(
            min_gap_ms=float(limits_raw.get("min_gap_ms", 45.0)),
            max_pulses_sec=float(limits_raw.get("max_pulses_sec", 14.0)),
            max_duty=float(limits_raw.get("max_duty", 0.55)),
        ),
        bands=[_band_from_dict(b) for b in bands_raw],
        x1_profile=raw.get("x1_profile"),
        source=path,
    )

    if not profile.bands:
        raise ValueError(f"{path}: profile defines no bands")
    return profile


def profile_to_dict(p: Profile) -> dict:
    """Serialise a Profile back to the YAML document shape."""
    return {
        "name": p.name,
        "description": p.description,
        "processes": list(p.processes),
        **({"x1_profile": p.x1_profile} if p.x1_profile else {}),
        "strength_scale": round(p.strength_scale, 3),
        "limits": {
            "min_gap_ms": round(p.limits.min_gap_ms, 1),
            "max_pulses_sec": round(p.limits.max_pulses_sec, 1),
            "max_duty": round(p.limits.max_duty, 3),
        },
        "bands": [
            {
                "name": b.name,
                "low_hz": round(b.low_hz, 1),
                "high_hz": round(b.high_hz, 1),
                "sensitivity": round(b.sensitivity, 3),
                "gate": round(b.gate, 6),
                "refractory_ms": round(b.refractory_ms, 1),
                "min_share": round(b.min_share, 3),
                "duration_ms": int(b.duration_ms),
                "strength_min": int(b.strength_min),
                "strength_max": int(b.strength_max),
                "level_floor_db": round(b.level_floor_db, 1),
                "level_ceil_db": round(b.level_ceil_db, 1),
                "priority": int(b.priority),
                "enabled": bool(b.enabled),
            }
            for b in p.bands
        ],
    }


def save_profile(p: Profile, path: Path | None = None) -> Path:
    """Write a profile to YAML.

    Defaults to the user profile directory rather than overwriting a shipped
    profile in the repo, so edits made in the GUI survive an update and the
    originals stay intact. `discover()` gives user profiles precedence.
    """
    if path is None:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in p.name)
        path = USER_DIR / f"{safe.strip().replace(' ', '_').lower()}.yaml"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile_to_dict(p), f, sort_keys=False, allow_unicode=True)
    p.source = path
    return path


def discover(extra_dirs: list[Path] | None = None) -> dict[str, Profile]:
    """Load every profile from the builtin and user directories.

    User profiles shadow builtins of the same name, so a shipped profile can be
    customised without editing the repo copy.
    """
    profiles: dict[str, Profile] = {}
    dirs = [BUILTIN_DIR, USER_DIR, *(extra_dirs or [])]

    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            try:
                p = load_profile(path)
            except Exception as e:
                log.error("skipping %s: %s", path, e)
                continue
            if p.name in profiles:
                log.info("profile %r from %s overrides %s",
                         p.name, path, profiles[p.name].source)
            profiles[p.name] = p

    return profiles


def match_profile(profiles: dict[str, Profile], process_name: str) -> Profile | None:
    """Find the profile claiming `process_name`, if any."""
    for p in profiles.values():
        if p.matches(process_name):
            return p
    return None
