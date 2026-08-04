"""
Per-weapon identification from audio, and per-weapon pulse shaping.

Battlefield exposes no client-side telemetry that could tell us which gun fired,
and reading the game's memory is not an option on a title with kernel-level
anti-cheat. So the weapon is identified from the sound itself.

Each weapon is represented by a spectral fingerprint -- a log-spaced, loudness-
normalised summary of its report -- learned from samples you capture in-game
with `z2haptics learn`. At runtime each gunfire onset is matched against every
template by cosine similarity, and the best match above a confidence floor
selects that weapon's pulse shape.

Honest limits, because they shape how this should be used:

  * It classifies what it *hears*, not what you fired. A teammate's LMG beside
    you can easily beat your own rifle.
  * Accuracy is best for your own weapon -- loudest, closest, most consistent.
  * Overlapping fire blurs the fingerprint. Confidence drops accordingly, which
    is why an unmatched onset falls back to the band's own settings rather than
    guessing.
  * Distance changes timbre as well as level. Templates captured close up will
    match distant fire less well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)

WEAPONS_DIR = Path.home() / ".z2haptics" / "weapons"


@dataclass
class WeaponTemplate:
    """One weapon's learned fingerprint and how it should feel."""

    name: str
    feature: np.ndarray            # unit-norm spectral fingerprint
    samples: int = 0               # how many captures it was averaged from
    spread: float = 0.0            # mean similarity of those captures to the mean

    # Pulse shaping. None means "inherit the band's value".
    duration_ms: int | None = None
    strength_min: int | None = None
    strength_max: int | None = None

    def similarity(self, feature: np.ndarray) -> float:
        """Cosine similarity. Both vectors are unit-norm, so this is a dot product."""
        if feature is None or feature.shape != self.feature.shape:
            return -1.0
        return float(np.dot(self.feature, feature))


@dataclass
class WeaponSet:
    """A collection of weapon templates, usually one per game."""

    name: str
    templates: list[WeaponTemplate] = field(default_factory=list)

    # Minimum similarity to accept a match, and how far ahead of the runner-up
    # the winner must be. Below either, the onset is treated as unidentified and
    # the band's own pulse shape is used.
    #
    # These defaults are chosen to bound the ERROR rate, not to maximise
    # accuracy, because a confidently wrong weapon feels worse than a generic
    # hit. Measured across clean, quiet, varied, distant and over-music
    # conditions (tools/experiment_weapons.py), 0.75/0.20 holds worst-case
    # misidentification near 5% while still naming about half of clean shots.
    # Loosening the margin to 0.04 raises clean accuracy to ~78% but lets
    # worst-case error reach 27%, which is the wrong trade here.
    min_confidence: float = 0.75
    min_margin: float = 0.20

    source: Path | None = None

    # Centroid of all templates -- the "generic gunshot" every weapon shares.
    _centroid: np.ndarray | None = field(default=None, repr=False)
    _centered: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Recompute the centroid and the centred templates.

        Raw fingerprints of different guns score 0.88-0.99 against each other:
        every gunshot shares the same broad structure, and that shared component
        dominates the similarity. Comparing raw vectors made the margin between
        first and second place vanish, so the classifier declined ~67% of even
        clean shots.

        Subtracting the centroid removes what all weapons have in common and
        leaves only what distinguishes them, which is the part we actually want
        to match on.
        """
        if not self.templates:
            self._centroid, self._centered = None, {}
            return

        centroid = np.mean([t.feature for t in self.templates], axis=0)
        self._centroid = centroid.astype(np.float32)

        self._centered = {}
        for t in self.templates:
            self._centered[t.name] = _unit(t.feature - self._centroid)

    def classify(self, feature: np.ndarray | None) -> tuple[WeaponTemplate | None, float]:
        """Best-matching template and its confidence, or (None, score) if unsure."""
        if feature is None or not self.templates or self._centroid is None:
            return None, 0.0
        if feature.shape != self._centroid.shape:
            return None, 0.0

        probe = _unit(feature - self._centroid)
        scored = sorted(
            ((t, float(np.dot(self._centered[t.name], probe))) for t in self.templates),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best, best_score = scored[0]

        if best_score < self.min_confidence:
            return None, best_score
        if len(scored) > 1 and (best_score - scored[1][1]) < self.min_margin:
            return None, best_score
        return best, best_score

    def confusability(self) -> list[tuple[str, str, float]]:
        """Pairwise similarity between centred templates, most confusable first."""
        pairs = []
        names = [t.name for t in self.templates]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs.append((a, b, float(np.dot(self._centered[a], self._centered[b]))))
        pairs.sort(key=lambda p: p[2], reverse=True)
        return pairs


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return (v / norm).astype(np.float32) if norm > 1e-9 else v.astype(np.float32)


# -- persistence --------------------------------------------------------------

def to_dict(ws: WeaponSet) -> dict:
    return {
        "name": ws.name,
        "min_confidence": round(ws.min_confidence, 3),
        "min_margin": round(ws.min_margin, 3),
        "weapons": [
            {
                "name": t.name,
                "samples": t.samples,
                "spread": round(t.spread, 4),
                **({"duration_ms": t.duration_ms} if t.duration_ms is not None else {}),
                **({"strength_min": t.strength_min} if t.strength_min is not None else {}),
                **({"strength_max": t.strength_max} if t.strength_max is not None else {}),
                "feature": [round(float(v), 5) for v in t.feature],
            }
            for t in ws.templates
        ],
    }


def save(ws: WeaponSet, path: Path | None = None) -> Path:
    if path is None:
        WEAPONS_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in ws.name)
        path = WEAPONS_DIR / f"{safe.strip().replace(' ', '_').lower()}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(ws), f, sort_keys=False)
    ws.source = path
    return path


def load(path: Path) -> WeaponSet:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    templates = []
    for w in raw.get("weapons", []) or []:
        feature = np.asarray(w.get("feature", []), dtype=np.float32)
        if feature.size == 0:
            log.warning("weapon %r has no feature vector; skipping", w.get("name"))
            continue
        norm = float(np.linalg.norm(feature))
        if norm > 1e-9:
            feature = feature / norm
        templates.append(WeaponTemplate(
            name=w.get("name", "?"),
            feature=feature,
            samples=int(w.get("samples", 0)),
            spread=float(w.get("spread", 0.0)),
            duration_ms=w.get("duration_ms"),
            strength_min=w.get("strength_min"),
            strength_max=w.get("strength_max"),
        ))

    return WeaponSet(
        name=raw.get("name") or path.stem,
        templates=templates,
        min_confidence=float(raw.get("min_confidence", 0.55)),
        min_margin=float(raw.get("min_margin", 0.04)),
        source=path,
    )


def discover(extra_dirs: list[Path] | None = None) -> dict[str, WeaponSet]:
    sets: dict[str, WeaponSet] = {}
    for d in [WEAPONS_DIR, *(extra_dirs or [])]:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            try:
                ws = load(path)
            except Exception as e:
                log.error("skipping weapon set %s: %s", path, e)
                continue
            sets[ws.name] = ws
    return sets


def resolve(reference: str, extra_dirs: list[Path] | None = None) -> WeaponSet | None:
    """Look up a weapon set by name, or load it from an explicit path."""
    path = Path(reference)
    if path.is_file():
        return load(path)
    return discover(extra_dirs).get(reference)


# -- building templates from captured samples ---------------------------------

def build_template(name: str, features: list[np.ndarray]) -> WeaponTemplate:
    """Average captured fingerprints into one template.

    `spread` records how tightly the captures agree with their own mean. A low
    value means the samples disagree, which usually means the captures were
    contaminated by other sounds -- worth knowing before trusting the template.
    """
    stacked = np.vstack(features)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 1e-9:
        mean = mean / norm

    sims = [float(np.dot(mean, f)) for f in features]
    return WeaponTemplate(
        name=name,
        feature=mean.astype(np.float32),
        samples=len(features),
        spread=float(np.mean(sims)) if sims else 0.0,
    )


def separability(ws: WeaponSet) -> list[tuple[str, str, float]]:
    """Pairwise similarity between templates, most confusable first.

    Measured on centred templates, so it reflects what the classifier actually
    compares. Two weapons scoring near 1.0 cannot be told apart, and reporting
    that is more useful than letting the classifier flap between them at runtime.
    """
    return ws.confusability()
