r"""
Identify the held weapon by looking at the game's HUD.

Rather than OCR the weapon name, this matches a captured screen region against
reference images taken once per weapon. Template matching sidesteps the fragile
parts of OCR entirely -- stylised game fonts, outlines, drop shadows, partial
transparency over changing backgrounds -- and it needs no OCR engine. The cost
is that a template is tied to the resolution and HUD scale it was captured at,
so changing either means recapturing.

Matching is done on a normalised, edge-emphasised grayscale image so that the
scene showing through a translucent HUD does not dominate the comparison.

Known limitation: screen capture cannot see an exclusive-fullscreen game. If
captures come back black, switch the game to borderless windowed. `capture()`
reports that case rather than silently matching nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HUD_DIR = Path.home() / ".z2haptics" / "hud"


@dataclass
class Region:
    """A screen rectangle, in physical pixels."""

    left: int
    top: int
    width: int
    height: int

    def as_dict(self) -> dict:
        return {"left": self.left, "top": self.top,
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        return cls(int(d["left"]), int(d["top"]), int(d["width"]), int(d["height"]))


class ScreenCapture:
    """Grabs a screen region. Thread-confined: mss is not thread-safe."""

    def __init__(self):
        self._sct = None

    def _ensure(self):
        if self._sct is None:
            import mss
            self._sct = mss.mss()
        return self._sct

    def grab(self, region: Region) -> np.ndarray:
        """Return the region as an (h, w, 3) uint8 RGB array."""
        sct = self._ensure()
        raw = sct.grab(region.as_dict())
        # mss returns BGRA.
        arr = np.frombuffer(raw.rgb, dtype=np.uint8)
        return arr.reshape(raw.height, raw.width, 3)

    def screen_size(self) -> tuple[int, int]:
        sct = self._ensure()
        mon = sct.monitors[1]
        return mon["width"], mon["height"]

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None


def is_blank(image: np.ndarray, threshold: float = 2.0) -> bool:
    """True when a capture is featureless -- almost always exclusive fullscreen."""
    return float(image.std()) < threshold


def fingerprint(image: np.ndarray, size: tuple[int, int] = (64, 16)) -> np.ndarray:
    """Reduce a HUD crop to a comparable vector.

    Grayscale, downsampled, then gradient-emphasised and normalised. The
    gradient step matters: HUD text sits over whatever the player is looking at,
    so absolute brightness swings wildly while the *shape* of the glyphs does
    not. Comparing raw pixels would mostly compare the scenery behind them.
    """
    if image.ndim == 3:
        gray = image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        gray = image.astype(np.float32)

    h, w = gray.shape
    out_w, out_h = size
    # Box-downsample by averaging over integer blocks.
    ys = np.linspace(0, h, out_h + 1).astype(int)
    xs = np.linspace(0, w, out_w + 1).astype(int)
    small = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            block = gray[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)]
            small[i, j] = float(block.mean()) if block.size else 0.0

    gx = np.diff(small, axis=1, prepend=small[:, :1])
    gy = np.diff(small, axis=0, prepend=small[:1, :])
    feat = np.abs(gx) + np.abs(gy)

    feat = feat.ravel()
    feat -= feat.mean()
    norm = float(np.linalg.norm(feat))
    return (feat / norm).astype(np.float32) if norm > 1e-6 else feat.astype(np.float32)


@dataclass
class HudTemplate:
    name: str
    feature: np.ndarray
    samples: int = 1

    def similarity(self, feature: np.ndarray) -> float:
        if feature is None or feature.shape != self.feature.shape:
            return -1.0
        return float(np.dot(self.feature, feature))


@dataclass
class HudSet:
    """Region plus one visual template per weapon."""

    name: str
    region: Region | None = None
    templates: list[HudTemplate] = field(default_factory=list)
    min_confidence: float = 0.60
    min_margin: float = 0.05
    source: Path | None = None

    def identify(self, image: np.ndarray) -> tuple[HudTemplate | None, float]:
        if not self.templates or image is None or image.size == 0:
            return None, 0.0
        if is_blank(image):
            return None, 0.0

        probe = fingerprint(image)
        scored = sorted(((t, t.similarity(probe)) for t in self.templates),
                        key=lambda p: p[1], reverse=True)
        best, score = scored[0]
        if score < self.min_confidence:
            return None, score
        if len(scored) > 1 and (score - scored[1][1]) < self.min_margin:
            return None, score
        return best, score

    def confusability(self) -> list[tuple[str, str, float]]:
        pairs = []
        for i, a in enumerate(self.templates):
            for b in self.templates[i + 1:]:
                pairs.append((a.name, b.name, a.similarity(b.feature)))
        pairs.sort(key=lambda p: p[2], reverse=True)
        return pairs


# -- persistence --------------------------------------------------------------

def save(hs: HudSet, path: Path | None = None) -> Path:
    if path is None:
        HUD_DIR.mkdir(parents=True, exist_ok=True)
        path = HUD_DIR / f"{hs.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": hs.name,
        "region": hs.region.as_dict() if hs.region else None,
        "min_confidence": hs.min_confidence,
        "min_margin": hs.min_margin,
        "templates": [
            {"name": t.name, "samples": t.samples,
             "feature": [round(float(v), 5) for v in t.feature]}
            for t in hs.templates
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    hs.source = path
    return path


def load(path: Path) -> HudSet:
    raw = json.loads(path.read_text(encoding="utf-8"))
    templates = []
    for t in raw.get("templates", []):
        feat = np.asarray(t.get("feature", []), dtype=np.float32)
        if feat.size:
            templates.append(HudTemplate(name=t["name"], feature=feat,
                                         samples=int(t.get("samples", 1))))
    region = Region.from_dict(raw["region"]) if raw.get("region") else None
    return HudSet(
        name=raw.get("name", path.stem),
        region=region,
        templates=templates,
        min_confidence=float(raw.get("min_confidence", 0.60)),
        min_margin=float(raw.get("min_margin", 0.05)),
        source=path,
    )


def discover() -> dict[str, HudSet]:
    sets = {}
    if HUD_DIR.is_dir():
        for path in sorted(HUD_DIR.glob("*.json")):
            try:
                hs = load(path)
            except Exception as e:
                log.error("skipping HUD set %s: %s", path, e)
                continue
            sets[hs.name] = hs
    return sets


def resolve(reference: str) -> HudSet | None:
    path = Path(reference)
    if path.is_file():
        return load(path)
    return discover().get(reference)


def build_template(name: str, images: list[np.ndarray]) -> HudTemplate:
    feats = [fingerprint(im) for im in images]
    mean = np.mean(feats, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 1e-6:
        mean = mean / norm
    return HudTemplate(name=name, feature=mean.astype(np.float32), samples=len(feats))


class HudWatcher:
    """Polls the HUD region and reports which weapon is showing.

    Deliberately slow: the weapon only changes when the player swaps, so there
    is nothing to gain from sampling fast, and screen capture is expensive
    enough to matter inside a game. A change must also persist for a couple of
    reads before it is accepted, since swap animations pass through frames that
    match nothing or match the wrong thing.
    """

    def __init__(self, hud_set: HudSet, on_change=None, interval: float = 0.4,
                 confirmations: int = 2):
        self.hud_set = hud_set
        self.on_change = on_change
        self.interval = interval
        self.confirmations = confirmations

        self.current: str | None = None
        self.last_score = 0.0
        self.blank_reads = 0

        self._candidate: str | None = None
        self._candidate_count = 0
        self._capture = ScreenCapture()

    def poll(self) -> str | None:
        if self.hud_set.region is None:
            return self.current
        try:
            image = self._capture.grab(self.hud_set.region)
        except Exception as e:
            log.warning("HUD capture failed: %s", e)
            return self.current

        if is_blank(image):
            self.blank_reads += 1
            return self.current
        self.blank_reads = 0

        match, score = self.hud_set.identify(image)
        self.last_score = score
        name = match.name if match else None

        if name == self.current:
            self._candidate, self._candidate_count = None, 0
            return self.current

        if name == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = name, 1

        if self._candidate_count >= self.confirmations:
            self.current = self._candidate
            self._candidate, self._candidate_count = None, 0
            if self.on_change:
                try:
                    self.on_change(self.current)
                except Exception:
                    log.exception("HUD on_change handler raised")
        return self.current

    def close(self) -> None:
        self._capture.close()
