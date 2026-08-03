r"""
Event capture and spectral profiling.

Guessing band edges from intuition only gets you so far -- a plasma rifle in one
game and a suppressed carbine in another sit in completely different places. This
module lets you play the game, tap a hotkey each time you hear the event you care
about, and derive the bands from what the game actually sounds like.

Workflow:

    z2haptics learn --labels gunshot,laser,explosion --name avatar
        Play. Tap F9/F10/F11 when you hear each event. F8 to finish.
        Segments are written to a session directory as WAV plus metadata.

    z2haptics analyze avatar
        Averages each label's spectrum, contrasts it against the ambient
        background, and emits both a readable report and a suggested profile.

Because you tap the key *after* hearing the event, capture is retrospective: a
rolling buffer keeps the last few seconds, and a mark extracts the window that
preceded the press.
"""

from __future__ import annotations

import ctypes
import json
import logging
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".z2haptics" / "sessions"

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short

# Virtual-key codes. F-keys are used by default because they are polled globally
# and rarely bound in game, but any VK code works.
VK = {f"F{i}": 0x6F + i for i in range(1, 13)}          # F1..F12 -> 0x70..0x7B
VK.update({f"NUM{i}": 0x60 + i for i in range(10)})     # NUMPAD0..9
DEFAULT_MARK_KEYS = ["F9", "F10", "F11", "F12", "NUM1", "NUM2", "NUM3", "NUM4"]
DEFAULT_STOP_KEY = "F8"
DEFAULT_AMBIENT_KEY = "F7"


# -- capture ------------------------------------------------------------------

class RingBuffer:
    """Fixed-length rolling buffer of mono samples."""

    def __init__(self, samplerate: int, seconds: float):
        self.samplerate = samplerate
        self.size = int(samplerate * seconds)
        self._buf = np.zeros(self.size, dtype=np.float32)
        self._written = 0
        self._lock = threading.Lock()

    @property
    def total_written(self) -> int:
        return self._written

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        with self._lock:
            if n >= self.size:
                self._buf[:] = data[-self.size:]
            else:
                self._buf = np.roll(self._buf, -n)
                self._buf[-n:] = data
            self._written += n

    def extract(self, end_offset: int, length: int) -> np.ndarray:
        """Extract `length` samples ending `end_offset` samples before the write head."""
        with self._lock:
            end = self.size - end_offset
            start = max(0, end - length)
            return self._buf[start:max(end, start)].copy()


class HotkeyListener:
    """Polls global key state so marks register while a game has focus.

    RegisterHotKey would need a message pump on a dedicated thread and steals the
    key from the game. Polling GetAsyncKeyState is passive -- the game still sees
    the keypress -- which matters when marking events mid-firefight.
    """

    def __init__(self, keys: dict[str, str], on_press, poll_hz: float = 60.0):
        self.keys = keys                    # key name -> action label
        self.on_press = on_press
        self.interval = 1.0 / poll_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        down = {name: False for name in self.keys}
        # Drain any state left over from before we started listening.
        for name in self.keys:
            user32.GetAsyncKeyState(VK[name])

        while not self._stop.is_set():
            for name, action in self.keys.items():
                pressed = bool(user32.GetAsyncKeyState(VK[name]) & 0x8000)
                if pressed and not down[name]:
                    down[name] = True
                    try:
                        self.on_press(action)
                    except Exception:
                        log.exception("hotkey handler raised")
                elif not pressed:
                    down[name] = False
            time.sleep(self.interval)


@dataclass
class Sample:
    label: str
    index: int
    filename: str
    timestamp: float
    peak: float
    rms: float


@dataclass
class SessionMeta:
    name: str
    samplerate: int
    pre_roll_s: float
    post_roll_s: float
    labels: list[str]
    key_map: dict[str, str]
    samples: list[Sample] = field(default_factory=list)
    created: float = 0.0
    device: str = ""


class LearnSession:
    """Captures labelled audio segments to a session directory."""

    def __init__(
        self,
        name: str,
        labels: list[str],
        samplerate: int = 48000,
        pre_roll_s: float = 0.65,
        post_roll_s: float = 0.20,
        buffer_s: float = 4.0,
        root: Path | None = None,
    ):
        self.name = name
        self.labels = labels
        self.samplerate = samplerate
        self.pre_roll_s = pre_roll_s
        self.post_roll_s = post_roll_s

        self.dir = (root or SESSIONS_DIR) / name
        self.dir.mkdir(parents=True, exist_ok=True)

        self.ring = RingBuffer(samplerate, buffer_s)
        self.counts: dict[str, int] = {lbl: 0 for lbl in [*labels, "ambient"]}
        self.samples: list[Sample] = []
        self._pending: list[tuple[int, str]] = []   # (write position, label)
        self._lock = threading.Lock()
        self.device = ""

    # -- audio path -----------------------------------------------------------

    def on_audio(self, mono: np.ndarray) -> None:
        self.ring.write(mono)
        self._drain_pending()

    def mark(self, label: str) -> None:
        """Record that an event happened now; extraction waits for the post-roll."""
        with self._lock:
            self._pending.append((self.ring.total_written, label))

    def _drain_pending(self) -> None:
        post = int(self.post_roll_s * self.samplerate)
        length = int((self.pre_roll_s + self.post_roll_s) * self.samplerate)

        with self._lock:
            ready = [(p, l) for (p, l) in self._pending
                     if self.ring.total_written - p >= post]
            self._pending = [(p, l) for (p, l) in self._pending
                             if self.ring.total_written - p < post]

        for mark_pos, label in ready:
            end_offset = self.ring.total_written - (mark_pos + post)
            seg = self.ring.extract(end_offset=max(0, end_offset), length=length)
            if seg.size == 0:
                continue
            self._save(label, seg)

    def _save(self, label: str, seg: np.ndarray) -> None:
        self.counts[label] = self.counts.get(label, 0) + 1
        idx = self.counts[label]
        fname = f"{label}_{idx:03d}.wav"

        pcm = np.clip(seg, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        with wave.open(str(self.dir / fname), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.samplerate)
            w.writeframes(pcm16.tobytes())

        self.samples.append(
            Sample(
                label=label,
                index=idx,
                filename=fname,
                timestamp=time.time(),
                peak=float(np.max(np.abs(seg))),
                rms=float(np.sqrt(np.mean(seg ** 2))),
            )
        )

    # -- persistence ----------------------------------------------------------

    def flush(self) -> None:
        """Extract anything still pending, then write session metadata."""
        time.sleep(self.post_roll_s + 0.05)
        with self._lock:
            pending = list(self._pending)
            self._pending = []
        length = int((self.pre_roll_s + self.post_roll_s) * self.samplerate)
        for _, label in pending:
            seg = self.ring.extract(end_offset=0, length=length)
            if seg.size:
                self._save(label, seg)

        meta = SessionMeta(
            name=self.name,
            samplerate=self.samplerate,
            pre_roll_s=self.pre_roll_s,
            post_roll_s=self.post_roll_s,
            labels=self.labels,
            key_map={},
            samples=self.samples,
            created=time.time(),
            device=self.device,
        )
        with open(self.dir / "session.json", "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)


# -- analysis -----------------------------------------------------------------

@dataclass
class LabelSpectrum:
    label: str
    count: int
    freqs: np.ndarray
    mean_db: np.ndarray       # average spectrum across this label's samples
    peak_db: float
    peak_hz: float

    def contrast_against(self, background_db: np.ndarray) -> np.ndarray:
        return self.mean_db - background_db


def _spectrum_db(seg: np.ndarray, samplerate: int, nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Average magnitude spectrum in dB over the loudest part of a segment.

    Only the region around the peak is used -- the pre-roll deliberately contains
    ambience, and averaging that in would wash out the event's signature.
    """
    if seg.size < nfft:
        seg = np.pad(seg, (0, nfft - seg.size))

    hop = nfft // 4
    frames = []
    energies = []
    for start in range(0, len(seg) - nfft + 1, hop):
        frame = seg[start:start + nfft]
        energies.append(float(np.sqrt(np.mean(frame ** 2))))
        frames.append(frame)

    if not frames:
        frames = [seg[:nfft]]
        energies = [1.0]

    peak_i = int(np.argmax(energies))
    lo = max(0, peak_i - 1)
    hi = min(len(frames), peak_i + 3)

    window = np.hanning(nfft)
    acc = None
    for frame in frames[lo:hi]:
        mag = np.abs(np.fft.rfft(frame * window)) / (nfft / 4)
        acc = mag if acc is None else acc + mag
    mag = acc / max(hi - lo, 1)

    freqs = np.fft.rfftfreq(nfft, 1.0 / samplerate)
    db = 20.0 * np.log10(np.maximum(mag, 1e-10))
    return freqs, db


def analyze_session(session_dir: Path, nfft: int = 4096) -> dict:
    """Load a session and compute a mean spectrum per label."""
    meta_path = session_dir / "session.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no session.json in {session_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    samplerate = meta["samplerate"]

    by_label: dict[str, list[np.ndarray]] = {}
    for s in meta["samples"]:
        path = session_dir / s["filename"]
        if not path.exists():
            continue
        with wave.open(str(path), "rb") as w:
            raw = w.readframes(w.getnframes())
        seg = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        by_label.setdefault(s["label"], []).append(seg)

    spectra: dict[str, LabelSpectrum] = {}
    freqs = None
    for label, segs in by_label.items():
        acc = None
        peak_db = -np.inf
        for seg in segs:
            freqs, db = _spectrum_db(seg, samplerate, nfft)
            acc = db if acc is None else acc + db
            peak_db = max(peak_db, float(np.max(db)))
        mean_db = acc / len(segs)
        spectra[label] = LabelSpectrum(
            label=label,
            count=len(segs),
            freqs=freqs,
            mean_db=mean_db,
            peak_db=peak_db,
            peak_hz=float(freqs[int(np.argmax(mean_db))]),
        )

    return {"meta": meta, "samplerate": samplerate, "spectra": spectra, "freqs": freqs}


def suggest_bands(
    spec: LabelSpectrum,
    background_db: np.ndarray | None,
    max_bands: int = 2,
    min_width_hz: float = 40.0,
    contrast_db: float = 6.0,
    peak_drop_db: float = 12.0,
) -> list[dict]:
    """Propose frequency bands where this label stands out from the background.

    Regions are found by thresholding the contrast curve, merging adjacent bins,
    then keeping the strongest few. Edges are snapped to round numbers because
    false precision in a band edge helps nobody.

    The threshold is anchored to the contrast *peak*, not just to an absolute
    floor. A narrowband event sitting in near-silence clears any fixed floor
    across almost the whole spectrum -- thresholding on `contrast_db` alone
    returned one 40Hz-7kHz band covering everything. Requiring bins to sit
    within `peak_drop_db` of the strongest bin keeps the band on the energy that
    actually characterises the event.
    """
    freqs = spec.freqs
    if background_db is None:
        # No ambient reference: contrast against this label's own median level.
        background_db = np.full_like(spec.mean_db, float(np.median(spec.mean_db)))

    contrast = spec.mean_db - background_db

    # Ignore the extreme ends: sub-25Hz is inaudible rumble the motor cannot
    # meaningfully distinguish, and above ~14kHz there is little game content.
    usable = (freqs >= 25) & (freqs <= 14000)
    if not usable.any():
        return []

    peak_contrast = float(np.max(contrast[usable]))
    threshold = max(contrast_db, peak_contrast - peak_drop_db)
    mask = (contrast >= threshold) & usable

    regions: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(mask)))

    scored = []
    for lo_i, hi_i in regions:
        lo_hz, hi_hz = float(freqs[lo_i]), float(freqs[min(hi_i, len(freqs) - 1)])
        if hi_hz - lo_hz < min_width_hz:
            continue
        strength = float(np.mean(contrast[lo_i:hi_i]))
        scored.append({
            "low_hz": round(lo_hz, -1) or 25.0,
            "high_hz": round(hi_hz, -1),
            "mean_contrast_db": round(strength, 1),
            "peak_db": round(float(np.max(spec.mean_db[lo_i:hi_i])), 1),
        })

    scored.sort(key=lambda r: r["mean_contrast_db"], reverse=True)
    return scored[:max_bands]
