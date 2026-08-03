"""
WASAPI loopback capture of the Windows default output device.

Loopback reads whatever is already playing on the chosen output, so no virtual
cable, Stereo Mix, or game-side integration is needed. PortAudio in the common
sounddevice wheels does not expose loopback devices, so we use `soundcard`,
which opens a WASAPI loopback capture client directly against a speaker.

Capture runs on its own thread and hands mono blocks to a callback. The callback
runs on that thread, so it must not block -- HapticSink's queue exists for
exactly this reason.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
import soundcard as sc

from .analysis import to_mono

log = logging.getLogger(__name__)


class LoopbackCapture:
    """Continuously capture the default (or named) output device."""

    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        device_name: str | None = None,
        samplerate: int = 48000,
        blocksize: int = 512,
    ):
        self.callback = callback
        self.device_name = device_name
        self.samplerate = samplerate
        self.blocksize = blocksize

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mic = None
        self.resolved_name: str | None = None
        self.overruns = 0

    # -- device resolution ----------------------------------------------------

    def _resolve(self):
        if self.device_name:
            # Try an exact/substring match against loopback-capable devices first.
            candidates = [
                m for m in sc.all_microphones(include_loopback=True)
                if m.isloopback and self.device_name.lower() in m.name.lower()
            ]
            if candidates:
                return candidates[0]
            # Fall back to treating it as a speaker name.
            try:
                return sc.get_microphone(id=self.device_name, include_loopback=True)
            except Exception as e:
                raise RuntimeError(
                    f"No loopback device matching {self.device_name!r}. "
                    f"Run `z2haptics devices` to list options."
                ) from e

        spk = sc.default_speaker()
        try:
            return sc.get_microphone(id=str(spk.name), include_loopback=True)
        except Exception:
            loopbacks = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
            if not loopbacks:
                raise RuntimeError("No WASAPI loopback devices available")
            log.warning("could not open default speaker loopback, using %s", loopbacks[0].name)
            return loopbacks[0]

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._mic = self._resolve()
        self.resolved_name = self._mic.name
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="loopback", daemon=True)
        self._thread.start()
        log.info("capturing loopback from %s @ %dHz", self.resolved_name, self.samplerate)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> "LoopbackCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _run(self) -> None:
        try:
            with self._mic.recorder(
                samplerate=self.samplerate, channels=2, blocksize=self.blocksize
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self.blocksize)
                    if data.size == 0:
                        continue
                    try:
                        self.callback(to_mono(data))
                    except Exception:
                        log.exception("capture callback raised")
        except Exception:
            log.exception("loopback capture stopped unexpectedly")


def list_devices() -> list[tuple[str, bool]]:
    """Return (name, is_default) for every loopback-capable device."""
    try:
        default = sc.default_speaker().name
    except Exception:
        default = ""
    out = []
    for m in sc.all_microphones(include_loopback=True):
        if m.isloopback:
            out.append((m.name, m.name == default))
    return out
