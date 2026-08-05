"""
Passive global input observation: mouse buttons and keys.

Fire events come from the mouse rather than from audio because a click is
ground truth. Audio detection has to infer a shot from a mix containing music,
explosions, teammates' weapons and overlapping fire; a button press is simply
the thing itself.

This only *observes* input. It installs no hooks that modify or inject events,
reads no process memory, and never touches the game. It is the same mechanism
used for the capture hotkeys, which ran alongside the game without trouble --
and the same category as push-to-talk in a voice client.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections.abc import Callable
from ctypes import wintypes

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short

# Virtual-key codes we care about.
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_MBUTTON = 0x04

VK_CODES: dict[str, int] = {
    "mouse_left": VK_LBUTTON,
    "mouse_right": VK_RBUTTON,
    "mouse_middle": VK_MBUTTON,
    "r": 0x52,
    "q": 0x51,
    "e": 0x45,
    "f": 0x46,
    "g": 0x47,
    "v": 0x56,
    "shift": 0x10,
    "ctrl": 0x11,
    "space": 0x20,
    "tab": 0x09,
    "escape": 0x1B,
    **{str(d): 0x30 + d for d in range(10)},          # '1'..'9', '0'
    **{f"f{i}": 0x6F + i for i in range(1, 13)},      # F1..F12
    **{f"num{d}": 0x60 + d for d in range(10)},       # numpad
}


def key_name_to_vk(name: str) -> int | None:
    return VK_CODES.get(name.strip().lower())


class InputWatcher:
    """Polls a set of keys and reports press/release transitions.

    Polling rather than a low-level hook: a hook runs inside the event stream
    and a slow callback stalls the whole desktop's input. Polling costs a little
    latency but cannot wedge anything, and at 250Hz the error is ~4ms -- far
    below what the motor can express anyway.
    """

    def __init__(
        self,
        keys: list[str],
        on_press: Callable[[str], None] | None = None,
        on_release: Callable[[str], None] | None = None,
        poll_hz: float = 250.0,
    ):
        self.keys = {}
        for name in keys:
            vk = key_name_to_vk(name)
            if vk is None:
                log.warning("unknown key %r, ignoring", name)
            else:
                self.keys[name] = vk

        self.on_press = on_press
        self.on_release = on_release
        self.interval = 1.0 / max(poll_hz, 1.0)

        self._down: dict[str, bool] = {name: False for name in self.keys}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = True

    def is_down(self, name: str) -> bool:
        return self._down.get(name, False)

    def start(self) -> None:
        if self._thread is not None:
            return
        # Clear any state latched before we started listening.
        for vk in self.keys.values():
            user32.GetAsyncKeyState(vk)
        self._down = {name: False for name in self.keys}

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="input-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "InputWatcher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            for name, vk in self.keys.items():
                pressed = bool(user32.GetAsyncKeyState(vk) & 0x8000)
                was = self._down[name]
                if pressed and not was:
                    self._down[name] = True
                    if self.enabled and self.on_press:
                        try:
                            self.on_press(name)
                        except Exception:
                            log.exception("on_press handler raised")
                elif not pressed and was:
                    self._down[name] = False
                    if self.enabled and self.on_release:
                        try:
                            self.on_release(name)
                        except Exception:
                            log.exception("on_release handler raised")
            time.sleep(self.interval)
