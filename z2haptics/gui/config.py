"""
Application-level settings for the GUI, persisted to ~/.z2haptics/config.json.

Distinct from profiles: a profile describes how a *game* should feel, this
describes how the *app* should behave -- which device to capture, whether to
start with Windows, and so on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".z2haptics"
CONFIG_PATH = CONFIG_DIR / "config.json"

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "z2haptics"


@dataclass
class AppConfig:
    device_name: str = ""            # "" means the current Windows default output
    samplerate: int = 48000
    active_profile: str = "Default"
    auto_switch: bool = True
    start_engine_on_launch: bool = True
    start_minimised: bool = False
    autostart: bool = False
    show_notifications: bool = True

    # Test-panel state, remembered between sessions.
    test_duration_ms: int = 80
    test_strength: int = 70

    window_geometry: list[int] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        path = path or CONFIG_PATH
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("could not read %s (%s); using defaults", path, e)
            return cls()

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            log.info("ignoring unknown config keys: %s", sorted(unknown))
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except Exception:
            log.exception("could not save config to %s", path)


# -- Windows autostart --------------------------------------------------------

def _run_command() -> str:
    """The command Windows should run at login.

    pythonw.exe is used so no console window appears. Falls back to python.exe
    if the windowed launcher is missing.
    """
    import sys

    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    launcher = pythonw if pythonw.exists() else exe
    return f'"{launcher}" -m z2haptics.gui --minimised'


def get_autostart() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, RUN_VALUE)
            return True
    except OSError:
        return False


def set_autostart(enabled: bool) -> bool:
    """Add or remove the login entry. Returns the resulting state."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            if enabled:
                winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, _run_command())
            else:
                try:
                    winreg.DeleteValue(key, RUN_VALUE)
                except FileNotFoundError:
                    pass
        return enabled
    except OSError:
        log.exception("could not update autostart registry entry")
        return get_autostart()
