r"""
Client for the Swiftpoint X1 Control Panel local command API.

The Control Panel (v3.0.9.1+) opens a Qt QLocalServer named pipe when
``X1API=true`` is set in its settings.ini. The pipe speaks a line-oriented text
protocol and replies ``OK`` or ``ERROR: ...`` to every command.

Discovered by static analysis of the Control Panel binary; this is an
undocumented internal interface, not a published SDK. See docs/PROTOCOL.md.

Measured characteristics on the reference machine:
    vibrate round-trip   ~1.7 ms median
    fire-and-forget      ~0.25 ms
    burst ceiling        ~3000 commands/sec
    connection           reusable across many commands

The transport is therefore never the bottleneck. The constraint that matters is
the physical motor, which is why HapticSink applies motor-protective rate limiting
rather than transport-protective throttling.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

PIPE_V2 = r"\\.\pipe\swiftpoint.x1.v2.command"
PIPE_LEGACY = r"\\.\pipe\swiftpoint.x1.profileswitch"

# The Control Panel validates strength to 0-100 inclusive and rejects anything
# outside it. Duration is NOT validated -- out-of-range values wrap silently
# through a uint16, so a "70000ms" pulse becomes 4464ms. We clamp it ourselves.
STRENGTH_MIN, STRENGTH_MAX = 0, 100
DURATION_MIN, DURATION_MAX = 0, 65535


class X1Error(RuntimeError):
    """The API returned an ERROR response or the pipe could not be reached."""


class X1Connection:
    """A single reusable connection to the X1 command pipe.

    Thread-safe: every command is a write followed by a blocking read, so two
    threads sharing one handle would interleave and each could consume the
    other's response -- which deadlocks, because both then block forever waiting
    for a reply that has already been read. A lock serialises whole round trips.

    Being safe to share is not the same as being cheap to share. `Profile Set`
    writes configuration to the mouse and takes far longer than a `vibrate`, so
    callers must never issue commands from a UI thread; see EngineController,
    which routes all device I/O through a dedicated worker.
    """

    def __init__(self, pipe: str = PIPE_V2, fallback: str | None = PIPE_LEGACY):
        self.pipe = pipe
        self.fallback = fallback
        self._f = None
        self._active_pipe: str | None = None
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._f is not None

    @property
    def active_pipe(self) -> str | None:
        return self._active_pipe

    def connect(self) -> None:
        with self._lock:
            if self._f is not None:
                return
            last: Exception | None = None
            for name in (self.pipe, self.fallback):
                if not name:
                    continue
                try:
                    self._f = open(name, "r+b", buffering=0)
                    self._active_pipe = name
                    log.info("connected to X1 API at %s", name)
                    return
                except OSError as e:
                    last = e
            raise X1Error(
                "Could not open the X1 API pipe. Check that the Swiftpoint X1 Control "
                "Panel is running and that X1API=true is set in its settings.ini "
                f"(then restart it). Last error: {last}"
            )

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                try:
                    self._f.close()
                except OSError:
                    pass
                self._f = None
                self._active_pipe = None

    def command(self, cmd: str, expect_response: bool = True) -> str:
        """Send one command and return its response.

        Responses are always read, even when the caller ignores them, so the
        server's write buffer cannot fill up over a long session.
        """
        with self._lock:
            self.connect()
            assert self._f is not None
            try:
                self._f.write((cmd + "\n").encode("utf-8"))
                self._f.flush()
                if not expect_response:
                    return ""
                data = self._f.read(4096)
            except OSError as e:
                self.close()
                raise X1Error(f"X1 API write/read failed: {e}") from e
        return data.decode("utf-8", errors="replace").strip()

    # -- convenience wrappers over the documented command set -----------------

    def vibrate(self, duration_ms: int, strength: int) -> str:
        d = max(DURATION_MIN, min(DURATION_MAX, int(duration_ms)))
        s = max(STRENGTH_MIN, min(STRENGTH_MAX, int(strength)))
        return self.command(f"vibrate {d} {s}")

    def profile_get(self) -> str:
        return self.command("Profile Get")

    def profile_list(self) -> list[str]:
        resp = self.command("Profile List")
        if resp.startswith("ERROR"):
            raise X1Error(resp)
        return [line.strip() for line in resp.splitlines() if line.strip()]

    def profile_set(self, name: str) -> str:
        return self.command(f"Profile Set {name}")

    def dpi(self, value: int) -> str:
        return self.command(f"DPI {int(value)}")

    def rgb_fixed(self, hex_colour: str) -> str:
        if not hex_colour.startswith("#"):
            hex_colour = "#" + hex_colour
        return self.command(f"rgb fixed {hex_colour}")

    def rgb_override(self, enabled: bool) -> str:
        return self.command(f"rgb override {'true' if enabled else 'false'}")

    def oled_message(self, text: str) -> str:
        # The Control Panel enforces a 19-character limit and errors above it.
        return self.command(f"OLED message {text[:19]}")


@dataclass
class Pulse:
    """One haptic event queued for delivery."""

    duration_ms: int
    strength: int
    label: str = ""
    priority: int = 0


class HapticSink:
    """Thread-safe, motor-protective delivery of pulses to the mouse.

    Audio callbacks call :meth:`fire` and never block: pulses go onto a bounded
    queue drained by a worker thread. If the queue is full the weakest pending
    pulse is dropped rather than the newest, so a loud transient still lands
    during a busy passage.

    Motor protection has three layers:

    ``min_gap_ms``      hard refractory period between pulses
    ``max_pulses_sec``  sliding-window cap on pulse count
    ``max_duty``        fraction of wall-clock the motor may be driven

    These exist because the API happily accepts thousands of commands per second
    but the motor is a physical actuator with spin-up time and finite life.
    """

    def __init__(
        self,
        connection: X1Connection | None = None,
        min_gap_ms: float = 45.0,
        max_pulses_sec: float = 14.0,
        max_duty: float = 0.55,
        queue_size: int = 24,
        dry_run: bool = False,
    ):
        self.conn = connection or X1Connection()
        self.min_gap_ms = min_gap_ms
        self.max_pulses_sec = max_pulses_sec
        self.max_duty = max_duty
        self.dry_run = dry_run

        self._q: queue.Queue[Pulse] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._last_fire = 0.0
        self._recent: list[tuple[float, float]] = []  # (timestamp, duration_s)

        # Counters, surfaced by the monitor UI.
        self.sent = 0
        self.dropped_rate = 0
        self.dropped_queue = 0
        self.errors = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self.dry_run:
            self.conn.connect()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="haptic-sink", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.conn.close()

    def __enter__(self) -> "HapticSink":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- producer side --------------------------------------------------------

    def fire(self, pulse: Pulse) -> bool:
        """Queue a pulse. Returns False if it was dropped. Never blocks."""
        try:
            self._q.put_nowait(pulse)
            return True
        except queue.Full:
            # Make room by discarding the weakest queued pulse, but only if this
            # one is actually stronger -- otherwise drop the newcomer.
            weakest, items = None, []
            try:
                while True:
                    items.append(self._q.get_nowait())
            except queue.Empty:
                pass
            if items:
                weakest = min(items, key=lambda p: (p.priority, p.strength))
            if weakest is not None and (pulse.priority, pulse.strength) > (
                weakest.priority, weakest.strength
            ):
                items.remove(weakest)
                items.append(pulse)
                accepted = True
            else:
                accepted = False
            for p in items:
                try:
                    self._q.put_nowait(p)
                except queue.Full:
                    break
            self.dropped_queue += 1
            return accepted

    # -- consumer side --------------------------------------------------------

    def _budget_allows(self, now: float, duration_s: float) -> bool:
        """Check the pulse against the refractory, rate and duty-cycle limits."""
        if (now - self._last_fire) * 1000.0 < self.min_gap_ms:
            return False

        window = 1.0
        self._recent = [(t, d) for (t, d) in self._recent if now - t < window]

        if len(self._recent) >= self.max_pulses_sec:
            return False

        driven = sum(d for (_, d) in self._recent) + duration_s
        if driven / window > self.max_duty:
            return False
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                pulse = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.perf_counter()
            duration_s = pulse.duration_ms / 1000.0

            if not self._budget_allows(now, duration_s):
                self.dropped_rate += 1
                continue

            self._last_fire = now
            self._recent.append((now, duration_s))

            if self.dry_run:
                self.sent += 1
                log.debug("[dry-run] vibrate %d %d (%s)",
                          pulse.duration_ms, pulse.strength, pulse.label)
                continue

            try:
                resp = self.conn.vibrate(pulse.duration_ms, pulse.strength)
                if resp.startswith("ERROR"):
                    self.errors += 1
                    log.warning("X1 API: %s (pulse %s)", resp, pulse.label)
                else:
                    self.sent += 1
            except X1Error as e:
                self.errors += 1
                log.warning("haptic send failed, will reconnect: %s", e)
                time.sleep(0.25)

    # -- introspection --------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "sent": self.sent,
            "dropped_rate": self.dropped_rate,
            "dropped_queue": self.dropped_queue,
            "errors": self.errors,
        }
