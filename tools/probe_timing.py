"""
Measure real X1 API timing: blocking round-trip vs fire-and-forget write cost,
and the maximum sustainable pulse rate.

These numbers decide the engine architecture -- specifically whether haptic sends
can happen inline or must be offloaded to a worker thread with a rate limiter.
"""

import statistics
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from x1_cli import send, send_nowait  # noqa: E402


def stats(label: str, samples: list[float]) -> None:
    s = sorted(samples)
    print(f"  {label:<34} min={s[0]:6.2f}  med={statistics.median(s):6.2f}  "
          f"p95={s[int(len(s) * 0.95)]:6.2f}  max={s[-1]:6.2f}   (ms)")


def main() -> int:
    print("=== blocking round-trip: 'Profile Get' (read waits for response) ===")
    lat = []
    for _ in range(40):
        t0 = time.perf_counter()
        send("Profile Get")
        lat.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.01)
    stats("round-trip w/ response", lat)

    print("\n=== blocking round-trip: 'vibrate 1 0' (no-op pulse) ===")
    lat = []
    for _ in range(40):
        t0 = time.perf_counter()
        send("vibrate 1 0")
        lat.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.01)
    stats("vibrate round-trip", lat)

    print("\n=== fire-and-forget write cost (no response read) ===")
    lat = []
    for _ in range(40):
        t0 = time.perf_counter()
        send_nowait("vibrate 1 0")
        lat.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.01)
    stats("send_nowait", lat)

    print("\n=== burst: 50 fire-and-forget pulses back to back ===")
    t0 = time.perf_counter()
    failures = 0
    for _ in range(50):
        try:
            send_nowait("vibrate 1 0")
        except OSError:
            failures += 1
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  50 sends in {elapsed:.1f}ms -> {elapsed / 50:.2f}ms/send, "
          f"{1000 / (elapsed / 50):.0f} sends/sec ceiling, {failures} failures")

    print("\n=== connection reuse: does the server keep a connection open? ===")
    try:
        f = open(r"\\.\pipe\swiftpoint.x1.v2.command", "r+b", buffering=0)
        f.write(b"Profile Get\n")
        f.flush()
        r1 = f.read(256)
        f.write(b"Profile Get\n")
        f.flush()
        try:
            r2 = f.read(256)
        except OSError as e:
            r2 = f"!! {e}"
        f.close()
        print(f"  first  -> {r1!r}")
        print(f"  second -> {r2!r}")
        print("  => connection is REUSABLE" if r2 and not str(r2).startswith("!!")
              else "  => one command per connection")
    except OSError as e:
        print(f"  !! {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
