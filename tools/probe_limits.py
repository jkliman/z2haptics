"""
Map the accepted ranges and timing characteristics of the X1 API `vibrate` command.

The Control Panel validates duration and strength separately and reports
"ERROR: Invalid vibration duration" / "ERROR: Invalid vibration strength",
so boundary probing tells us the legal range without any guesswork.
"""

import statistics
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from x1_cli import send  # noqa: E402


def probe(label: str, values, fmt) -> None:
    print(f"\n=== {label} ===")
    for v in values:
        cmd = fmt(v)
        try:
            resp = send(cmd)
        except OSError as e:
            resp = f"!! {type(e).__name__}"
        print(f"  {cmd:<28} -> {resp}")
        time.sleep(0.12)


def main() -> int:
    # Duration bounds. uint16 in the Qt signal, so 65535 is the theoretical ceiling.
    probe(
        "duration (strength fixed at 50)",
        [0, 1, 5, 10, 50, 1000, 5000, 10000, 30000, 65535, 65536, 100000, -1],
        lambda v: f"vibrate {v} 50",
    )

    # Strength. UI text said ">100% is overdrive", so values above 100 may be legal.
    probe(
        "strength (duration fixed at 50)",
        [0, 1, 50, 99, 100, 101, 150, 200, 255, 256, 1000, -1],
        lambda v: f"vibrate 50 {v}",
    )

    # Malformed input, to confirm parser strictness.
    print("\n=== malformed ===")
    for cmd in ["vibrate", "vibrate 100", "vibrate 100 50 50", "vibrate abc 50",
                "vibrate 100.5 50", "VIBRATE 100 50", "Vibrate 100 50"]:
        try:
            resp = send(cmd)
        except OSError as e:
            resp = f"!! {type(e).__name__}"
        print(f"  {cmd:<28} -> {resp}")
        time.sleep(0.12)

    # Round-trip latency: this sets the floor on how responsive the haptics can be
    # and how fast we can safely fire pulses.
    print("\n=== round-trip latency (30x 'Profile Get') ===")
    lat = []
    for _ in range(30):
        t0 = time.perf_counter()
        send("Profile Get")
        lat.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.02)
    print(f"  min={min(lat):.2f}ms  median={statistics.median(lat):.2f}ms  "
          f"p95={sorted(lat)[int(len(lat) * 0.95)]:.2f}ms  max={max(lat):.2f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
