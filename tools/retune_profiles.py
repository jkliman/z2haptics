"""
One-shot migration: replace the old `dynamic_range` knob with a dB strength window.

Strength now comes from measured loudness, so each band needs a floor and ceiling
in dBFS. The floor is anchored to the band's own `gate` -- an event exactly at the
gate is the quietest thing that can ever trigger, so that is the natural bottom of
the strength range -- and the ceiling sits a fixed span above it.
"""

import math
import re
from pathlib import Path

SPAN_DB = 26.0  # dynamic range from "just audible" to "full strength"

PROFILES = Path(__file__).resolve().parent.parent / "profiles"


def main() -> None:
    for path in sorted(PROFILES.glob("*.yaml")):
        lines = path.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        current_gate = 0.003

        for line in lines:
            m = re.match(r"^(\s*)gate:\s*([0-9.]+)\s*$", line)
            if m:
                current_gate = float(m.group(2))
                out.append(line)
                continue

            m = re.match(r"^(\s*)dynamic_range:\s*[0-9.]+\s*$", line)
            if m:
                indent = m.group(1)
                floor = round(20.0 * math.log10(max(current_gate, 1e-9)), 1)
                ceil = round(floor + SPAN_DB, 1)
                out.append(f"{indent}level_floor_db: {floor}")
                out.append(f"{indent}level_ceil_db: {ceil}")
                continue

            out.append(line)

        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"  retuned {path.name}")


if __name__ == "__main__":
    main()
