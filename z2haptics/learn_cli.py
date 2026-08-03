r"""
CLI commands for capturing real in-game event signatures and turning them into
a profile: `z2haptics learn` and `z2haptics analyze`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .audio import LoopbackCapture
from .learn import (
    DEFAULT_AMBIENT_KEY,
    DEFAULT_MARK_KEYS,
    DEFAULT_STOP_KEY,
    SESSIONS_DIR,
    HotkeyListener,
    LearnSession,
    analyze_session,
    suggest_bands,
)


def cmd_learn(args) -> int:
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if not labels:
        print("give at least one label, e.g. --labels gunshot,laser", file=sys.stderr)
        return 1
    if len(labels) > len(DEFAULT_MARK_KEYS):
        print(f"at most {len(DEFAULT_MARK_KEYS)} labels supported", file=sys.stderr)
        return 1

    session = LearnSession(
        name=args.name,
        labels=labels,
        samplerate=args.samplerate,
        pre_roll_s=args.pre_roll,
        post_roll_s=args.post_roll,
        root=Path(args.session_dir) if args.session_dir else None,
    )

    key_map: dict[str, str] = {}
    for key, label in zip(DEFAULT_MARK_KEYS, labels):
        key_map[key] = label
    key_map[DEFAULT_AMBIENT_KEY] = "ambient"
    key_map[DEFAULT_STOP_KEY] = "__stop__"

    stop_flag = {"stop": False}
    last_mark = {"label": "", "at": 0.0}

    def on_press(action: str) -> None:
        if action == "__stop__":
            stop_flag["stop"] = True
            return
        session.mark(action)
        last_mark["label"] = action
        last_mark["at"] = time.time()

    cap = LoopbackCapture(session.on_audio, device_name=args.device,
                          samplerate=args.samplerate, blocksize=512)
    try:
        cap.start()
    except Exception as e:
        print(f"capture failed: {e}", file=sys.stderr)
        return 1
    session.device = cap.resolved_name or ""

    keys = HotkeyListener(key_map, on_press)
    keys.start()

    print(f"\nLearning session {args.name!r}")
    print(f"  device:  {cap.resolved_name}")
    print(f"  writing: {session.dir}")
    print(f"  window:  {args.pre_roll:.2f}s before to {args.post_roll:.2f}s after each mark\n")
    print("  Hotkeys work while the game has focus. Tap the key right after you")
    print("  hear the event -- capture is retrospective, so a late tap is fine.\n")
    for key, label in key_map.items():
        if label == "__stop__":
            print(f"    {key:<6} finish and save")
        elif label == "ambient":
            print(f"    {key:<6} ambient / background reference (grab a few of these)")
        else:
            print(f"    {key:<6} {label}")
    print("\n  Ctrl+C also finishes.\n")

    try:
        while not stop_flag["stop"]:
            time.sleep(0.15)
            counts = "  ".join(f"{l}:{session.counts.get(l, 0)}"
                               for l in [*labels, "ambient"])
            recent = ""
            if last_mark["label"] and time.time() - last_mark["at"] < 1.2:
                recent = f"   <- {last_mark['label']}"
            sys.stdout.write(f"\r  captured  {counts}{recent}        ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        cap.stop()
        session.flush()

    total = sum(session.counts.values())
    print(f"\n\nSaved {total} sample(s) to {session.dir}")
    if total == 0:
        print("Nothing captured -- no marks were registered.")
        return 1

    ambient = session.counts.get("ambient", 0)
    if ambient == 0:
        print(f"\nNote: no ambient samples. Band suggestions will contrast each event")
        print(f"against its own median instead, which is less precise. Capture a few")
        print(f"with {DEFAULT_AMBIENT_KEY} during quiet gameplay for better results.")

    print(f"\nNext: z2haptics analyze {args.name}")
    return 0


def cmd_analyze(args) -> int:
    root = Path(args.session_dir) if args.session_dir else SESSIONS_DIR
    session_dir = Path(args.name) if Path(args.name).is_dir() else root / args.name
    if not session_dir.is_dir():
        print(f"no session at {session_dir}", file=sys.stderr)
        available = sorted(p.name for p in root.glob("*") if p.is_dir()) if root.is_dir() else []
        if available:
            print(f"available: {', '.join(available)}", file=sys.stderr)
        return 1

    try:
        result = analyze_session(session_dir, nfft=args.nfft)
    except Exception as e:
        print(f"analysis failed: {e}", file=sys.stderr)
        return 1

    spectra = result["spectra"]
    if not spectra:
        print("no usable samples in session", file=sys.stderr)
        return 1

    background = spectra.get("ambient")
    background_db = background.mean_db if background else None

    print(f"\nSession: {session_dir}")
    print(f"Samples: {sum(s.count for s in spectra.values())} across {len(spectra)} label(s)")
    if background_db is None:
        print("No ambient reference -- contrasting each label against its own median.")
    print()

    event_labels = [l for l in spectra if l != "ambient"]

    for label in event_labels:
        spec = spectra[label]
        print(f"=== {label}  ({spec.count} samples) ===")
        print(f"  spectral peak: {spec.peak_hz:.0f}Hz at {spec.peak_db:.1f}dB")

        _print_spectrum(spec, background_db)

        bands = suggest_bands(spec, background_db, max_bands=args.max_bands,
                              contrast_db=args.contrast)
        if bands:
            print("  suggested bands:")
            for b in bands:
                print(f"    {b['low_hz']:>6.0f} - {b['high_hz']:<6.0f}Hz   "
                      f"contrast +{b['mean_contrast_db']}dB   peak {b['peak_db']}dB")
        else:
            print(f"  no region exceeded +{args.contrast}dB contrast; "
                  f"try --contrast {max(3, args.contrast - 3)}")
        print()

    if args.write_profile:
        path = _write_profile(session_dir, args, spectra, background_db, event_labels)
        print(f"Wrote profile: {path}")
        print(f"Try it with:   z2haptics monitor -p {args.profile_name or session_dir.name}")
    else:
        print("Re-run with --write-profile to emit a starter profile YAML.")

    return 0


def _print_spectrum(spec, background_db, rows: int = 12) -> None:
    """Compact log-frequency bar chart of where this event's energy sits."""
    edges = np.geomspace(30, 14000, rows + 1)
    contrast = spec.mean_db - (background_db if background_db is not None
                               else np.median(spec.mean_db))

    vals = []
    for i in range(rows):
        sel = (spec.freqs >= edges[i]) & (spec.freqs < edges[i + 1])
        vals.append(float(np.mean(contrast[sel])) if sel.any() else 0.0)

    hi = max(max(vals), 1.0)
    print("  contrast vs background:")
    for i, v in enumerate(vals):
        width = int(max(v, 0) / hi * 34)
        lo_hz, hi_hz = edges[i], edges[i + 1]
        label = f"{lo_hz:>6.0f}-{hi_hz:<6.0f}"
        print(f"    {label}Hz |{'#' * width:<34}| {v:+6.1f}dB")


def _write_profile(session_dir, args, spectra, background_db, event_labels) -> Path:
    """Emit a starter profile from the measured bands.

    Values are a starting point, not a finished tune: gates and sensitivities
    still need a pass with `z2haptics monitor` against live gameplay.
    """
    name = args.profile_name or session_dir.name
    out = Path(args.out) if args.out else session_dir / f"{name}.yaml"

    lines = [
        f"name: {name}",
        "description: >",
        f"  Generated by `z2haptics analyze` from captured gameplay in session",
        f"  {session_dir.name!r}. Band edges come from measured spectral contrast;",
        "  gates, sensitivities and strengths are starting points -- verify with",
        "  `z2haptics monitor` and adjust.",
        "",
        "processes:",
        f"  # - {name.lower()}.exe",
        "",
        "strength_scale: 1.0",
        "",
        "limits:",
        "  min_gap_ms: 45",
        "  max_pulses_sec: 15",
        "  max_duty: 0.55",
        "",
        "bands:",
    ]

    # Lower-frequency events get longer, heavier pulses; high events get short
    # light accents. Priority follows the same ordering.
    entries = []
    for label in event_labels:
        bands = suggest_bands(spectra[label], background_db, max_bands=1,
                              contrast_db=args.contrast)
        if bands:
            entries.append((label, bands[0]))

    entries.sort(key=lambda e: e[1]["low_hz"])
    total = len(entries)

    for i, (label, b) in enumerate(entries):
        centre = (b["low_hz"] + b["high_hz"]) / 2
        if centre < 150:
            duration, smin, smax = 95, 55, 100
        elif centre < 1200:
            duration, smin, smax = 45, 35, 80
        else:
            duration, smin, smax = 22, 22, 52
        priority = total - i

        # Anchor the gate a little below the measured peak so ordinary instances
        # of the event clear it but ambience does not.
        gate = round(max(10 ** ((b["peak_db"] - 26.0) / 20.0), 0.0008), 5)
        floor_db = round(20.0 * np.log10(max(gate, 1e-9)), 1)

        lines += [
            f"  # measured contrast +{b['mean_contrast_db']}dB over background",
            f"  - name: {label}",
            f"    low_hz: {b['low_hz']:.0f}",
            f"    high_hz: {b['high_hz']:.0f}",
            f"    sensitivity: 1.6",
            f"    gate: {gate}",
            f"    refractory_ms: {90 if centre < 1200 else 130}",
            f"    duration_ms: {duration}",
            f"    strength_min: {smin}",
            f"    strength_max: {smax}",
            f"    level_floor_db: {floor_db}",
            f"    level_ceil_db: {round(floor_db + 26.0, 1)}",
            f"    priority: {priority}",
            "",
        ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def register(sub) -> None:
    p = sub.add_parser("learn", help="capture in-game event signatures with hotkeys")
    p.add_argument("--name", required=True, help="session name, e.g. avatar")
    p.add_argument("--labels", required=True,
                   help="comma-separated event names, e.g. gunshot,laser,explosion")
    p.add_argument("-d", "--device", help="loopback device name substring")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--pre-roll", type=float, default=0.65,
                   help="seconds of audio kept before each mark")
    p.add_argument("--post-roll", type=float, default=0.20,
                   help="seconds of audio kept after each mark")
    p.add_argument("--session-dir", help="override where sessions are stored")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("analyze", help="turn a learn session into band suggestions")
    p.add_argument("name", help="session name or path")
    p.add_argument("--nfft", type=int, default=4096)
    p.add_argument("--contrast", type=float, default=6.0,
                   help="dB above background required to call a region distinctive")
    p.add_argument("--max-bands", type=int, default=2, help="bands to report per label")
    p.add_argument("--write-profile", action="store_true", help="emit a starter YAML")
    p.add_argument("--profile-name", help="name for the generated profile")
    p.add_argument("--out", help="explicit output path for the profile")
    p.add_argument("--session-dir", help="override where sessions are stored")
    p.set_defaults(func=cmd_analyze)
