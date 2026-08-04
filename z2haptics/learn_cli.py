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
    build_weapon_set,
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
            time.sleep(0.1)
            counts = "  ".join(f"{l}:{session.counts.get(l, 0)}"
                               for l in [*labels, "ambient"])
            recent = ""
            if last_mark["label"] and time.time() - last_mark["at"] < 1.2:
                recent = f"   <- {last_mark['label']}"

            # Live level, so silence is obvious immediately rather than after
            # the session is analysed and found to be worthless.
            peak = session.recent_peak
            bars = int(min(peak * 60, 20))
            meter = "#" * bars + "-" * (20 - bars)
            warn = "  NO AUDIO" if session.session_peak < 0.002 else ""

            sys.stdout.write(f"\r  [{meter}] {peak:6.4f}{warn}   {counts}{recent}    ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        cap.stop()
        session.flush()

    total = sum(session.counts.values())
    print(f"\n\nSaved {total} sample(s) to {session.dir}")

    if session.session_peak < 0.002:
        print(f"\nWARNING: peak level over the whole session was "
              f"{session.session_peak:.5f} -- effectively silence.")
        print("These captures are unusable. Likely causes:")
        print("  * the game mutes its audio when it loses focus")
        print("  * game audio is routed to a device other than the one captured")
        print("Run `python tools/find_game_audio.py` while firing to find the right")
        print("device, then pass it with --device.")
        return 1

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


def cmd_weapons(args) -> int:
    """Build a weapon set from a learn session, or inspect an existing one."""
    from .weapons import discover, resolve, save, separability

    if args.live:
        return _live_classify(args)

    if args.list:
        sets = discover()
        if not sets:
            print("No weapon sets. Build one with:")
            print("  z2haptics learn --name bf6 --labels ak74,m4,sniper")
            print("  z2haptics weapons --from bf6")
            return 0
        for ws in sets.values():
            print(f"\n{ws.name}   ({len(ws.templates)} weapons)  {ws.source}")
            for t in ws.templates:
                shaping = []
                if t.duration_ms is not None:
                    shaping.append(f"{t.duration_ms}ms")
                if t.strength_min is not None or t.strength_max is not None:
                    shaping.append(f"{t.strength_min}-{t.strength_max}")
                extra = f"   [{', '.join(shaping)}]" if shaping else ""
                print(f"    {t.name:<14} {t.samples:>3} samples   "
                      f"consistency {t.spread:.2f}{extra}")
        return 0

    if not args.from_session:
        print("Give --from SESSION to build, or --list to inspect.", file=sys.stderr)
        return 1

    root = Path(args.session_dir) if args.session_dir else SESSIONS_DIR
    session_dir = (Path(args.from_session) if Path(args.from_session).is_dir()
                   else root / args.from_session)
    if not session_dir.is_dir():
        print(f"no session at {session_dir}", file=sys.stderr)
        return 1

    merge: dict[str, str] = {}
    if args.merge:
        for pair in args.merge.split(","):
            src, _, dst = pair.partition("=")
            if src.strip() and dst.strip():
                merge[src.strip()] = dst.strip()

    report: list = []
    try:
        ws = build_weapon_set(session_dir, set_name=args.name, report=report,
                              merge=merge or None)
    except Exception as e:
        print(f"could not build weapon set: {e}", file=sys.stderr)
        return 1

    skipped = report[0] if report else {}
    if skipped:
        total_skipped = sum(skipped.values())
        print(f"Skipped {total_skipped} silent capture(s): "
              f"{', '.join(f'{k} x{v}' for k, v in skipped.items())}")
        print("Those marks landed on silence -- game muted, wrong device, or the")
        print("shot fell outside the capture window.\n")

    if not ws.templates:
        print("No usable samples -- every capture was silent, or nothing was "
              "labelled.", file=sys.stderr)
        return 1

    print(f"Weapon set {ws.name!r} from {session_dir}\n")
    for t in ws.templates:
        warn = "  <- few samples" if t.samples < 4 else ""
        print(f"  {t.name:<14} {t.samples:>3} samples   "
              f"consistency {t.spread:.2f}{warn}")

    print("\nConfusability (1.00 = indistinguishable):")
    pairs = separability(ws)
    if not pairs:
        print("  (only one weapon)")
    for a, b, sim in pairs[:8]:
        flag = "  <- too similar to tell apart" if sim > 0.9 else ""
        print(f"  {a:<12} vs {b:<12} {sim:5.2f}{flag}")

    path = save(ws, Path(args.out) if args.out else None)
    print(f"\nWrote {path}")
    print("\nTo use it, add to your profile:")
    print(f"  weapon_set: {ws.name}")
    print("and set `classify: true` on the band that should be identified.")

    weak = [t.name for t in ws.templates if t.spread < 0.8]
    if weak:
        print(f"\nLow consistency: {', '.join(weak)}. Those captures disagree with "
              f"each other, usually because other sounds bled in. Recapture them "
              f"somewhere quieter if classification proves unreliable.")
    return 0


def _live_classify(args) -> int:
    """Name each detected shot in real time, so accuracy can be judged in game.

    Prints every onset with its best match and confidence, including the ones it
    declines -- a declined shot falls back to a generic pulse, which is the safe
    outcome, and seeing how often that happens is the point.
    """
    import time

    from .analysis import BandAnalyzer
    from .audio import LoopbackCapture
    from .profiles import discover as discover_profiles
    from .weapons import resolve

    ws = resolve(args.set) if args.set else None
    if ws is None:
        print(f"No weapon set {args.set!r}. Build one first:", file=sys.stderr)
        print("  z2haptics weapons --from SESSION", file=sys.stderr)
        return 1

    profiles = discover_profiles()
    profile = profiles.get(args.profile) if args.profile else None
    if profile is None:
        profile = profiles.get("Battlefield 6") or profiles.get("FPS")
    if profile is None:
        print("No suitable profile found.", file=sys.stderr)
        return 1

    bands = [b for b in profile.bands if b.classify] or profile.bands
    analyzer = BandAnalyzer(bands, samplerate=args.samplerate)
    analyzer.compute_features = True

    tally: dict[str, int] = {}
    declined = 0

    def on_audio(mono):
        nonlocal declined
        for onset in analyzer.push(mono):
            match, score = ws.classify(onset.feature)
            if match is None:
                declined += 1
                name, mark = "(declined)", " "
            else:
                name = match.name
                tally[name] = tally.get(name, 0) + 1
                mark = "*"
            print(f"  {mark} {onset.band:<10} {name:<14} conf {score:5.2f}   "
                  f"level {onset.level_db:6.1f}dB  flat {onset.flatness:.2f}")

    cap = LoopbackCapture(on_audio, device_name=args.device,
                          samplerate=args.samplerate, blocksize=512)
    try:
        cap.start()
    except Exception as e:
        print(f"capture failed: {e}", file=sys.stderr)
        return 1

    print(f"Weapon set : {ws.name}  ({len(ws.templates)} weapons)")
    print(f"Profile    : {profile.name}   bands: {', '.join(b.name for b in bands)}")
    print(f"Device     : {cap.resolved_name}")
    print(f"Thresholds : confidence >= {ws.min_confidence}, margin >= {ws.min_margin}")
    print("\nFire away. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()

    total = sum(tally.values()) + declined
    print(f"\n\n{total} onsets:")
    for name, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {count:>4}")
    print(f"  {'(declined)':<14} {declined:>4}")
    if total:
        print(f"\nnamed {1 - declined / total:.0%} of onsets")
        if declined / total > 0.7:
            print("Mostly declining. Either the templates need more/cleaner samples,")
            print("or these weapons genuinely sound too alike to separate.")
    return 0


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

    p = sub.add_parser("weapons", help="build or inspect a per-weapon classifier")
    p.add_argument("--from", dest="from_session", metavar="SESSION",
                   help="learn session to build the weapon set from")
    p.add_argument("--name", help="name for the weapon set (default: session name)")
    p.add_argument("--merge", metavar="A=X,B=X",
                   help="merge labels into one class, e.g. assault=rifle,smg=rifle. "
                        "Use when two weapons are too alike to separate -- one "
                        "combined class still distinguishes them from the rest.")
    p.add_argument("--out", help="explicit output path")
    p.add_argument("--list", action="store_true", help="list existing weapon sets")
    p.add_argument("--live", action="store_true",
                   help="name each shot in real time, to check accuracy in game")
    p.add_argument("--set", help="weapon set to use with --live")
    p.add_argument("--profile", help="profile whose bands to detect with (--live)")
    p.add_argument("-d", "--device", help="loopback device name substring")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--session-dir", help="override where sessions are stored")
    p.set_defaults(func=cmd_weapons)
