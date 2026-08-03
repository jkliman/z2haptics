r"""
Command line interface.

    z2haptics doctor              check the setup end to end
    z2haptics enable-api          flip X1API=true in the Control Panel settings
    z2haptics devices             list loopback-capable output devices
    z2haptics profiles            list available profiles
    z2haptics test                fire a sweep of test pulses
    z2haptics monitor [-p NAME]   live band levels and onsets, no haptics
    z2haptics run [-p NAME]       run the engine

    z2haptics learn --name avatar --labels gunshot,laser
                                  capture real event signatures with hotkeys
    z2haptics analyze avatar --write-profile
                                  derive bands from what you captured
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from . import __version__, learn_cli
from .analysis import BandAnalyzer
from .api import PIPE_LEGACY, PIPE_V2, HapticSink, Pulse, X1Connection, X1Error
from .audio import LoopbackCapture, list_devices
from .engine import HapticEngine
from .foreground import foreground_process
from .profiles import discover, match_profile

SETTINGS_INI = Path.home() / "AppData" / "Local" / "Swiftpoint X1 Control Panel" / "settings.ini"


# -- helpers ------------------------------------------------------------------

def _pipe_present() -> str | None:
    import os
    try:
        names = os.listdir(r"\\.\pipe")
    except OSError:
        return None
    for candidate in (PIPE_V2, PIPE_LEGACY):
        short = candidate.rsplit("\\", 1)[-1]
        if short in names:
            return candidate
    return None


def _load_profiles(args) -> dict:
    extra = [Path(args.profile_dir)] if getattr(args, "profile_dir", None) else None
    profiles = discover(extra)
    if not profiles:
        print("No profiles found.", file=sys.stderr)
        sys.exit(1)
    return profiles


def _pick_profile(profiles: dict, name: str | None):
    if name:
        for key, p in profiles.items():
            if key.lower() == name.lower():
                return p
        print(f"No profile named {name!r}. Available: {', '.join(profiles)}", file=sys.stderr)
        sys.exit(1)

    proc = foreground_process()
    matched = match_profile(profiles, proc)
    if matched:
        print(f"Matched profile {matched.name!r} for foreground process {proc!r}")
        return matched
    fallback = profiles.get("Default") or next(iter(profiles.values()))
    print(f"No profile matches {proc or 'unknown process'}; using {fallback.name!r}")
    return fallback


# -- commands -----------------------------------------------------------------

def cmd_doctor(args) -> int:
    ok = True
    print(f"z2haptics {__version__}\n")

    print("[1] Swiftpoint X1 Control Panel process")
    import subprocess
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Swiftpoint X1 Control Panel.exe", "/NH"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        if "Swiftpoint" in out:
            print("    running")
        else:
            print("    NOT RUNNING -- start it from the Start menu")
            ok = False
    except Exception as e:
        print(f"    could not check: {e}")

    print("\n[2] X1API setting")
    if SETTINGS_INI.exists():
        text = SETTINGS_INI.read_text(encoding="utf-8", errors="replace")
        if "X1API=true" in text:
            print(f"    enabled in {SETTINGS_INI}")
        elif "X1API=false" in text:
            print(f"    DISABLED in {SETTINGS_INI}")
            print("    fix: z2haptics enable-api   (then restart the Control Panel)")
            ok = False
        else:
            print("    key not present; the Control Panel may be too old")
            ok = False
    else:
        print(f"    settings.ini not found at {SETTINGS_INI}")
        ok = False

    print("\n[3] Command pipe")
    pipe = _pipe_present()
    if pipe:
        print(f"    listening: {pipe}")
    else:
        print("    no pipe found -- restart the Control Panel after enabling the API")
        ok = False

    print("\n[4] Device response")
    if pipe:
        try:
            conn = X1Connection()
            conn.connect()
            active = conn.profile_get()
            print(f"    active X1 profile: {active}")
            resp = conn.vibrate(1, 0)
            if resp.startswith("ERROR"):
                print(f"    vibrate rejected: {resp}")
                ok = False
            else:
                print("    vibrate accepted")
            conn.close()
        except X1Error as e:
            print(f"    {e}")
            ok = False

    print("\n[5] Loopback audio")
    try:
        devices = list_devices()
        if devices:
            for name, is_default in devices[:8]:
                print(f"    {'*' if is_default else ' '} {name}")
            if len(devices) > 8:
                print(f"      ... and {len(devices) - 8} more")
        else:
            print("    no loopback devices found")
            ok = False
    except Exception as e:
        print(f"    capture check failed: {e}")
        ok = False

    print("\n[6] Profiles")
    try:
        profiles = discover()
        print(f"    {len(profiles)} loaded: {', '.join(profiles)}")
    except Exception as e:
        print(f"    {e}")
        ok = False

    print("\n" + ("All checks passed." if ok else "Some checks failed -- see above."))
    return 0 if ok else 1


def cmd_enable_api(args) -> int:
    if not SETTINGS_INI.exists():
        print(f"settings.ini not found at {SETTINGS_INI}", file=sys.stderr)
        return 1

    text = SETTINGS_INI.read_text(encoding="utf-8", errors="replace")
    if "X1API=true" in text:
        print("Already enabled.")
        return 0
    if "X1API=false" not in text:
        print("No X1API key in settings.ini -- Control Panel may be too old.", file=sys.stderr)
        return 1

    backup = SETTINGS_INI.with_suffix(".ini.bak")
    shutil.copy2(SETTINGS_INI, backup)
    SETTINGS_INI.write_text(text.replace("X1API=false", "X1API=true"), encoding="utf-8")
    print(f"Enabled. Backup written to {backup}")
    print("\nNow FULLY restart the Control Panel (quit from the tray, then relaunch).")
    print("The setting is only read at startup, and the app rewrites settings.ini")
    print("on a clean exit -- so quit it before editing if you ever revert this by hand.")
    return 0


def cmd_devices(args) -> int:
    devices = list_devices()
    if not devices:
        print("No loopback-capable devices found.")
        return 1
    print("Loopback-capable output devices (* = current default):\n")
    for name, is_default in devices:
        print(f"  {'*' if is_default else ' '} {name}")
    return 0


def cmd_profiles(args) -> int:
    profiles = _load_profiles(args)
    print(f"{len(profiles)} profile(s):\n")
    for p in profiles.values():
        procs = ", ".join(p.processes) if p.processes else "(no auto-match)"
        print(f"  {p.name}")
        print(f"    {p.description.strip().splitlines()[0] if p.description else ''}")
        print(f"    processes: {procs}")
        print(f"    bands:     {', '.join(b.name for b in p.bands)}")
        if p.x1_profile:
            print(f"    x1 profile: {p.x1_profile}")
        print(f"    source:    {p.source}")
        print()
    return 0


def cmd_test(args) -> int:
    print("Firing test pulses. Hold the mouse.\n")
    try:
        conn = X1Connection()
        conn.connect()
    except X1Error as e:
        print(e, file=sys.stderr)
        return 1

    print("  strength ramp at 60ms:")
    for s in (10, 25, 40, 55, 70, 85, 100):
        print(f"    strength {s:3d} ... ", end="", flush=True)
        print(conn.vibrate(60, s))
        time.sleep(0.6)

    print("\n  duration ramp at strength 70:")
    for d in (10, 25, 50, 100, 200, 400):
        print(f"    duration {d:3d}ms ... ", end="", flush=True)
        print(conn.vibrate(d, 70))
        time.sleep(0.9)

    print("\n  simulated FPS burst (5 shots, 90ms apart):")
    for _ in range(5):
        conn.vibrate(45, 65)
        time.sleep(0.09)
    time.sleep(0.6)

    print("  simulated explosion (one heavy pulse):")
    conn.vibrate(110, 100)
    time.sleep(0.5)

    conn.close()
    print("\nDone.")
    return 0


def cmd_monitor(args) -> int:
    """Live band levels and onset detection with no haptic output.

    This is the tuning tool: watch which bands light up on the sounds you care
    about, then adjust sensitivity and gate in the profile YAML until the right
    events -- and only the right events -- register.
    """
    profiles = _load_profiles(args)
    profile = _pick_profile(profiles, args.profile)
    print(f"\nMonitoring with profile {profile.name!r}. Ctrl+C to stop.\n")

    analyzer = BandAnalyzer(profile.bands, samplerate=args.samplerate)
    counts = {b.name: 0 for b in profile.bands}
    recent: list[tuple[float, str, float]] = []

    def on_audio(mono):
        for onset in analyzer.push(mono):
            counts[onset.band] += 1
            recent.append((time.time(), onset.band, onset.strength))
            del recent[:-6]

    cap = LoopbackCapture(on_audio, device_name=args.device,
                          samplerate=args.samplerate, blocksize=512)
    try:
        cap.start()
    except Exception as e:
        print(f"capture failed: {e}", file=sys.stderr)
        return 1

    print(f"device: {cap.resolved_name}\n")
    try:
        while True:
            time.sleep(0.1)
            lines = []
            for band in profile.bands:
                st = analyzer.state[band.name]
                gated = st.level < band.gate
                bar = "#" * int(min(st.level / max(band.gate, 1e-9) * 12, 44))
                flag = " " if not gated else "."
                lines.append(
                    f"  {band.name:<9} {band.low_hz:>5.0f}-{band.high_hz:<5.0f}Hz "
                    f"{flag}|{bar:<44}| n={counts[band.name]:<5}"
                )
            recent_txt = "  ".join(
                f"{b}:{s:.2f}" for (_, b, s) in recent[-5:]
            ) or "(none yet)"
            out = "\n".join(lines) + f"\n\n  recent onsets: {recent_txt}"
            sys.stdout.write("\033[H\033[J" + f"profile: {profile.name}\n\n" + out + "\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nstopping...")
    finally:
        cap.stop()
    return 0


def cmd_run(args) -> int:
    profiles = _load_profiles(args)
    profile = _pick_profile(profiles, args.profile)
    fallback = profiles.get("Default")

    if not args.dry_run and not _pipe_present():
        print("The X1 command pipe is not listening. Run `z2haptics doctor`.", file=sys.stderr)
        return 1

    engine = HapticEngine(
        profile=profile,
        profiles=profiles,
        device_name=args.device,
        samplerate=args.samplerate,
        dry_run=args.dry_run,
        auto_switch=not args.no_auto_switch,
        fallback_profile=fallback,
    )

    try:
        engine.start()
    except Exception as e:
        print(f"failed to start: {e}", file=sys.stderr)
        return 1

    mode = " [DRY RUN - no haptics]" if args.dry_run else ""
    print(f"\nRunning{mode}. Profile: {profile.name}. Ctrl+C to stop.")
    print(f"device: {engine.capture.resolved_name}")
    print(f"auto-switch: {'on' if not args.no_auto_switch else 'off'}\n")

    try:
        while True:
            time.sleep(0.5)
            s, k = engine.stats, engine.sink.stats()
            sys.stdout.write(
                f"\r  {engine.profile.name:<18} onsets={s.onsets:<6} pulses={s.pulses:<6} "
                f"sent={k['sent']:<6} rate-drop={k['dropped_rate']:<5} err={k['errors']:<4}"
            )
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nstopping...")
    finally:
        engine.stop()

    s, k = engine.stats, engine.sink.stats()
    print(f"\nonsets detected : {s.onsets}")
    print(f"pulses queued   : {s.pulses}")
    print(f"pulses sent     : {k['sent']}")
    print(f"dropped (rate)  : {k['dropped_rate']}")
    print(f"dropped (queue) : {k['dropped_queue']}")
    print(f"stacking skips  : {s.suppressed_stacking}")
    print(f"profile switches: {s.profile_switches}")
    print(f"errors          : {k['errors']}")
    return 0


# -- entry point --------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="z2haptics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"z2haptics {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("-p", "--profile", help="profile name (default: match foreground app)")
        p.add_argument("-d", "--device", help="loopback device name substring")
        p.add_argument("--samplerate", type=int, default=48000)
        p.add_argument("--profile-dir", help="extra directory to load profiles from")

    sub.add_parser("doctor", help="check the setup end to end").set_defaults(func=cmd_doctor)
    sub.add_parser("enable-api", help="set X1API=true in settings.ini").set_defaults(func=cmd_enable_api)
    sub.add_parser("devices", help="list loopback devices").set_defaults(func=cmd_devices)
    sub.add_parser("test", help="fire test pulses").set_defaults(func=cmd_test)

    p = sub.add_parser("profiles", help="list profiles")
    p.add_argument("--profile-dir", help="extra directory to load profiles from")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("monitor", help="live band levels, no haptics")
    add_common(p)
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("run", help="run the haptic engine")
    add_common(p)
    p.add_argument("--dry-run", action="store_true", help="detect but never drive the motor")
    p.add_argument("--no-auto-switch", action="store_true",
                   help="stay on one profile instead of following the foreground app")
    p.set_defaults(func=cmd_run)

    learn_cli.register(sub)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
