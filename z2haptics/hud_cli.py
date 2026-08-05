r"""
CLI for HUD-based weapon identification and click-driven haptics.

    z2haptics hud region --name bf6        pick the HUD region on screen
    z2haptics hud learn  --name bf6 --weapons assault,smg,lmg,sniper
    z2haptics hud check  --name bf6        watch what it identifies, live
    z2haptics fire --hud bf6               run click-driven haptics
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .hud import (
    HUD_DIR,
    HudSet,
    Region,
    ScreenCapture,
    build_template,
    is_blank,
    resolve,
    save,
)
from .inputs import InputWatcher


def _load_or_new(name: str) -> HudSet:
    hs = resolve(name)
    return hs if hs is not None else HudSet(name=name)


def cmd_hud_region(args) -> int:
    """Pick the HUD region with a click-drag overlay."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("Region picking needs PySide6: pip install PySide6", file=sys.stderr)
        print("Alternatively pass --rect LEFT,TOP,WIDTH,HEIGHT", file=sys.stderr)
        return 1

    hs = _load_or_new(args.name)

    if args.rect:
        try:
            left, top, width, height = (int(v) for v in args.rect.split(","))
        except ValueError:
            print("--rect wants LEFT,TOP,WIDTH,HEIGHT", file=sys.stderr)
            return 1
        hs.region = Region(left, top, width, height)
    else:
        from .hud_picker import pick_region

        app = QApplication.instance() or QApplication(sys.argv)
        print("Drag a box around the weapon name on the HUD. Esc cancels.")
        print("Have the game visible behind this -- borderless windowed, not")
        print("exclusive fullscreen, or the screen grab comes back black.\n")
        region = pick_region(app)
        if region is None:
            print("cancelled")
            return 1
        hs.region = region

    cap = ScreenCapture()
    try:
        image = cap.grab(hs.region)
    finally:
        cap.close()

    r = hs.region
    print(f"Region: {r.width}x{r.height} at ({r.left}, {r.top})")
    if is_blank(image):
        print("\nWARNING: that region captured as featureless/black.")
        print("That normally means the game is in exclusive fullscreen, which")
        print("screen capture cannot see. Switch it to borderless windowed.")

    path = save(hs, HUD_DIR / f"{args.name}.json")
    print(f"Saved {path}")
    print(f"\nNext: z2haptics hud learn --name {args.name} --weapons assault,smg,lmg,sniper")
    return 0


def cmd_hud_learn(args) -> int:
    """Capture a reference image of the HUD for each weapon."""
    hs = _load_or_new(args.name)
    if hs.region is None:
        print(f"No region set. Run: z2haptics hud region --name {args.name}",
              file=sys.stderr)
        return 1

    weapons = [w.strip() for w in args.weapons.split(",") if w.strip()]
    if not weapons:
        print("give at least one weapon with --weapons", file=sys.stderr)
        return 1

    keys = ["f9", "f10", "f11", "f12", "num1", "num2", "num3", "num4"]
    if len(weapons) > len(keys):
        print(f"at most {len(keys)} weapons", file=sys.stderr)
        return 1

    key_for = dict(zip(keys, weapons))
    captured: dict[str, list] = {w: [] for w in weapons}
    cap = ScreenCapture()
    done = {"stop": False}

    def on_press(key: str) -> None:
        if key == "f8":
            done["stop"] = True
            return
        weapon = key_for.get(key)
        if weapon is None:
            return
        try:
            image = cap.grab(hs.region)
        except Exception as e:
            print(f"\n  capture failed: {e}")
            return
        if is_blank(image):
            print(f"\n  {weapon}: capture was blank -- game in exclusive fullscreen?")
            return
        captured[weapon].append(image)
        print(f"\n  captured {weapon} ({len(captured[weapon])})")

    watcher = InputWatcher(list(key_for) + ["f8"], on_press=on_press, poll_hz=60)
    watcher.start()

    print(f"\nHUD learning for {args.name!r}")
    print(f"  region: {hs.region.width}x{hs.region.height} "
          f"at ({hs.region.left}, {hs.region.top})\n")
    print("  Hold each weapon in game, then press its key:")
    for key, weapon in key_for.items():
        print(f"    {key.upper():<6} {weapon}")
    print("    F8     finish and save\n")
    print("  Two or three captures per weapon is plenty -- the HUD text does not")
    print("  vary the way a gunshot does. Ctrl+C also finishes.\n")

    try:
        while not done["stop"]:
            time.sleep(0.15)
            counts = "  ".join(f"{w}:{len(v)}" for w, v in captured.items())
            sys.stdout.write(f"\r  {counts}          ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        cap.close()

    hs.templates = [build_template(w, imgs) for w, imgs in captured.items() if imgs]
    if not hs.templates:
        print("\n\nNothing captured.", file=sys.stderr)
        return 1

    print(f"\n\nTemplates: {', '.join(f'{t.name} ({t.samples})' for t in hs.templates)}")

    pairs = hs.confusability()
    if pairs:
        print("\nConfusability (1.00 = indistinguishable):")
        for a, b, sim in pairs[:8]:
            flag = "  <- too similar, HUD region may not include the name" if sim > 0.9 else ""
            print(f"  {a:<12} vs {b:<12} {sim:5.2f}{flag}")

    path = save(hs, HUD_DIR / f"{args.name}.json")
    print(f"\nSaved {path}")
    print(f"\nCheck it with:  z2haptics hud check --name {args.name}")
    return 0


def cmd_hud_check(args) -> int:
    """Show what the HUD region identifies, live."""
    hs = resolve(args.name)
    if hs is None:
        print(f"No HUD set {args.name!r}", file=sys.stderr)
        return 1
    if hs.region is None or not hs.templates:
        print("HUD set has no region or no templates yet.", file=sys.stderr)
        return 1

    cap = ScreenCapture()
    print(f"Watching {hs.region.width}x{hs.region.height} at "
          f"({hs.region.left}, {hs.region.top}). Ctrl+C to stop.\n")
    print("Swap weapons in game and watch the reading change.\n")

    blanks = 0
    try:
        while True:
            image = cap.grab(hs.region)
            if is_blank(image):
                blanks += 1
                sys.stdout.write(f"\r  BLANK CAPTURE ({blanks})  -- exclusive fullscreen?   ")
            else:
                match, score = hs.identify(image)
                name = match.name if match else "(unrecognised)"
                sys.stdout.write(f"\r  {name:<20} confidence {score:5.2f}          ")
            sys.stdout.flush()
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n")
    finally:
        cap.close()
    return 0


def cmd_fire(args) -> int:
    """Run click-driven haptics."""
    from .weapon_engine import WeaponEngine

    hs = resolve(args.hud) if args.hud else None
    if args.hud and hs is None:
        print(f"No HUD set {args.hud!r}. Set one up with `z2haptics hud region`.",
              file=sys.stderr)
        return 1

    processes = [p.strip() for p in (args.processes or "").split(",") if p.strip()]

    engine = WeaponEngine(
        hud_set=hs,
        processes=processes,
        strength_scale=args.strength_scale,
        dry_run=args.dry_run,
    )

    if args.weapon:
        engine.set_weapon(args.weapon)

    try:
        engine.start()
    except Exception as e:
        print(f"failed to start: {e}", file=sys.stderr)
        return 1

    mode = " [DRY RUN]" if args.dry_run else ""
    print(f"\nClick-driven haptics running{mode}. Ctrl+C to stop.")
    print(f"  HUD set   : {hs.name if hs else '(none - fixed weapon)'}")
    print(f"  weapon    : {args.weapon or 'from HUD'}")
    print(f"  active in : {', '.join(processes) if processes else 'any window'}")
    print("\n  Left click fires. R reloads.\n")

    try:
        while True:
            time.sleep(0.25)
            s = engine.status()
            sys.stdout.write(
                f"\r  {str(s['weapon']):<12} ammo {s['ammo']:>3}/{s['magazine']:<3} "
                f"shots {s['shots_fired']:<5} reloads {s['reloads']:<3} "
                f"hud {s['hud_score']:.2f}   "
            )
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nstopping...")
    finally:
        engine.stop()

    s = engine.status()
    print(f"\nshots fired : {s['shots_fired']}")
    print(f"reloads     : {s['reloads']}")
    print(f"dry clicks  : {s['dry_clicks']}")
    print(f"weapon swaps: {s['swaps']}")
    print(f"sink        : {s['sink']}")
    return 0


def register(sub) -> None:
    hud = sub.add_parser("hud", help="HUD-based weapon identification")
    hud_sub = hud.add_subparsers(dest="hud_command", required=True)

    p = hud_sub.add_parser("region", help="pick the HUD region on screen")
    p.add_argument("--name", required=True)
    p.add_argument("--rect", help="set explicitly as LEFT,TOP,WIDTH,HEIGHT")
    p.set_defaults(func=cmd_hud_region)

    p = hud_sub.add_parser("learn", help="capture a HUD reference per weapon")
    p.add_argument("--name", required=True)
    p.add_argument("--weapons", required=True, help="comma-separated weapon names")
    p.set_defaults(func=cmd_hud_learn)

    p = hud_sub.add_parser("check", help="show live what the HUD identifies")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_hud_check)

    p = sub.add_parser("fire", help="click-driven haptics (HUD weapon + mouse trigger)")
    p.add_argument("--hud", help="HUD set to identify the weapon with")
    p.add_argument("--weapon", help="fix the weapon instead of reading the HUD")
    p.add_argument("--processes", default="bf6.exe,bf.exe",
                   help="only fire while one of these has focus")
    p.add_argument("--strength-scale", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_fire)
