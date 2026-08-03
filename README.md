# z2haptics

Turn your Windows default audio output into haptic feedback on a **Swiftpoint Z2**
mouse. Explosions thump, gunfire snaps, lasers tick — driven entirely by what the
game is already playing, with no game-side integration.

Per-game profiles decide which frequencies matter and how each event should feel,
and the app can follow your foreground window to switch profiles automatically.

> **Status:** works, and the full chain is verified on real hardware. It drives an
> **undocumented** Swiftpoint API that a Control Panel update could change. See
> [docs/PROTOCOL.md](docs/PROTOCOL.md).

## How it works

```
WASAPI loopback  ->  FFT band split  ->  per-band onset detection
                                              |
                                     pick one winner per frame
                                              |
                                   pulse shaping (duration, strength)
                                              |
                              rate limiter  ->  X1 API  ->  motor
```

It is **onset-triggered**, not amplitude-following. The engine detects the
*attack* of an event and fires one shaped pulse for it, rather than continuously
tracking loudness. That keeps discrete events feeling discrete, leaves the motor
idle between hits, and matches what a small actuator can actually reproduce.

Strength comes from how **loud** the event is, mapped across a per-band dB
window. (Deriving it from how far the onset overshot its detection threshold
does not work: the adaptive threshold collapses toward zero in quiet passages,
so every event saturates at maximum.)

## Requirements

- Windows 10/11
- Swiftpoint Z2 (Z/Z3 likely work; only the Z2 is verified)
- **Swiftpoint X1 Control Panel running** — the API is a front end to it
- Python 3.11+

```bash
pip install -r requirements.txt
```

## Setup

The Swiftpoint API is **disabled by default**:

```bash
python -m z2haptics.cli enable-api     # takes a backup of settings.ini
```

Then **fully quit the Control Panel from the tray icon** and relaunch it. The
setting is only read at startup, and the app rewrites `settings.ini` on a clean
exit — so a restart that isn't a full quit will silently revert it.

Verify:

```bash
python -m z2haptics.cli doctor
```

```
[1] Swiftpoint X1 Control Panel process ... running
[2] X1API setting ....................... enabled
[3] Command pipe ........................ listening: \\.\pipe\swiftpoint.x1.v2.command
[4] Device response ..................... vibrate accepted
[5] Loopback audio ...................... * Headset Earphone (...)
[6] Profiles ............................ 6 loaded
All checks passed.
```

Feel it working:

```bash
python -m z2haptics.cli test
```

## The tray app

```bash
pip install PySide6
python -m z2haptics.gui
```

Runs in the system tray. Left-click or **Settings…** opens the window; closing it
hides back to the tray rather than quitting.

**Status** — live band meters showing level, gate marker and an onset flash per
hit, plus counters for onsets, pulses sent and rate-limited drops. This is the
fastest way to see whether a gate is set sensibly: a band that never crosses its
gate can never trigger, and one that sits above it constantly will fire on
everything.

**Profile** — full tuning UI. Every band's frequency range, sensitivity, gate,
refractory, pulse duration, strength range, loudness window and priority, plus
master strength and the motor limits. **Edits apply live** to the running engine,
so you can tune while a game plays and feel the change immediately. Save writes
to `~/.z2haptics/profiles/` rather than overwriting the shipped profile.

**Audio** — pick which output to capture. If you use a virtual mixer (Wave Link,
VoiceMeeter), choose the device carrying game audio.

**Test** — strength and duration sliders with a Fire button, plus presets (light
tick, gunshot, explosion, long rumble), a 5-shot burst and a strength ramp. These
bypass the rate limiter, since you asked for that exact pulse.

**General** — auto profile switching, start with Windows, start minimised, and an
X1 API status check with a one-click enable.

```bash
python -m z2haptics.gui --minimised     # what the autostart entry uses
```

## Command line

```bash
python -m z2haptics.cli run                  # match profile to foreground app
python -m z2haptics.cli run -p "Battlefield 6"
python -m z2haptics.cli run --dry-run        # detect, but never drive the motor
python -m z2haptics.cli monitor              # live band levels, no haptics
python -m z2haptics.cli devices
python -m z2haptics.cli profiles
```

`monitor` is the tuning tool. Watch which bands light up on the sounds you care
about, then adjust `sensitivity` and `gate` until the right events — and only the
right events — register.

## Building a profile from real gameplay

Guessing band edges only goes so far: a plasma rifle and a suppressed carbine
live in completely different places. So measure them instead.

Play the game and tap a hotkey each time you hear the event you care about:

```bash
python -m z2haptics.cli learn --name avatar --labels gunshot,laser,explosion
```

```
F9   gunshot
F10  laser
F11  explosion
F7   ambient / background reference
F8   finish and save
```

Hotkeys are polled globally, so they register while the game has focus, and the
game still receives the keypress. Capture is **retrospective** — a rolling buffer
keeps the last few seconds, so tapping the key *after* you hear the event is the
intended way to use it. Grab a handful of `ambient` samples too; they become the
contrast reference.

Then derive the bands:

```bash
python -m z2haptics.cli analyze avatar --write-profile
```

```
=== laser  (4 samples) ===
  spectral peak: 3891Hz at -28.4dB
  contrast vs background:
      1805-3012  Hz |#############                     |   +9.4dB
      3012-5027  Hz |##################################|  +23.3dB
      5027-8389  Hz |####                              |   +3.1dB
  suggested bands:
      3280 - 3940  Hz   contrast +47.0dB   peak -31.2dB
```

That writes a starter profile with measured band edges. Copy it to
`~/.z2haptics/profiles/`, add the game's `.exe` under `processes:`, and tune the
gates against live gameplay with `monitor`.

## Profiles

Shipped: `Default`, `FPS`, `Battlefield 6`, `Icarus`, `Racing`, `Music`.

Loaded from `profiles/` in the repo and `~/.z2haptics/profiles/`. User profiles
shadow shipped ones of the same name, so you can customise without editing the
repo copy.

```yaml
name: FPS
processes: [cs2.exe, r5apex.exe]
strength_scale: 1.0

limits:                  # motor protection
  min_gap_ms: 45         # hard refractory between pulses
  max_pulses_sec: 15     # sliding-window cap
  max_duty: 0.55         # max fraction of time the motor may be driven

bands:
  - name: impact
    low_hz: 20
    high_hz: 90
    sensitivity: 1.5     # flux must exceed this * rolling median
    gate: 0.0035         # absolute floor; below this the band is silent
    refractory_ms: 110   # min spacing between onsets in this band
    min_share: 0.0       # optional: min share of frame flux, rejects leakage
    duration_ms: 95      # motor on-time
    strength_min: 55     # strength at level_floor_db
    strength_max: 100    # strength at level_ceil_db
    level_floor_db: -49.1
    level_ceil_db: -23.1
    priority: 3          # thumb on the scale when bands compete
```

Optional `x1_profile:` also switches the Control Panel's own profile when this
one activates, so button mappings follow the game too.

### Tuning notes

- **Too many pulses?** Raise `sensitivity` and `gate`.
- **Missing events?** Lower them, and confirm the band actually covers the event
  with `analyze`.
- **Everything feels the same?** Widen the gap between `level_floor_db` and
  `level_ceil_db`, or spread `strength_min`/`strength_max`.
- **Wrong band winning?** A sharp transient leaks into neighbouring bands. Raise
  `min_share` on the band being falsely triggered. `priority` deliberately only
  nudges the ranking — it cannot override a much stronger detection, because
  letting it do so meant leakage into `impact` hijacked every gunshot.

## Motor protection

The API accepts ~3000 commands/sec, but the motor is a physical actuator. Three
limits apply, all per-profile: a refractory period, a pulses-per-second cap, and
a duty-cycle ceiling. Audio callbacks never block — pulses go onto a bounded
queue, and when it is full the *weakest* pending pulse is dropped so a loud
transient still lands during a busy passage.

## Development

```bash
python -m pytest tests -q                    # 64 tests
python tools/probe_timing.py                 # measure API latency
python tools/selftest_engine.py --profile FPS  # end-to-end detection check
python tools/make_demo_session.py            # synthetic session for `analyze`
```

`tools/` also holds the probes used to reverse-engineer the API
(`x1_cli.py` is a REPL for it).

## Layout

| Path | |
|---|---|
| `z2haptics/api.py` | X1 API client, pulse queue, rate limiting |
| `z2haptics/audio.py` | WASAPI loopback capture |
| `z2haptics/analysis.py` | FFT band split, onset detection |
| `z2haptics/engine.py` | orchestration, winner selection, pulse shaping |
| `z2haptics/learn.py` | event capture and spectral profiling |
| `z2haptics/profiles.py` | profile loading and saving |
| `z2haptics/gui/` | tray icon, settings window, live meters |
| `docs/PROTOCOL.md` | the reverse-engineered API |

## Licence

MIT
