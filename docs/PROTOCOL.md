# Swiftpoint X1 Control Panel local command API

Reverse-engineered notes on the local API exposed by the Swiftpoint X1 Control
Panel. **This is an undocumented internal interface, not a published SDK.**
Swiftpoint publishes no developer SDK for the Z/Z2/Z3 as of this writing; the
interface described here was found by static analysis of the Control Panel
binary and confirmed against real hardware.

Verified against:

| | |
|---|---|
| Control Panel | 3.0.9.1 |
| Device | Swiftpoint Z2, `VID_214E` / `PID_001E`, firmware `000208` |
| OS | Windows 11 |

Treat all of this as liable to change without notice between Control Panel
versions.

## Enabling the API

The server is **off by default**. It is gated by a single key in

```
%LOCALAPPDATA%\Swiftpoint X1 Control Panel\settings.ini
```

```ini
X1API=false      ; change to true
```

The value is read **once at startup**, and the Control Panel **rewrites
settings.ini on a clean exit**. So the order matters:

1. Fully quit the Control Panel (from the tray icon, not just close the window).
2. Edit `settings.ini`, set `X1API=true`.
3. Relaunch.

Editing while it runs, then letting it exit normally, will silently revert your
change. `z2haptics enable-api` performs the edit and takes a backup, but you
still have to restart the app yourself.

## Transport

A Qt `QLocalServer`, i.e. a Windows named pipe. Two names are used:

| Pipe | Role |
|---|---|
| `\\.\pipe\swiftpoint.x1.v2.command` | current; full command set |
| `\\.\pipe\swiftpoint.x1.profileswitch` | legacy fallback |

Only one binds per run. The v2 name is assembled at runtime from the stored
fragments `swiftpoint.x1` and `.v2.command`, which is why it does not appear as
a single string in the binary. If a stale instance still holds the v2 pipe, a
newly launched instance falls back to the legacy name — so if you see
`profileswitch` rather than `v2.command`, check for a leftover process.

The protocol is **line-oriented UTF-8 text**. Write a command, read the reply.
Commands are case-insensitive. The connection is **reusable** — many commands
may be issued on one open handle, and each returns exactly one response.

Every command replies either `OK`, a value, or `ERROR: <reason>`.

## Command set

```
Profile Set <name>       switch the active Control Panel profile
Profile Get              -> active profile name
Profile List             -> newline-separated profile names
DPI <value>              set DPI (validated against supported steps)
vibrate <duration> <strength>
rgb fixed #RRGGBB
rgb huerotation <brightness> <duration>
rgb override <true|false>
IMU Zero
OLED <Blank|DPI|Profile|Force|Angles|Cube|Firmware|Battery|Message|Image|Override>
OLED message <text>      max 19 characters
OLED image <512 hex chars>
```

Errors observed include `ERROR: No active device connected`,
`ERROR: Invalid command`, `ERROR: Active device does not support vibration`,
`ERROR: Active device does not have an OLED display`, and per-argument
validation failures.

## `vibrate` specifics

```
vibrate <duration_ms> <strength>
```

Internally this maps to a signal typed `X1API_vibration(uint16_t duration, uint8_t strength)`.

**strength** — validated to **0–100 inclusive**. Anything outside, including
negatives and values above 100, returns `ERROR: Invalid vibration strength`.
The Control Panel UI describes strengths above 100% as "overdrive", but that
does *not* apply to the API — 101 is rejected.

**duration** — in milliseconds, and **not range-checked**. Non-integers are
rejected (`ERROR: Invalid vibration duration`), but any integer is accepted and
silently truncated through a `uint16`. `vibrate 65536 50` and `vibrate 100000 50`
both return `OK` and produce a wrapped, much shorter pulse. **Clamp to 0–65535
yourself.**

The parser is lenient about argument count: `vibrate`, `vibrate 100`, and
`vibrate 100 50 50` all return `OK`, applying defaults for whatever is missing.
Do not rely on `OK` as confirmation your arguments were understood.

## Measured performance

Reference machine, Control Panel 3.0.9.1, over one open connection:

| Operation | median | p95 |
|---|---|---|
| `Profile Get` round-trip | 0.35 ms | 1.02 ms |
| `vibrate` round-trip | 1.71 ms | 2.42 ms |
| fire-and-forget write | 0.25 ms | 0.57 ms |

Sustained burst: 50 back-to-back fire-and-forget sends in 16.4 ms — roughly a
**3000 commands/sec** ceiling, with no failures.

**The transport is not the bottleneck.** Any rate limiting you apply should be
sized for the physical actuator — its spin-up time, how distinct you want
consecutive pulses to feel, and motor wear — not for the pipe. `z2haptics`
enforces a refractory period, a pulses-per-second cap and a duty-cycle ceiling
for exactly that reason; see `HapticSink` in [`z2haptics/api.py`](../z2haptics/api.py).

## Minimal client

```python
PIPE = r"\\.\pipe\swiftpoint.x1.v2.command"

with open(PIPE, "r+b", buffering=0) as f:
    f.write(b"vibrate 400 100\n")
    f.flush()
    print(f.read(4096).decode())      # -> 'OK\n'
```

## Caveats

- Undocumented and unsupported. A Control Panel update may rename the pipe,
  change the grammar, or remove the server.
- The Control Panel must be **running**; the API is a front end to it, not a
  direct path to the device.
- Responses must be drained. If you write continuously and never read, the
  server's buffer eventually fills.
- `OK` means "command parsed", not "the motor did what you meant".
