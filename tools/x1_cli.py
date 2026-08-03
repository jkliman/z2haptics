r"""
Minimal interactive client for the Swiftpoint X1 Control Panel v2 command API.

Requires settings.ini -> X1API=true and the Control Panel running.
Pipe: \\.\pipe\swiftpoint.x1.v2.command   (legacy fallback: swiftpoint.x1.profileswitch)

Usage:
    python x1_cli.py "Profile Get"
    python x1_cli.py "vibrate 400 100"
    python x1_cli.py            # interactive REPL
"""

import sys
import time

PIPE_V2 = r"\\.\pipe\swiftpoint.x1.v2.command"
PIPE_LEGACY = r"\\.\pipe\swiftpoint.x1.profileswitch"


def send(cmd: str, pipe: str = PIPE_V2, terminator: str = "\n", settle: float = 0.0) -> str:
    """Send one command, return the response text.

    `settle` is an optional pause before reading. It defaults to 0 -- the pipe read
    blocks until the server responds, so a sleep here only inflates measured latency.
    """
    payload = (cmd + terminator).encode("utf-8")
    with open(pipe, "r+b", buffering=0) as f:
        f.write(payload)
        f.flush()
        if settle:
            time.sleep(settle)
        try:
            data = f.read(4096)
        except OSError:
            data = b""
    return data.decode("utf-8", errors="replace").strip()


def send_nowait(cmd: str, pipe: str = PIPE_V2, terminator: str = "\n") -> None:
    """Fire-and-forget: write the command without waiting for a response.

    This is what the haptic engine uses on its hot path -- we do not care about
    the 'OK' and must never block the audio callback waiting for it.
    """
    with open(pipe, "wb", buffering=0) as f:
        f.write((cmd + terminator).encode("utf-8"))
        f.flush()


def main() -> int:
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        print(f">>> {cmd}")
        print(f"<<< {send(cmd)!r}")
        return 0

    print("X1 API REPL. Blank line or 'quit' to exit.")
    while True:
        try:
            cmd = input("x1> ").strip()
        except EOFError:
            break
        if not cmd or cmd.lower() in ("quit", "exit"):
            break
        try:
            print(f"<<< {send(cmd)!r}")
        except OSError as e:
            print(f"!! {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
