"""
Verify WASAPI loopback capture of the default output device.

PortAudio (sounddevice) in this build does not expose loopback devices, so we use
`soundcard`, which opens a WASAPI loopback capture client against the default
speaker directly -- no virtual cable or Stereo Mix required.

Play some audio before running this, or the levels will read silent.
"""

import sys
import time

import numpy as np
import soundcard as sc

SAMPLERATE = 48000
BLOCK = 2048


def main() -> int:
    print("=== speakers ===")
    for s in sc.all_speakers():
        print(f"  {s.name}")

    spk = sc.default_speaker()
    print(f"\n=== default speaker ===\n  {spk.name}")

    try:
        mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
    except Exception as e:
        print(f"!! could not open loopback for default speaker: {type(e).__name__}: {e}")
        print("   falling back to first loopback-capable device")
        loopbacks = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
        if not loopbacks:
            print("!! no loopback devices available")
            return 1
        mic = loopbacks[0]

    print(f"\n=== capturing 5s of loopback @ {SAMPLERATE}Hz ===")
    print(f"  device: {mic.name}  (isloopback={mic.isloopback})")

    # Self-test: play a bass sweep out the default speaker so the capture has
    # something to see even if nothing else is playing.
    tone_thread = None
    if "--tone" in sys.argv:
        import threading

        def play_tone():
            t = np.linspace(0, 4, SAMPLERATE * 4, endpoint=False)
            freq = np.linspace(40, 400, t.size)          # sweep through the bass band
            wave = 0.25 * np.sin(2 * np.pi * np.cumsum(freq) / SAMPLERATE)
            spk.play(np.column_stack([wave, wave]), samplerate=SAMPLERATE)

        tone_thread = threading.Thread(target=play_tone, daemon=True)
        tone_thread.start()
        print("  playing a 40->400Hz test sweep\n")
    else:
        print("  (play some audio now, or re-run with --tone)\n")

    peaks = []
    with mic.recorder(samplerate=SAMPLERATE, channels=2, blocksize=BLOCK) as rec:
        t_end = time.time() + 5
        while time.time() < t_end:
            data = rec.record(numframes=BLOCK)
            mono = data.mean(axis=1) if data.ndim > 1 else data
            rms = float(np.sqrt(np.mean(mono ** 2)))
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            peaks.append(peak)
            bar = "#" * int(min(rms * 300, 50))
            print(f"  rms={rms:8.5f} peak={peak:8.5f} |{bar}")

    if peaks and max(peaks) > 1e-5:
        print(f"\nLOOPBACK OK -- captured signal, max peak {max(peaks):.5f}")
        return 0
    print("\nLoopback opened but captured silence. Was anything playing?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
