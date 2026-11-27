"""
send_knobs.py - talk to the Seed3 overdrive firmware over USB serial, without any ML.

    python send_knobs.py --port COM5 --knobs 0.7 0.4 0.5       # drive tone level, each 0..1
    python send_knobs.py --port COM5 --sweep drive             # step one knob 0 -> 1 so you can hear it
    python send_knobs.py --port COM5 --bypass                  # clean pass-through again
    python send_knobs.py --port COM5 --ping                    # is the firmware alive?
    python send_knobs.py --knobs 0.7 0.4 0.5 --dry-run         # print the line, no hardware

Windows: the port is COMx (Device Manager > Ports). Raspberry Pi: /dev/ttyACM0.
Exit code 0 = the Seed3 answered OK/PONG, 1 = ERR or no answer.
Needs: pip install pyserial
"""
import argparse
import sys
import time


def build_line(drive, tone, level):
    return f"{drive:.4f},{tone:.4f},{level:.4f}\n"


def exchange(ser, line, timeout=1.5):
    """Write one line, return the first reply line that starts with OK / PONG / ERR (or None)."""
    ser.reset_input_buffer()
    ser.write(line.encode("ascii"))
    ser.flush()
    deadline = time.time() + timeout
    buf = b""
    while time.time() < deadline:
        buf += ser.read(256)
        for raw in buf.splitlines():
            text = raw.decode("ascii", errors="replace").strip()
            if text.startswith(("OK", "PONG", "ERR")):
                return text
    return None


def send(port, line, timeout=1.5, baud=115200):
    import serial
    with serial.Serial(port, baud, timeout=0.1) as ser:
        return exchange(ser, line, timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--knobs", type=float, nargs=3, metavar=("DRIVE", "TONE", "LEVEL"))
    g.add_argument("--sweep", choices=["drive", "tone", "level"])
    g.add_argument("--bypass", action="store_true")
    g.add_argument("--ping", action="store_true")
    g.add_argument("--test", action="store_true", help="play the built-in test signal (no input cable needed)")
    g.add_argument("--live", action="store_true", help="go back to the input jack")
    ap.add_argument("--steps", type=int, default=6, help="sweep: number of settings from 0 to 1")
    ap.add_argument("--dwell", type=float, default=2.0, help="sweep: seconds to hold each setting")
    ap.add_argument("--timeout", type=float, default=1.5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.bypass:
        lines = ["bypass\n"]
    elif a.ping:
        lines = ["ping\n"]
    elif a.test:
        lines = ["test\n"]
    elif a.live:
        lines = ["live\n"]
    elif a.knobs:
        lines = [build_line(*a.knobs)]
    else:
        idx = {"drive": 0, "tone": 1, "level": 2}[a.sweep]
        lines = []
        for i in range(a.steps):
            k = [0.5, 0.5, 0.5]
            k[idx] = i / max(a.steps - 1, 1)
            lines.append(build_line(*k))

    if a.dry_run:
        for l in lines:
            print("would send:", l.strip())
        return 0

    import serial
    ok = True
    with serial.Serial(a.port, 115200, timeout=0.1) as ser:
        for n, l in enumerate(lines):
            reply = exchange(ser, l, a.timeout)
            print(f"sent {l.strip():28s} reply: {reply}")
            if reply is None or reply.startswith("ERR"):
                ok = False
                if reply is None:
                    print(f"no reply from {a.port} (firmware running? right port? cable in the Seed's USB?)")
                    break
            if len(lines) > 1 and n < len(lines) - 1:
                time.sleep(a.dwell)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
