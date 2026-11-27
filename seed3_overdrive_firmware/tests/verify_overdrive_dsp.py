"""
verify_overdrive_dsp.py - check that the firmware's C++ overdrive matches the Python reference.
Builds tests/host_test.cpp with g++ (the same headers that go on the Seed3), runs the protocol and
unit tests, then runs a test signal through C++ and Python for several knob settings.

    python verify_overdrive_dsp.py        (needs g++, numpy, scipy)
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference_overdrive as ref  # noqa: E402

SR = 48000
CASES = {  # name: (drive, tone, level)
    "clean-ish":   (0.0, 0.5, 0.5),
    "mid":         (0.5, 0.5, 0.5),
    "dark":        (0.5, 0.0, 0.5),
    "bright":      (0.5, 1.0, 0.5),
    "high drive":  (1.0, 0.5, 0.5),
    "quiet":       (0.75, 0.25, 0.0),
    "loud":        (0.75, 0.75, 1.0),
}


def main():
    tmp = Path(tempfile.mkdtemp(prefix="od_verify_")); exe = tmp / "host_test"
    print("building host_test ...")
    subprocess.run(["g++", "-O2", "-std=c++14", "-Wall", "-Wextra", "-o", str(exe), str(HERE / "host_test.cpp")], check=True)
    print("\n== protocol and unit tests ==")
    if subprocess.run([str(exe), "protocol"]).returncode != 0:
        sys.exit(1)
    print("\n== C++ vs Python reference ==")
    x = ref.test_signal(SR).astype(np.float32); xin = tmp / "in.f32"; x.tofile(xin)
    ok = True
    print(f"{'case':12s} {'null residual':>14s} {'max |error|':>12s}  result")
    for name, (d, t, l) in CASES.items():
        out = tmp / "out.f32"
        subprocess.run([str(exe), "dsp", str(xin), str(out), str(SR), str(d), str(t), str(l)], check=True)
        y_cpp = np.fromfile(out, dtype=np.float32).astype(np.float64)
        y_ref = ref.overdrive(x.astype(np.float64), SR, d, t, l)
        err = y_cpp - y_ref
        null_db = 10 * np.log10((np.sum(err ** 2) + 1e-20) / (np.sum(y_ref ** 2) + 1e-20))
        maxerr = float(np.max(np.abs(err)))
        good = null_db < -50 and maxerr < 1e-2
        ok &= good
        print(f"{name:12s} {null_db:11.1f} dB {maxerr:12.2e}  {'PASS' if good else 'FAIL'}")
    print("\nALL PASS" if ok else "\nMISMATCH between C++ and Python")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
