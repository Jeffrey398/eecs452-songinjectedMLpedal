# Seed3 overdrive: bridge from the Random Forest to the amp

    song / guitar stem --(Pi: features + Random Forest)--> drive, tone, level --USB--> Seed3 --> amp
                                                                                     (Wiener-Hammerstein overdrive)

| File | Runs on | What it does |
|---|---|---|
| `overdrive_dsp.h` | Seed3 | the Wiener-Hammerstein effect: input filter, static nonlinearity, output filter, level |
| `knob_protocol.h` | Seed3 | USB line parser (`drive,tone,level`), commands, hand-off to the audio callback |
| `main.cpp`, `Makefile` | Seed3 | firmware. Starts as a clean pass-through; the effect engages on the first valid line |
| `test_source.h` | Seed3 | built-in plucked test signal for demos with no input cable |
| `send_knobs.py` | Pi / laptop | send knobs, sweep a knob, bypass, ping (no ML needed) |
| `run_song.py` | Pi / laptop | stem or YouTube URL -> predicted knobs -> Seed3. Goes in the repo root |
| `tests/` | laptop | `python tests/verify_overdrive_dsp.py` checks the C++ against a Python reference |

## Protocol (Pi -> Seed3, one line each)
    0.7500,0.4000,0.5000     drive,tone,level, each 0..1 (two numbers also accepted: level unchanged)
    bypass                   clean pass-through
    ping                     -> PONG
    test                     play a built-in guitar-like signal (no input cable needed)
    live                     back to the input jack
Replies: `OK 750,400,500` (accepted values x1000), `OK bypass`, `PONG`, or `ERR <reason>`.
This matches what `predictOverdrive.py` / `trainOverdrive.py` produce (drive, tone, level).

## PLACEHOLDER: the Wiener-Hammerstein numbers
The structure and plumbing are final, but the values in `MapKnobs()` (overdrive_dsp.h, mirrored in
`tests/reference_overdrive.py`) are NOT fitted to the real pedal. They make a plausible overdrive that
responds to all three knobs, so the whole chain can be demonstrated. To use the fitted model: replace
`MapKnobs()` with the identified filter corners / nonlinearity / gains (a formula, or interpolation over the
5x5 drive/tone grid), update the Python reference the same way, and re-run `verify_overdrive_dsp.py`.

## Order of testing
1. `python tests/verify_overdrive_dsp.py`  (laptop, expect ALL PASS)
2. Build and flash (same toolchain steps as before; target file is `build/seed3_overdrive.bin`)
3. Guitar or laptop audio in, amp out: clean pass-through (nothing sent yet)
4. `python send_knobs.py --port COM5 --ping`, then `--knobs 0.7 0.4 0.5`, then `--sweep drive`, then `--bypass`
5. `python run_song.py --stem <guitar stem>.wav --port COM5`
6. Same with `--url "<youtube link>"` for the full pipeline

## Things to confirm on the board
- `kInputChannel` (0 = left jack) in `main.cpp`.
- Block size 48 at 48 kHz.
- Pots are not read; knobs come only from USB.
