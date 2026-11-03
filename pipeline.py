"""
pipeline.py — End-to-end: YouTube URL → guitar stem WAV
========================================================
Chains Stacher.py (YouTube → WAV) → MoisesAPI.py (Moises stem separation).
The final guitar stem WAV is ready to feed into the ML tone model.

Requires:
    pip install yt-dlp musicai-sdk requests
    winget install Gyan.FFmpeg
    $env:MOISES_API_KEY = "your-key"   # from https://music.ai (Moises account)

Usage
-----
    # Interactive (user picks audio quality):
    python pipeline.py "https://www.youtube.com/watch?v=..."

    # Fully automatic (no prompts):
    python pipeline.py "https://www.youtube.com/watch?v=..." --auto

    # From Python:
    from pipeline import run_pipeline
    guitar_wav = run_pipeline(url, auto=True)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Stacher   import youtube_to_wav
from MoisesAPI import separate_guitar

# Output root — all downloads and stems land here
OUTPUT_ROOT = Path(__file__).parent / "pipeline_output"


def run_pipeline(
    url: str,
    auto: bool = False,
    workflow: str = "music-ai/stems-vocals-accompaniment",
    stem_key: str = "guitar",
    api_key: str | None = None,
) -> Path:
    """
    Full pipeline: YouTube URL -> WAV download -> stem separation -> guitar WAV.

    Parameters
    ----------
    url      : YouTube video URL.
    auto     : If True, skip interactive audio-quality prompt (picks best).
    workflow : Music.ai workflow ID for stem separation.
    stem_key : Which stem to treat as the "guitar" output.
    api_key  : Music.ai API key (falls back to MUSICAI_API_KEY env var).

    Returns
    -------
    Path to the isolated guitar stem WAV.
    """

    # ------------------------------------------------------------------ #
    # Step 1 — Download from YouTube as WAV                               #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  STEP 1 — Download from YouTube")
    print("="*60)

    wav_path = youtube_to_wav(
        url=url,
        output_dir=OUTPUT_ROOT / "raw",
        auto_best=auto,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — Separate stems via Moises / Music.ai                       #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  STEP 2 — Stem separation (Moises / Music.ai)")
    print("="*60)

    stems = separate_guitar(
        wav_path=wav_path,
        output_dir=OUTPUT_ROOT / "stems",
        workflow=workflow,
        stem_key=stem_key,
        api_key=api_key,
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Return guitar stem path for downstream ML                  #
    # ------------------------------------------------------------------ #
    guitar_wav = stems.get(stem_key)

    if guitar_wav is None:
        available = list(stems.keys())
        raise KeyError(
            f"Stem '{stem_key}' not in results.\n"
            f"Available stems: {available}\n"
            f"Pass stem_key=<name> matching one of the above."
        )

    print("\n" + "="*60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Guitar stem: {guitar_wav}")
    print("="*60 + "\n")

    return guitar_wav


# ---------------------------------------------------------------------- #
# CLI                                                                     #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube -> WAV -> guitar stem (Stacher + Moises pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py \"https://youtu.be/...\"\n"
            "  python pipeline.py \"https://youtu.be/...\" --auto\n"
            "  python pipeline.py \"https://youtu.be/...\" --auto --stem electric_guitar\n"
        ),
    )
    parser.add_argument("url",  help="YouTube video URL")
    parser.add_argument("--auto",     action="store_true",
                        help="Skip audio-quality prompt; pick best automatically.")
    parser.add_argument("--workflow", default="music-ai/stems-vocals-accompaniment",
                        help="Music.ai workflow ID (default: stems-vocals-accompaniment)")
    parser.add_argument("--stem",     default="guitar",
                        help="Stem key to return as guitar output (default: guitar)")
    parser.add_argument("--apikey",   default=None,
                        help="Music.ai API key (overrides MUSICAI_API_KEY env var)")
    args = parser.parse_args()

    guitar_wav = run_pipeline(
        url=args.url,
        auto=args.auto,
        workflow=args.workflow,
        stem_key=args.stem,
        api_key=args.apikey,
    )
    print(guitar_wav)   # machine-readable final path for shell piping

