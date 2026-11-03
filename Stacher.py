"""
Stacher.py — YouTube → WAV downloader for guitar tone ML pipeline
=================================================================
Stacher (stacher.io) is a GUI wrapper around yt-dlp + FFmpeg.
This module replicates that workflow programmatically, giving you:
  1. Fetch all available audio streams for a YouTube URL.
  2. Present them to the user for selection.
  3. Download the chosen stream and convert it to a 44.1 kHz / 16-bit WAV
     ready for downstream ML processing.

Dependencies (install once):
    pip install yt-dlp
    # FFmpeg must also be on PATH — grab it from https://ffmpeg.org/download.html
    # or install via: winget install Gyan.FFmpeg
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

# yt-dlp ships its own bundled ffmpeg locator; we can also override the path.
try:
    import yt_dlp
except ImportError:
    sys.exit(
        "[Stacher] yt-dlp is not installed.\n"
        "Run:  pip install yt-dlp"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "downloads"

# WAV sample rate expected by the ML pipeline (change if your model differs)
WAV_SAMPLE_RATE = 44100          # Hz
WAV_BIT_DEPTH   = "pcm_s16le"   # 16-bit signed little-endian PCM


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_valid_youtube_url(url: str) -> bool:
    """Return True if *url* looks like a YouTube video link."""
    patterns = [
        r"(https?://)?(www\.)?(youtube\.com/watch\?.*v=[\w-]+)",
        r"(https?://)?(youtu\.be/[\w-]+)",
        r"(https?://)?(www\.)?(youtube\.com/shorts/[\w-]+)",
    ]
    return any(re.search(p, url) for p in patterns)


def _clean_url(url: str) -> str:
    """
    Isolate just the video ID from a YouTube URL, dropping playlist/radio
    parameters (&list=..., &start_radio=..., etc.) that cause PowerShell
    to choke on unquoted ampersands and that yt-dlp doesn't need.

    Examples
    --------
    https://www.youtube.com/watch?v=abc123&list=RD...  ->  https://www.youtube.com/watch?v=abc123
    https://youtu.be/abc123?si=xyz                     ->  https://www.youtube.com/watch?v=abc123
    """
    # Extract bare video ID from any YouTube URL shape
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url   # fallback: return as-is and let yt-dlp handle it


def _sanitize_filename(name: str) -> str:
    """Strip characters that are illegal in Windows filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fetch_audio_formats(url: str) -> tuple[dict, list[dict]]:
    """
    Query YouTube for all available audio-only streams.

    Parameters
    ----------
    url : str
        Full YouTube video URL.

    Returns
    -------
    info : dict
        Full yt-dlp info dict (title, duration, uploader, …).
    audio_formats : list[dict]
        Filtered list of audio-only format dicts, sorted by bitrate descending.
        Each entry has at minimum:
            format_id, ext, abr (kbps), acodec, filesize (bytes, may be None)
    """
    if not _is_valid_youtube_url(url):
        raise ValueError(f"Not a recognised YouTube URL: {url!r}")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,   # ignore playlist; grab only the single video
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    # Keep only audio-only streams (vcodec == 'none')
    audio_formats = [
        f for f in info.get("formats", [])
        if f.get("vcodec") == "none" and f.get("acodec") != "none"
    ]

    # Sort by ML-quality score (see _format_score), then bitrate as tiebreaker
    audio_formats.sort(
        key=lambda f: (_format_score(f), f.get("abr") or 0),
        reverse=True,
    )

    return info, audio_formats


def _format_score(fmt: dict) -> int:
    """
    Rank an audio format for ML guitar-tone suitability (higher = better).

    Scoring rationale
    -----------------
    - Unknown bitrate (formats 233/234-style mp4 muxed streams): score 0
      These are usually video containers mislabelled as audio-only.
    - HE-AAC (mp4a.40.5 / mp4a.40.29): score 1
      Uses spectral-band replication to fake high frequencies — misleads
      the model about the actual frequency content.
    - Opus (opus): score 2
      Excellent perceptual codec, accurate frequency representation.
    - AAC-LC (mp4a.40.2): score 3
      Highest compatibility and straightforward frequency mapping;
      preferred when bitrate is comparable to Opus alternatives.
    """
    acodec = (fmt.get("acodec") or "").lower()
    abr    = fmt.get("abr")

    if abr is None:
        return 0                         # unknown bitrate — likely a mux stream

    if "40.5" in acodec or "40.29" in acodec:
        return 1                         # HE-AAC — avoid for ML

    if "opus" in acodec:
        return 2                         # Opus — good

    if "mp4a" in acodec or "aac" in acodec:
        return 3                         # AAC-LC — preferred

    return 2                             # anything else: treat like Opus


def pick_format_auto(audio_formats: list[dict]) -> dict:
    """
    Automatically select the best format for ML processing without user input.

    Priority order:
      1. Codec quality score  (AAC-LC > Opus > HE-AAC > unknown)
      2. Bitrate              (higher is better within the same codec tier)

    The list is already sorted by (_format_score, abr) descending from
    fetch_audio_formats(), so index 0 is always the best choice.
    """
    chosen = audio_formats[0]
    abr    = chosen.get("abr", "?")
    codec  = chosen.get("acodec", "?")
    ext    = chosen.get("ext", "?")
    print(
        f"  [Auto] Selected format {chosen['format_id']} — "
        f"{codec} @ {abr} kbps ({ext})"
    )
    return chosen


def display_formats(info: dict, audio_formats: list[dict]) -> None:
    """Pretty-print the available audio streams to stdout."""
    title    = info.get("title", "Unknown Title")
    duration = info.get("duration", 0)
    mins, secs = divmod(int(duration), 60)

    print(f"\n{'='*60}")
    print(f"  Title    : {title}")
    print(f"  Duration : {mins}m {secs:02d}s")
    print(f"  Uploader : {info.get('uploader', 'N/A')}")
    print(f"{'='*60}")
    print(f"\n  {'#':<4} {'Format ID':<12} {'Codec':<10} {'Bitrate':>10}  {'Size':>10}  Ext")
    print(f"  {'-'*56}")

    for idx, fmt in enumerate(audio_formats, start=1):
        abr      = fmt.get("abr")
        bitrate  = f"{abr:.0f} kbps" if abr else "   unknown"
        size     = fmt.get("filesize") or fmt.get("filesize_approx")
        size_str = f"{size / 1_048_576:.1f} MB" if size else "  unknown"
        print(
            f"  {idx:<4} {fmt['format_id']:<12} "
            f"{fmt.get('acodec','?'):<10} {bitrate:>10}  {size_str:>10}  {fmt.get('ext','?')}"
        )
    print()


def pick_format_interactive(audio_formats: list[dict]) -> dict:
    """
    Prompt the user to choose an audio stream from *audio_formats*.
    Returns the selected format dict.
    """
    while True:
        raw = input(f"  Enter number [1-{len(audio_formats)}] "
                    f"(or press Enter for best quality): ").strip()
        if raw == "":
            chosen = audio_formats[0]   # highest bitrate
            print(f"  -> Using best quality: {chosen['format_id']} "
                  f"({chosen.get('abr', '?')} kbps, {chosen.get('ext', '?')})")
            return chosen
        try:
            idx = int(raw)
            if 1 <= idx <= len(audio_formats):
                return audio_formats[idx - 1]
        except ValueError:
            pass
        print(f"  [!] Please enter a number between 1 and {len(audio_formats)}.")


def download_as_wav(
    url: str,
    format_id: str,
    title: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ffmpeg_location: Optional[str] = None,
) -> Path:
    """
    Download a specific yt-dlp format and convert it to a WAV file.

    Parameters
    ----------
    url : str
        YouTube video URL.
    format_id : str
        yt-dlp format ID to download (from audio_formats[i]['format_id']).
    title : str
        Video title used to build the output filename.
    output_dir : Path
        Directory where the WAV file will be saved (created if absent).
    ffmpeg_location : str | None
        Optional explicit path to the ffmpeg binary. Leave None to use PATH.

    Returns
    -------
    Path
        Absolute path to the saved WAV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title      = _sanitize_filename(title)
    output_template = str(output_dir / f"{safe_title}.%(ext)s")

    ydl_opts: dict = {
        "format": format_id,
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # Extra FFmpeg flags: force 44.1 kHz, 16-bit signed PCM, stereo
        "postprocessor_args": {
            "FFmpegExtractAudio": [
                "-ar", str(WAV_SAMPLE_RATE),
                "-acodec", WAV_BIT_DEPTH,
                "-ac", "2",   # stereo — change to "1" for mono
            ]
        },
    }

    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    print(f"\n  Downloading format {format_id!r} and converting to WAV ...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    wav_path = output_dir / f"{safe_title}.wav"
    if not wav_path.exists():
        # yt-dlp might have produced a slightly different filename; search for it
        candidates = list(output_dir.glob(f"{safe_title}*.wav"))
        if candidates:
            wav_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"WAV file not found in {output_dir}. "
                "Ensure FFmpeg is installed and on PATH."
            )

    return wav_path.resolve()


# ---------------------------------------------------------------------------
# High-level convenience function — primary entry point for the ML pipeline
# ---------------------------------------------------------------------------

def youtube_to_wav(
    url: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ffmpeg_location: Optional[str] = None,
    auto_best: bool = False,
) -> Path:
    """
    Full pipeline: fetch formats -> user picks version -> download as WAV.

    Parameters
    ----------
    url : str
        YouTube video URL (the song you want as a WAV).
    output_dir : Path
        Where to save the resulting .wav file.
    ffmpeg_location : str | None
        Optional explicit FFmpeg binary path.
    auto_best : bool
        If True, skip interactive selection and always pick the highest
        bitrate stream automatically (useful for batch / headless runs).

    Returns
    -------
    Path
        Absolute path to the downloaded WAV file.

    Example usage from your ML code
    --------------------------------
    >>> from Stacher import youtube_to_wav
    >>> wav_path = youtube_to_wav("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    >>> # Pass wav_path into your audio loading function, e.g.:
    >>> import librosa
    >>> y, sr = librosa.load(str(wav_path), sr=44100, mono=False)
    """
    # Strip playlist/radio params so bare &-separated URLs work unquoted
    url = _clean_url(url)
    print(f"\n[Stacher] Fetching stream info for:\n  {url}")
    info, audio_formats = fetch_audio_formats(url)

    if not audio_formats:
        raise RuntimeError(
            "No audio-only streams found. "
            "The video may be private, age-restricted, or geo-blocked."
        )

    if auto_best:
        chosen = pick_format_auto(audio_formats)
    else:
        display_formats(info, audio_formats)
        chosen = pick_format_interactive(audio_formats)

    wav_path = download_as_wav(
        url=url,
        format_id=chosen["format_id"],
        title=info.get("title", "audio"),
        output_dir=output_dir,
        ffmpeg_location=ffmpeg_location,
    )

    print(f"\n[Stacher] WAV saved to:\n  {wav_path}\n")
    return wav_path


# ---------------------------------------------------------------------------
# CLI entry-point  (python Stacher.py <YouTube URL> [--best] [--out DIR])
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download a YouTube song as a WAV file for the ML guitar tone pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python Stacher.py https://youtu.be/dQw4w9WgXcQ\n"
            "  python Stacher.py https://youtu.be/dQw4w9WgXcQ --best\n"
            "  python Stacher.py https://youtu.be/dQw4w9WgXcQ --out ./my_songs\n"
            "  python Stacher.py https://youtu.be/dQw4w9WgXcQ --ffmpeg C:\\ffmpeg\\bin\\ffmpeg.exe\n"
        ),
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--best",
        action="store_true",
        help="Skip format selection and automatically pick the highest-bitrate stream.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        metavar="PATH",
        help="Explicit path to the ffmpeg binary (e.g. C:\\ffmpeg\\bin\\ffmpeg.exe)",
    )
    args = parser.parse_args()

    wav_path = youtube_to_wav(
        url=args.url,
        output_dir=Path(args.out),
        ffmpeg_location=args.ffmpeg,
        auto_best=args.best,
    )
    # Print the final path so callers can capture it via shell piping
    print(wav_path)


if __name__ == "__main__":
    _cli()

