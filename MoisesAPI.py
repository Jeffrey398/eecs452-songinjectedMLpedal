"""
MoisesAPI.py — Guitar stem separator via Moises.ai
====================================================
Moises (https://moises.ai) provides AI stem separation through their
developer platform. Their developer API is accessed via the official
Python SDK: musicai-sdk

Developer portal & API key:  https://music.ai  (same account as moises.ai)
SDK docs:                     https://pypi.org/project/musicai-sdk/

Pipeline:
  1. Upload local WAV to Moises' temporary cloud storage
  2. Submit a stem-separation job with a chosen workflow
  3. Wait for the job to complete (SDK handles polling)
  4. Download all stems locally; return paths keyed by stem name

Dependencies:
    pip install musicai-sdk requests

API key setup:
    # PowerShell
    $env:MOISES_API_KEY = "your-key-here"
    # bash/zsh
    export MOISES_API_KEY="your-key-here"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

try:
    from musicai_sdk import MusicAiClient
except ImportError:
    sys.exit(
        "[MoisesAPI] 'musicai-sdk' is not installed.\n"
        "Run: pip install musicai-sdk\n"
        "Get your API key at: https://music.ai"
    )

try:
    import requests
except ImportError:
    sys.exit("[MoisesAPI] 'requests' is not installed.\nRun: pip install requests")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Workflow that produces individual stems: vocals, drums, bass, guitar, piano.
# List all workflows available to your key by calling list_workflows() below.
# Common options:
#   "music-ai/stems-vocals-accompaniment"   — 2 stems: vocals + accompaniment
#   "music-ai/stems-4"                      — 4 stems: vocals, drums, bass, other
#   "music-ai/stems-5"                      — 5 stems: adds piano
#   "music-ai/stems-6"                      — 6 stems: adds guitar
DEFAULT_WORKFLOW = "music-ai/stems-6"


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _get_client(api_key: Optional[str] = None) -> MusicAiClient:
    """Return an authenticated MusicAiClient (Moises developer SDK)."""
    key = api_key or os.environ.get("MOISES_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "[MoisesAPI] No API key found.\n"
            "Set the MOISES_API_KEY environment variable or pass api_key=...\n"
            "Get your key at: https://music.ai"
        )
    return MusicAiClient(api_key=key)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def list_workflows(api_key: Optional[str] = None) -> list[dict]:
    """
    Print and return all Moises workflows available to your API key.
    Run this once to find the right workflow ID for guitar separation.

    Example
    -------
    >>> from MoisesAPI import list_workflows
    >>> list_workflows()
    """
    client = _get_client(api_key)
    workflows = client.get_workflows()
    print(f"\n{'ID':<55} Name")
    print("-" * 80)
    for wf in workflows:
        print(f"  {wf.get('id', '?'):<53} {wf.get('name', '')}")
    print()
    return workflows


def _download_file(url: str, dest: Path) -> Path:
    """Stream a single file from *url* to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    return dest


# ---------------------------------------------------------------------------
# High-level function — primary entry point
# ---------------------------------------------------------------------------

def separate_guitar(
    wav_path: Path,
    output_dir: Optional[Path] = None,
    workflow: str = DEFAULT_WORKFLOW,
    stem_key: str = "guitar",
    api_key: Optional[str] = None,
) -> dict[str, Path]:
    """
    Upload a WAV to Moises, run stem separation, and download all stems.

    Parameters
    ----------
    wav_path : Path
        Local path to the WAV file produced by Stacher.py.
    output_dir : Path | None
        Directory to save the downloaded stems.
        Defaults to a 'stems/<song_name>/' folder next to the WAV.
    workflow : str
        Moises workflow ID. Use list_workflows() to see what your account
        has access to.  Default is "music-ai/stems-6" (6 stems incl. guitar).
    stem_key : str
        Key in the result that contains the guitar stem URL.
        Common values: "guitar", "electric_guitar", "other".
        If the key is wrong the function will print available keys.
    api_key : str | None
        Moises API key.  Falls back to MOISES_API_KEY environment variable.

    Returns
    -------
    dict[str, Path]
        Maps stem name -> local WAV path for every downloaded stem.
        e.g. {"vocals": Path(...), "guitar": Path(...), "drums": Path(...)}

    Example
    -------
    >>> from MoisesAPI import separate_guitar
    >>> from pathlib import Path
    >>> stems = separate_guitar(Path("downloads/Killer Queen.wav"))
    >>> guitar_wav = stems["guitar"]
    """
    wav_path   = Path(wav_path).resolve()
    output_dir = (
        Path(output_dir)
        if output_dir
        else wav_path.parent / "stems" / wav_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client(api_key)

    print(f"\n[MoisesAPI] Starting Moises stem separation")
    print(f"  Source  : {wav_path.name}")
    print(f"  Workflow: {workflow}")

    # ---- Step 1: Upload -------------------------------------------------- #
    print(f"\n  [1/3] Uploading to Moises ... ", end="", flush=True)
    input_url = client.upload_file(str(wav_path))
    print("done.")

    # ---- Step 2: Submit job ----------------------------------------------- #
    print(f"  [2/3] Submitting separation job ...")
    job = client.add_job(
        name=f"guitar-tone-{wav_path.stem}",
        workflow=workflow,
        params={"inputUrl": input_url},
    )
    job_id = job["id"]
    print(f"        Job ID: {job_id}")

    # ---- Step 3: Wait for completion (SDK handles polling) ---------------- #
    print(f"  [3/3] Processing ", end="", flush=True)
    completed = client.wait_for_job_completion(job_id)
    print(" done.")

    status = completed.get("status", "").upper()
    if status != "SUCCEEDED":
        raise RuntimeError(
            f"[MoisesAPI] Job {job_id} ended with status: {status}\n"
            f"Full response: {completed}"
        )

    # ---- Download each stem ---------------------------------------------- #
    result = completed.get("result", {})
    print(f"\n  Downloading stems to: {output_dir}")

    downloaded: dict[str, Path] = {}
    for stem_name, stem_url in result.items():
        if not isinstance(stem_url, str) or not stem_url.startswith("http"):
            continue
        suffix    = Path(stem_url.split("?")[0]).suffix or ".wav"
        dest_path = output_dir / f"{stem_name}{suffix}"
        print(f"    -> {stem_name}{suffix}")
        _download_file(stem_url, dest_path)
        downloaded[stem_name] = dest_path

    # Clean up the remote job to avoid storage charges
    try:
        client.delete_job(job_id)
    except Exception:
        pass  # non-fatal

    print(f"\n[MoisesAPI] {len(downloaded)} stem(s) saved.")

    if stem_key not in downloaded:
        print(
            f"\n  [!] Stem key '{stem_key}' not found in results.\n"
            f"      Available keys: {list(downloaded.keys())}\n"
            f"      Pass stem_key=<name> to separate_guitar() using one of the above.\n"
            f"      Or call list_workflows() to find a workflow with a dedicated guitar stem."
        )

    return downloaded
