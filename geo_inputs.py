"""Acquire the frozen GEO deposits with streaming copies and SHA-256 validation."""

import csv
import hashlib
import io
import os
import shutil
import tempfile
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen


def geo_records(manifest_text):
    records = []
    for row in csv.DictReader(io.StringIO(manifest_text), delimiter="\t"):
        if row["source"].startswith("https://"):
            records.append(
                (
                    row["path"].removeprefix("inputs/"),
                    row["source"],
                    int(row["bytes"]),
                    row["sha256"],
                )
            )
    return records


def matches_input(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == digest


def fetch_geo_inputs(cache, records, offline=False):
    """Check all GEO files before analysis. A retry reuses intact cache entries."""
    print(
        f"Checking {len(records)} GEO files ({sum(r[2] for r in records) / 2**20:.1f} MiB).",
        flush=True,
    )
    missing = []
    for index, (name, url, size, digest) in enumerate(records, start=1):
        path = cache / name
        if matches_input(path, size, digest):
            print(f"  [{index}/{len(records)}] cached  {name}", flush=True)
            continue
        if offline:
            missing.append(name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, 4):
            print(
                f"  [{index}/{len(records)}] fetching {name} ({size / 2**20:.1f} MiB, attempt {attempt}/3)",
                flush=True,
            )
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, suffix=".part", delete=False
                ) as target:
                    temporary = Path(target.name)
                    with urlopen(url, timeout=120) as response:
                        shutil.copyfileobj(response, target, length=1024 * 1024)
                if not matches_input(temporary, size, digest):
                    raise ValueError("Size or SHA-256 mismatch")
                os.replace(temporary, path)
                break
            except (OSError, ValueError, HTTPException) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                f"Could not obtain {name}. Last error: {last_error}\nSource: {url}\n"
                f"Place the file at {path}, or extract the complete distribution and retry. "
                "Previously verified files can be reused."
            ) from last_error
    if missing:
        raise FileNotFoundError(
            f"Offline cache at {cache} has missing or changed files:\n  "
            + "\n  ".join(missing)
            + "\nUse the inputs/ folder in the complete distribution, or allow a first online run."
        )
    print(f"Verified {len(records)} GEO files.", flush=True)
