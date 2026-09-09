"""Download the processed public Norman dataset linked by theislab/sc-pert.

The known object is 1,703,064,678 bytes. No raw data is redistributed by this script.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://ndownloader.figshare.com/files/34027562"
SIZE = 1703064678
root = Path(__file__).resolve().parent
path = root / "Norman_2019.h5ad"
if path.exists():
    raise SystemExit("Data already exists; verify its manifest before reuse.")
partial = path.with_suffix(".part")
started = time.monotonic()
digest = hashlib.sha256()
md5 = hashlib.md5()
total = 0
last_progress = 0
with urlopen(Request(URL, headers={"User-Agent": "Pertpy-evaluator-reproduction"}), timeout=60) as response:
    if response.status != 200 or int(response.headers.get("Content-Length", SIZE)) != SIZE:
        raise RuntimeError("Unexpected object status or byte size")
    print("Downloading the public processed Norman dataset (1.70 GB).", flush=True)
    with partial.open("xb") as stream:
        while block := response.read(4 * 1024 * 1024):
            total += len(block)
            if total > SIZE:
                raise RuntimeError("Download exceeded the known object size")
            stream.write(block)
            digest.update(block)
            md5.update(block)
            if total - last_progress >= 128 * 1024 * 1024:
                print(f"{total / SIZE:.0%} downloaded", flush=True)
                last_progress = total
    if total != SIZE:
        raise RuntimeError("Incomplete dataset")
    with partial.open("rb") as stream:
        if stream.read(8) != b"\x89HDF\r\n\x1a\n":
            raise RuntimeError("Not an HDF5 file")
    partial.rename(path)
    manifest = {
        "url": URL,
        "provenance": "https://github.com/theislab/sc-pert/blob/main/data_table.csv",
        "bytes": total,
        "sha256": digest.hexdigest(),
        "md5": md5.hexdigest(),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    }
    (root / "norman-download-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
