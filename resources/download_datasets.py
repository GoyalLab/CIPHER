"""Download CIPHER source datasets into the data directory, with md5 verification.

Large ``.h5ad`` datasets are not stored in git; this script fetches them from their
canonical hosts (figshare / Zenodo / NCBI) and checks each against a known md5 so a
changed upstream fails loudly instead of silently substituting different data.

Standard library only (``urllib`` + ``hashlib``), so it runs in any of the project
environments with no extra installs.

Examples
--------
    # everything, into the default data dir
    python resources/download_datasets.py

    # just sci-Plex 3, into a chosen directory
    python resources/download_datasets.py --only SrivatsanTrapnell2020_sciplex3 \
        --data-dir /path/to/cipher_data

``--data-dir`` defaults to ``$CIPHER_DATA_DIR`` (or the current directory if unset);
point it at the shared data dir used by ``.pull_data.sh``. No path is hardcoded.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB

#: default destination; overridable by --data-dir or $CIPHER_DATA_DIR (never hardcoded downstream)
DEFAULT_DATA_DIR = os.environ.get("CIPHER_DATA_DIR", "./")

# Each entry: the on-disk basename (matches resources/datasets.csv `file`), its exact
# size + md5 (verified), and a direct download URL.
DATASETS = [
    {
        "name": "SrivatsanTrapnell2020_sciplex3.h5ad",
        "size": 2213070032,
        "md5": "9b0155a44f12c2b60b018ea5afb19267",
        "url": "https://ndownloader.figshare.com/files/39324305",
        "source": "figshare 10.6084/m9.figshare.22122701 (biolord curation of sci-Plex 3, "
                  "Srivatsan et al. 2020)",
        # sci-Plex is a chemical screen: load with
        #   cipher.load_dataset(path, pert_key="product_name", require_target_in_var=False)
        "note": "chemical screen; use pert_key='product_name', require_target_in_var=False",
    },
]


def human_bytes(n) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def _progress(label, done, total, final=False):
    pct = 100.0 * done / total if total else 0.0
    end = "\n" if final else "\r"
    sys.stdout.write(f"    {label}: {pct:5.1f}%  ({human_bytes(done)} / {human_bytes(total)})   {end}")
    sys.stdout.flush()


def md5_of(path, progress_label=None) -> str:
    total = Path(path).stat().st_size
    h = hashlib.md5()
    done = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            done += len(block)
            if progress_label:
                _progress(progress_label, done, total)
    if progress_label:
        _progress(progress_label, done, total, final=True)
    return h.hexdigest()


def download(url, dest, label="download") -> None:
    # public, ungated hosts; urllib follows the cross-host 302 to the CDN/S3 automatically.
    req = urllib.request.Request(url, headers={"User-Agent": "cipher-download/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            _progress(label, done, total)
        _progress(label, done, total, final=True)


def provision(spec, data_dir, force) -> str:
    name = spec["name"]
    dest = data_dir / name
    print(f"\n=== {name}")
    print(f"    source: {spec['source']}")

    if dest.exists() and not force:
        print(f"    present ({human_bytes(dest.stat().st_size)}), verifying md5 ...")
        got = md5_of(dest, progress_label="md5")
        if got == spec["md5"]:
            print("    OK - md5 matches, skipping download.")
            return "ok (cached)"
        print(f"    md5 mismatch ({got}) -- refetching.")
    elif dest.exists() and force:
        print("    --force: refetching even though the file is present.")

    data_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix=".download_datasets.", dir=str(data_dir)))
    try:
        staged = tmpdir / name
        print(f"    downloading {human_bytes(spec['size'])} ...")
        download(spec["url"], staged)

        print("    verifying md5 ...")
        got = md5_of(staged, progress_label="md5")
        if got != spec["md5"]:
            raise RuntimeError(
                f"md5 mismatch for {name}:\n  expected {spec['md5']}\n  got      {got}\n"
                f"The upstream file at {spec['url']} changed and is NOT byte-identical to "
                "what CIPHER was validated against."
            )
        size = staged.stat().st_size
        if size != spec["size"]:
            raise RuntimeError(f"size mismatch for {name}: expected {spec['size']}, got {size}.")

        os.replace(str(staged), str(dest))
        print(f"    OK - installed {dest} ({human_bytes(size)})")
        if spec.get("note"):
            print(f"    note: {spec['note']}")
        return "ok (downloaded)"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv=None) -> int:
    names = [d["name"].replace(".h5ad", "") for d in DATASETS]
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Datasets:\n  " + "\n  ".join(
            f"{d['name']}  ({human_bytes(d['size'])})" for d in DATASETS),
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR),
                        help="destination directory (default: %(default)s; "
                             "or set $CIPHER_DATA_DIR)")
    parser.add_argument("--only", metavar="NAME", action="append", choices=names,
                        help="fetch just this dataset (repeatable). Choices: " + ", ".join(names))
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file is present and valid")
    args = parser.parse_args(argv)

    wanted = set(args.only) if args.only else None
    selected = [d for d in DATASETS if wanted is None or d["name"].replace(".h5ad", "") in wanted]
    data_dir = args.data_dir.expanduser().resolve()
    print(f"Data directory: {data_dir}")

    results, failed = [], False
    for spec in selected:
        try:
            results.append((spec["name"], provision(spec, data_dir, args.force)))
        except Exception as exc:  # report and continue
            failed = True
            print(f"\nERROR: {exc}", file=sys.stderr)
            results.append((spec["name"], "FAILED"))

    width = max((len(n) for n, _ in results), default=10)
    print("\n" + "=" * (width + 20))
    print("SUMMARY")
    print("=" * (width + 20))
    for name, status in results:
        print(f"  {name:<{width}}  {status}")
    print("=" * (width + 20))
    if failed:
        print("Some datasets failed; see errors above.")
        return 1
    print("All selected datasets are present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
