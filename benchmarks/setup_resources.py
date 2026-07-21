#!/usr/bin/env python
"""Provision the large benchmark resource files into ``benchmarks/resources/``.

These artifacts are too large to commit, so they are downloaded from their
upstream homes and verified byte-for-byte against the md5 checksums the CIPHER
benchmarks were validated against. Standard library only -- this runs in any of
the benchmark conda envs with no extra installs.

Example usage::

    # fetch everything into benchmarks/resources/ (the default)
    python benchmarks/setup_resources.py

    # fetch into an explicit location (e.g. shared scratch)
    python benchmarks/setup_resources.py --resources-dir /scratch/cipher/resources

    # just the small GEARS essential-gene list
    python benchmarks/setup_resources.py --only essential_all_data_pert_genes.pkl

    # re-download even though the file is already present and valid
    python benchmarks/setup_resources.py --only essential_all_data_pert_genes.pkl --force

Files already present with a matching md5 are skipped, so re-running is cheap and
safe. Exits non-zero if any required file ends up missing or invalid.

Note that ``gene_alias_to_symbol.pkl`` is NOT downloadable: it is a custom
HGNC-derived artifact with no upstream source and is tracked in git under
``benchmarks/resources/``. This script only verifies it.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

#: Streaming chunk size for downloads and md5 hashing.
CHUNK = 1 << 20  # 1 MiB

#: The GenePT release ships as a single zip; the member we want has a stray
#: TRAILING DOT in its archived name, which we strip when writing to disk.
GENEPT_MEMBER_PREFIX = "GenePT_gene_protein_embedding_model_3_text.pickle"

#: first-party scGPT mirror, pinned to a commit so the bytes cannot change.
SCGPT_BASE = (
    "https://huggingface.co/wanglab/scGPT-human/resolve/"
    "a24c237737a40f3720f75abb555489e9fe753be6"
)
#: scGPT loads a *directory*, so its three files must land together here,
#: which is also where ``run_scGPT.py`` looks by default.
SCGPT_SUBDIR = "scGPT/pretrained_models/scGPT_human"

#: Every resource the benchmarks expect in --resources-dir.
#:   source="zip"    -> download an archive, extract one member
#:   source="direct" -> download the file as-is
#:   source="repo"   -> not downloadable, ships in git; verify only
RESOURCES = [
    {
        "name": "GenePT_gene_protein_embedding_model_3_text.pickle",
        "size": 909175027,
        "md5": "c9c5ca556a306743b4e00cb6d543900b",
        "source": "zip",
        "url": "https://zenodo.org/api/records/10833191/files/GenePT_emebdding_v2.zip/content",
        "archive_md5": "3f6ce4317e3a0091978ae5cb8fbf05a3",
        "archive_note": "~574 MB zip (Zenodo record 10833191)",
        "member_prefix": GENEPT_MEMBER_PREFIX,
        "used_by": "GenePert, Scouter, scLAMBDA",
    },
    {
        "name": "essential_all_data_pert_genes.pkl",
        "size": 558811,
        "md5": "b7bc2a91ca513b86f27f090d963711a6",
        "source": "direct",
        "url": "https://dataverse.harvard.edu/api/access/datafile/6934320",
        "used_by": "GEARS",
    },
    {
        "name": "gene_alias_to_symbol.pkl",
        "size": 662884,
        "md5": "a87ac8059017f7ffcc8c7d5042fff4d3",
        "source": "repo",
        "used_by": "GEARS",
    },
    # --- scGPT whole-human pretrained checkpoint -----------------------------
    # Mirrored on the scGPT authors' own Hugging Face org (`wanglab` = WangLab
    # UofT = github.com/bowang-lab). All three files were verified byte-identical
    # to the reference copy. Upstream's README still only links a Google Drive
    # *folder* (not directly fetchable), so this mirror is what makes the
    # checkpoint scriptable with the standard library alone.
    #
    # The URL is pinned to a commit SHA rather than `main`, so a future push to
    # the mirror can never silently swap the weights underneath the md5 gate.
    # Do NOT add an Authorization header or custom User-Agent: the repo is public
    # and ungated, and the /resolve/ URL 302s to a cross-host CDN that plain
    # urllib follows correctly.
    {
        "name": "best_model.pt",
        "size": 205385258,
        "md5": "9922ec94305126e6e4f9c1575cf493ae",
        "source": "direct",
        "url": f"{SCGPT_BASE}/best_model.pt",
        "used_by": "scGPT",
        "dest_root": "models",
        "dest_subdir": SCGPT_SUBDIR,
    },
    {
        "name": "args.json",
        "size": 1300,
        "md5": "07d28cf62c2dac56007b927d31345d45",
        "source": "direct",
        "url": f"{SCGPT_BASE}/args.json",
        "used_by": "scGPT",
        "dest_root": "models",
        "dest_subdir": SCGPT_SUBDIR,
    },
    {
        "name": "vocab.json",
        "size": 1317639,
        "md5": "8efa7f3ca6949e1facdaa47e539e9855",
        "source": "direct",
        "url": f"{SCGPT_BASE}/vocab.json",
        "used_by": "scGPT",
        "dest_root": "models",
        "dest_subdir": SCGPT_SUBDIR,
    },
]


def default_resources_dir() -> Path:
    """The ``resources/`` dir sitting next to this script (no hardcoded paths)."""
    return Path(__file__).resolve().parent / "resources"


def default_models_dir() -> Path:
    """The ``models/`` dir sitting next to this script (no hardcoded paths)."""
    return Path(__file__).resolve().parent / "models"


def human_bytes(n) -> str:
    """Format a byte count for humans."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < step or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= step
    return f"{n:.1f} GB"


def md5_of(path, progress_label=None) -> str:
    """Stream a file through md5, optionally printing progress."""
    total = os.path.getsize(path)
    done = 0
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
            done += len(block)
            if progress_label:
                _progress(progress_label, done, total)
    if progress_label:
        _progress(progress_label, done, total, final=True)
    return h.hexdigest()


#: Last decile reported per label, so non-tty logs get milestones not spam.
_last_decile = {}


def _progress(label, done, total, final=False):
    """Report percent / MB progress.

    On a terminal this overwrites a single line. When stdout is redirected (a
    SLURM log, a pipe) carriage returns are not interpreted, so we instead emit
    one line per 10% to keep logs readable.
    """
    if total and total > 0:
        pct = 100.0 * done / total
        msg = f"    {label}: {pct:5.1f}%  ({human_bytes(done)} / {human_bytes(total)})"
    else:
        pct = None
        msg = f"    {label}: {human_bytes(done)}"

    if sys.stdout.isatty():
        sys.stdout.write("\r" + msg.ljust(72))
        sys.stdout.write("\n" if final else "")
        sys.stdout.flush()
        if final:
            _last_decile.pop(label, None)
        return

    decile = int(pct // 10) if pct is not None else None
    if final:
        # Avoid printing the 100% milestone twice.
        if _last_decile.get(label) != decile:
            print(msg, flush=True)
        _last_decile.pop(label, None)
    elif decile is not None and decile != _last_decile.get(label):
        _last_decile[label] = decile
        print(msg, flush=True)


def download(url, dest, label) -> None:
    """Stream ``url`` to ``dest``, printing progress."""
    req = urllib.request.Request(url, headers={"User-Agent": "cipher-benchmarks/1.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - fixed https URLs above
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as out:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                out.write(block)
                done += len(block)
                _progress(label, done, total)
    _progress(label, done, total, final=True)


def fail(spec, path, got, expected, what="file") -> None:
    """Raise a loud, explanatory error on a checksum mismatch."""
    raise RuntimeError(
        f"MD5 MISMATCH for {what} {path}\n"
        f"    expected: {expected}\n"
        f"    actual:   {got}\n"
        f"The upstream artifact has changed -- this download is NOT byte-identical to\n"
        f"the '{spec['name']}' the CIPHER benchmarks were validated against. Benchmark\n"
        f"numbers produced with it would not be comparable. The bad file has NOT been\n"
        f"installed. Investigate the upstream source before proceeding."
    )


def extract_member(zip_path, spec, dest) -> None:
    """Pull the one member we need out of the archive, tolerating the trailing dot."""
    with zipfile.ZipFile(zip_path) as zf:
        prefix = spec["member_prefix"]
        matches = [
            n for n in zf.namelist()
            if os.path.basename(n).startswith(prefix)
        ]
        if not matches:
            raise RuntimeError(
                f"Could not find a member starting with '{prefix}' in the archive.\n"
                f"Archive contains: {zf.namelist()[:20]}"
            )
        # Prefer an exact-basename hit, else the shortest match (least suffix noise).
        member = min(matches, key=lambda n: (os.path.basename(n) != prefix, len(n)))
        if member != matches[0] or len(matches) > 1:
            print(f"    selected member: {member!r}")
        else:
            print(f"    extracting member: {member!r}")
        # Write WITHOUT the archived name's trailing dot -- dest is already correct.
        with zf.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, CHUNK)


def destination(spec, resources_dir, models_dir) -> Path:
    """Where a resource is installed.

    Most land flat in ``--resources-dir``; the scGPT checkpoint files instead go
    under ``--models-dir`` in the directory layout scGPT expects.
    """
    root = models_dir if spec.get("dest_root") == "models" else resources_dir
    return root / spec.get("dest_subdir", "") / spec["name"]


def provision(spec, resources_dir, models_dir, force) -> str:
    """Ensure one resource is present and valid. Returns a status string."""
    name = spec["name"]
    dest = destination(spec, resources_dir, models_dir)
    print(f"\n=== {name}  [{spec['used_by']}]  -> {dest.parent}")

    if dest.exists() and not force:
        print(f"    present ({human_bytes(dest.stat().st_size)}), verifying md5 ...")
        got = md5_of(dest, progress_label="md5")
        if got == spec["md5"]:
            print("    OK - md5 matches, skipping download.")
            return "ok (cached)"
        if spec["source"] == "repo":
            fail(spec, dest, got, spec["md5"], what="in-repo file")
        print(f"    md5 mismatch ({got}) -- refetching.")
    elif dest.exists() and force:
        print("    --force: refetching even though the file is present.")

    if spec["source"] == "repo":
        raise RuntimeError(
            f"Missing: {dest}\n"
            f"'{name}' is NOT downloadable -- it is a custom HGNC-derived artifact with\n"
            f"no upstream source, so it ships in the repository and is tracked in git at\n"
            f"benchmarks/resources/{name}.\n"
            f"Restore it with:  git checkout -- benchmarks/resources/{name}\n"
            f"(or copy it from another checkout / the shared project directory)."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stage in the destination filesystem so the final move is atomic.
    tmpdir = Path(tempfile.mkdtemp(prefix=".setup_resources.", dir=str(dest.parent)))
    try:
        if spec["source"] == "zip":
            archive = tmpdir / "download.zip"
            print(f"    downloading {spec['archive_note']} ...")
            download(spec["url"], archive, label="download")
            print("    verifying archive md5 ...")
            got = md5_of(archive, progress_label="md5")
            if got != spec["archive_md5"]:
                fail(spec, archive, got, spec["archive_md5"], what="archive")
            print("    OK - archive md5 matches.")
            staged = tmpdir / name
            extract_member(archive, spec, staged)
            archive.unlink()  # reclaim ~574 MB before the final copy
        else:
            staged = tmpdir / name
            print("    downloading ...")
            download(spec["url"], staged, label="download")

        print("    verifying md5 ...")
        got = md5_of(staged, progress_label="md5")
        if got != spec["md5"]:
            fail(spec, staged, got, spec["md5"])

        size = staged.stat().st_size
        if size != spec["size"]:
            raise RuntimeError(
                f"Size mismatch for {name}: expected {spec['size']} bytes, got {size}."
            )

        # Only now does anything appear at the destination.
        os.replace(str(staged), str(dest))
        print(f"    OK - installed {dest} ({human_bytes(size)})")
        return "ok (downloaded)"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv=None) -> int:
    names = [r["name"] for r in RESOURCES]
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Resources:\n  " + "\n  ".join(
            f"{r['name']}  ({human_bytes(r['size'])}, {r['used_by']})"
            f"{' [ships in git, verify only]' if r['source'] == 'repo' else ''}"
            for r in RESOURCES
        ),
    )
    parser.add_argument(
        "--resources-dir", type=Path, default=default_resources_dir(),
        help="where the resource files live (default: %(default)s)",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=default_models_dir(),
        help="where vendored model source lives; the scGPT checkpoint is installed "
             "under <models-dir>/" + SCGPT_SUBDIR + " (default: %(default)s)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-download even if the file is already present and valid",
    )
    parser.add_argument(
        "--only", metavar="NAME", choices=names, action="append",
        help="fetch just this file (repeatable). Choices: " + ", ".join(names),
    )
    args = parser.parse_args(argv)

    selected = [r for r in RESOURCES if not args.only or r["name"] in args.only]
    resources_dir = args.resources_dir.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    print(f"Resources directory: {resources_dir}")
    if any(r.get("dest_root") == "models" for r in selected):
        print(f"Models directory:    {models_dir}")

    results = []
    failed = False
    for spec in selected:
        try:
            results.append(
                (spec["name"], provision(spec, resources_dir, models_dir, args.force)))
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failed = True
            print(f"\nERROR: {exc}", file=sys.stderr)
            results.append((spec["name"], "FAILED"))

    width = max(len(n) for n, _ in results)
    print("\n" + "=" * (width + 20))
    print("SUMMARY")
    print("=" * (width + 20))
    for name, status in results:
        print(f"  {name.ljust(width)}  {status}")
    print("=" * (width + 20))

    if failed:
        print("\nOne or more resources are missing or invalid; see errors above.",
              file=sys.stderr)
        return 1
    print("All selected resources are present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
