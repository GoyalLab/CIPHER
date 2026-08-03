#!/usr/bin/env python
"""Download everything the CIPHER notebooks need into ``$CIPHER_DATA_DIR``.

    export CIPHER_DATA_DIR=/path/to/cipher_data
    python resources/download_data.py

One command, one destination. Replaces the old ``download_resources.py`` (one file) and
``download_suppl.py`` (a GEO tar). Fetches all three groups of inputs:

* **main**   -- 22 curated Perturb-seq objects, Zenodo 10.5281/zenodo.21729034
* **suppl**  -- 12 supplementary inputs, Zenodo 10.5281/zenodo.21728754
* **public** -- 11 datasets that live at their original homes (scPerturb, GEO) and are
  deliberately not re-hosted

``resources/zenodo_manifest.csv`` is the source of truth: it supplies every file's md5, byte
size and destination path relative to ``CIPHER_DATA_DIR``. Zenodo's filename namespace is flat,
so the manifest is also what restores nesting (``Xtot_naive_resistant_melanoma_unbalanced.h5ad``
belongs in ``suppl/GSE233766/``).

Everything is verified against that md5. Downloads resume after an interruption (Zenodo and
NCBI both honour HTTP Range), already-correct files are skipped, and a partial or corrupted
file is never left where a notebook could read it.

Standard library only, so it runs in any project environment with no installs.

Examples
--------
    python resources/download_data.py                      # everything (~70 GB)
    python resources/download_data.py --record suppl       # just the supplement
    python resources/download_data.py --only Marson2025    # substring match
    python resources/download_data.py --check              # verify what is present, download nothing
    python resources/download_data.py --jobs 8             # more parallelism
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "resources" / "zenodo_manifest.csv"
CHUNK = 1 << 20
RETRIES = 4
UA = {"User-Agent": "cipher-download/2.0"}

#: The two published CIPHER records. File URLs are resolved from the record API at runtime so
#: this keeps working if Zenodo changes its URL scheme, and so upstream checksums get
#: cross-checked against the manifest before anything is transferred.
ZENODO_RECORDS = {"main": 21729034, "suppl": 21728754}

#: Public Zenodo collections holding datasets CIPHER uses but does not re-host. Resolved
#: through the API at runtime and cross-checked against the manifest md5 before any bytes move.
#: NOTE both ids are PINNED VERSION records, not the concept record (7041849's conceptrecid is
#: 7041848). That is deliberate: a pinned version cannot float to different bytes later, which
#: is what keeps these md5s stable. Do not "fix" these to the concept id.
PUBLIC_ZENODO = {
    "scPerturb v1.0": 7041849,
    "scPerturb v1.4 (sci-Plex 3)": 13350497,
}

#: Sources with no resolvable API. GEO publishes no checksums at all, so these md5s were
#: established by downloading the files in full and hashing them here.
#:
#: These two are the one case where upstream is NOT byte-identical to what the CIPHER authors
#: worked from: GEO serves a dense, uncompressed X (5.6 / 9.4 GB) whereas the lab copies are
#: the same matrix re-saved as gzipped CSR (1.3 / 2.0 GB), plus a convenience `perturbation`
#: obs column that is `gene` with 'non-targeting' relabelled 'control'. That column is NOT
#: required: cipher.data.PERT_KEY_CANDIDATES falls back to 'gene' and CONTROL_PATTERNS already
#: recognises 'non-targeting', so load_dataset() handles the GEO layout natively.
#: Consequence: these are verified against their own GEO md5 below, never against the
#: zenodo_manifest.csv md5, which describes the differently-compressed local copy.
_GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl"
PUBLIC_STATIC: dict[str, dict] = {
    "GSE264667_hepg2_raw_singlecell_01.h5ad": {
        "url": f"{_GEO}/GSE264667_hepg2_raw_singlecell_01.h5ad",
        "note": "GEO GSE264667 (dense original; load_dataset auto-detects obs['gene'])",
        "override_size": 5614460941,
        "override_md5": "dfc676e4186b8c8c173ff14e1194cc8a",
    },
    "GSE264667_jurkat_raw_singlecell_01.h5ad": {
        "url": f"{_GEO}/GSE264667_jurkat_raw_singlecell_01.h5ad",
        "note": "GEO GSE264667 (dense original; load_dataset auto-detects obs['gene'])",
        "override_size": 9366490264,
        "override_md5": "dae9fab2a8b0430099a76ce60d0c5405",
    },
}
# Equivalence of the GEO originals to the lab copies was established by full download:
#   hepg2  145473 x 9624  sampled rows identical=True maxabsdiff=0
#   jurkat 262956 x 8882  sampled rows identical=True maxabsdiff=0
# In both, obs differs only by the derived 'perturbation' column, which load_dataset does not
# need. See resources/ZENODO_UPLOAD.md section 5.

#: Datasets CIPHER does not re-host. Filled in by resolve_public_sources(); each value is
#: {"url": ..., "note": ...} and may carry "gunzip": True when the host serves a .gz.
PUBLIC_SOURCES: dict[str, dict] = {}

_print_lock = threading.Lock()


def log(msg=""):
    with _print_lock:
        print(msg, flush=True)


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def md5_of(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- sources
def _api(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def resolve_zenodo(record_id):
    """basename -> (url, size, md5) for every file in a published record."""
    data = _api(f"https://zenodo.org/api/records/{record_id}")
    out = {}
    for f in data.get("files", []):
        out[f["key"]] = (
            f"https://zenodo.org/api/records/{record_id}/files/{f['key']}/content",
            int(f.get("size", -1)),
            str(f.get("checksum", "")).replace("md5:", ""),
        )
    return out


def resolve_public_sources():
    """Collect every candidate source for the datasets CIPHER does not re-host.

    A filename can appear in several scPerturb versions with DIFFERENT bytes -- v1.0 (7041849)
    ships a 2,456,030,368-byte SrivatsanTrapnell2020_sciplex3.h5ad while v1.4 (13350497) ships
    a 2,526,631,614-byte one, and only the latter is what CIPHER was validated against. So
    every candidate is kept and the choice is made by checksum in build_plan(), never by which
    record happened to be consulted first.
    """
    for label, record in PUBLIC_ZENODO.items():
        try:
            files = resolve_zenodo(record)
        except Exception as e:
            log(f"WARNING: could not reach Zenodo record {record} ({label}): {e}")
            continue
        for key, (url, size, md5) in files.items():
            PUBLIC_SOURCES.setdefault(key, []).append(
                {"url": url, "md5": md5, "size": size, "note": f"{label} (Zenodo {record})"})
    for key, spec in PUBLIC_STATIC.items():
        PUBLIC_SOURCES.setdefault(key, []).append(spec)


#: Caches deliberately kept out of the deposit because they rebuild from the base objects.
#: Commands verified against each generator's argparse block -- note --out-root differs per
#: script and is NOT simply "$CIPHER_DATA_DIR/suppl" for all of them.
PRECOMPUTES = [
    ("suppl/precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED/  (~23 GB)",
     "fig3_atlas, fig4_cross_dataset E/F, figS7, figS9A, figS9B, figS19",
     "python notebooks/src/gen_fullcov_scores.py --data-dir $CIPHER_DATA_DIR \\\n"
     "    --out-root $CIPHER_DATA_DIR/suppl/precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED \\\n"
     "    --mode raw --thresholds 1.0 0.1"),
    ("suppl/precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0/  (~12 GB)",
     "figS16_forward, figS13D, figS14",
     "python notebooks/src/gen_forward_precompute.py --data-dir $CIPHER_DATA_DIR \\\n"
     "    --out-root $CIPHER_DATA_DIR/suppl/"
     "precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0"),
    ("suppl/figS15/response_breadth_per_perturbation.tsv  (~4.7 GB of intermediates)",
     "figS15_effective_N",
     "python notebooks/src/gen_breadth_table.py --data-dir $CIPHER_DATA_DIR \\\n"
     "    --out-root $CIPHER_DATA_DIR/suppl --mode raw"),
    ("suppl/posterior_inverse_fast_from_prerun_fullH_diag/  (~4 MB)",
     "figS9_generanking",
     "python notebooks/src/gen_inverse_summary.py --data-dir $CIPHER_DATA_DIR \\\n"
     "    --out-root $CIPHER_DATA_DIR/suppl --normalization raw"),
]


def print_precompute_note():
    """The deposit deliberately omits ~40 GB of regenerable caches. Say so explicitly, with
    working commands: a notebook failing later on a missing .npy is a miserable way to find
    out, and the equivalent section of ZENODO_UPLOAD.md had the flags wrong."""
    log("\n" + "=" * 78)
    log("NOT DOWNLOADED - regenerable caches (~40 GB), rebuilt from what you now have")
    log("=" * 78)
    log("These are not in either Zenodo record by design. Run the generators before the")
    log("notebooks that read them; each is CPU-only but heavy, so use a batch job, not a")
    log("login node. Run from the repo root with the package importable (pip install -e .).\n")
    for path, consumers, cmd in PRECOMPUTES:
        log(f"  {path}")
        log(f"      needed by: {consumers}")
        for ln in cmd.split("\n"):
            log(f"      {ln}")
        log("")
    log("Note --out-root differs per generator: gen_fullcov_scores and gen_forward_precompute")
    log("take the FULL precompute directory, while gen_breadth_table and gen_inverse_summary")
    log("take $CIPHER_DATA_DIR/suppl and append their own subdirectory. figS7 additionally")
    log("needs the 0.1 threshold, which is why --thresholds passes both 1.0 and 0.1.")


# --------------------------------------------------------------------------- transfer
def _open_ranged(url, offset):
    """GET url, resuming from offset. Returns (response, resuming) where resuming says whether
    the server honoured the Range (206) or restarted the whole file (200)."""
    headers = dict(UA)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=120)
    return resp, (resp.status == 206 if offset else False)


def fetch(url, staged: Path, expect_size, label):
    """Download to `staged`, resuming an existing partial file. Retries with backoff.

    Returns the number of bytes actually transferred, which is less than the file size when a
    partial download was resumed."""
    moved = 0
    for attempt in range(1, RETRIES + 1):
        offset = staged.stat().st_size if staged.exists() else 0
        if expect_size and offset == expect_size:
            return moved  # fully transferred already; caller verifies the md5
        if expect_size and offset > expect_size:
            staged.unlink()  # longer than it should be: start over
            offset = 0
        try:
            resp, resuming = _open_ranged(url, offset)
            mode = "ab" if resuming else "wb"
            if offset and not resuming:
                offset = 0  # server ignored Range; rewrite from the start
            if resuming:
                log(f"  [{label}] resuming from {human(offset)}")
            with resp, open(staged, mode) as out:
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    moved += len(block)
            if not expect_size or staged.stat().st_size == expect_size:
                return moved
            raise IOError(f"short read: {staged.stat().st_size} of {expect_size} bytes")
        except (urllib.error.URLError, urllib.error.HTTPError, IOError, TimeoutError) as e:
            if attempt == RETRIES:
                raise RuntimeError(f"{label}: gave up after {attempt} attempts - {e}") from e
            wait = 5 * 2 ** (attempt - 1)
            log(f"  [{label}] attempt {attempt} failed ({e}); retrying in {wait}s "
                f"(resuming from {human(staged.stat().st_size if staged.exists() else 0)})")
            time.sleep(wait)


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def looks_valid(path, row):
    """Structural check used when no md5 is published upstream. NCBI intermittently answers
    with a ~1 KB HTML error page under load; without this a downloader can land that page on
    disk under a .h5ad name and corrupt everything downstream."""
    if path.stat().st_size != row["bytes"]:
        return False, f"size {path.stat().st_size} != expected {row['bytes']}"
    if path.name.endswith(".h5ad"):
        with open(path, "rb") as f:
            if f.read(8) != HDF5_MAGIC:
                return False, "not an HDF5 file (got an error page?)"
    return True, ""


def verify(path, row):
    """(ok, detail). Prefers md5; falls back to size + file-type magic when the host publishes
    no checksum."""
    if row.get("verify") == "size+magic":
        return looks_valid(path, row)
    if path.stat().st_size != row["bytes"]:
        return False, f"size {path.stat().st_size} != expected {row['bytes']}"
    got = md5_of(path)
    return (got == row["md5"]), (f"md5 {got} != {row['md5']}" if got != row["md5"] else "")


def provision(row, args):
    """Return (name, status) for one manifest row."""
    name, dest = row["key"], row["dest"]
    present_ok = dest.is_file() and verify(dest, row)[0]

    # --check never downloads and never cares about --force: it only reports what is on disk.
    if args.check:
        if present_ok:
            return name, "ok (present)"
        return name, "CORRUPT (md5 mismatch)" if dest.exists() else "MISSING"

    if present_ok and not args.force:
        return name, "ok (present)"
    if dest.exists() and not args.force:
        log(f"  [{name}] present but md5 does not match the manifest - refetching")
    if not row.get("url"):
        return name, "NO SOURCE"

    dest.parent.mkdir(parents=True, exist_ok=True)
    staged = dest.with_name(dest.name + ".part")
    t0 = time.time()
    log(f"  [{name}] fetching {human(row['bytes'])}")
    moved = fetch(row["url"], staged, row["bytes"], name)

    if row.get("gunzip"):
        import gzip
        plain = dest.with_name(dest.name + ".ungz")
        with gzip.open(staged, "rb") as fin, open(plain, "wb") as fout:
            shutil.copyfileobj(fin, fout, CHUNK)
        staged.unlink()
        staged = plain

    ok, detail = verify(staged, row)
    if not ok:
        staged.unlink(missing_ok=True)
        return name, f"FAILED ({detail})"
    if row.get("verify") == "size+magic":
        log(f"  [{name}] verified by size + HDF5 magic ({md5_of(staged)}) - "
            "upstream publishes no checksum")
    os.replace(staged, dest)
    el = time.time() - t0
    # Rate is over bytes actually moved, so a resumed download does not report a fictitious
    # speed derived from the full file size.
    log(f"  [{name}] done {human(row['bytes'])} in {el:.0f}s "
        f"({moved / max(el, 1e-9) / 1e6:.1f} MB/s over {human(moved)} transferred)")
    return name, "ok (downloaded)"


# --------------------------------------------------------------------------- planning
def build_plan(args):
    if not MANIFEST.is_file():
        sys.exit(f"ERROR: manifest not found at {MANIFEST}")
    rows = list(csv.DictReader(open(MANIFEST)))

    zen = {}
    wanted_records = {"main", "suppl"} if args.record in (None, "all") else \
        ({args.record} if args.record in ZENODO_RECORDS else set())
    for rec in sorted(wanted_records):
        log(f"resolving Zenodo record {ZENODO_RECORDS[rec]} ({rec}) ...")
        zen[rec] = resolve_zenodo(ZENODO_RECORDS[rec])

    plan, problems = [], []
    for r in rows:
        rec = r["record"]
        group = rec if rec else "public"
        if args.record and args.record != "all" and args.record != group:
            continue
        key = Path(r["file"]).name
        if args.only and not any(o.lower() in key.lower() for o in args.only):
            continue

        row = {"key": key, "dest": Path(args.data_dir) / r["file"], "group": group,
               "bytes": int(r["bytes"]), "md5": r["md5"], "rel": r["file"]}

        if group in zen:
            up = zen[group].get(key)
            if not up:
                problems.append(f"{key}: not present in Zenodo record {ZENODO_RECORDS[group]}")
                continue
            url, size, upmd5 = up
            # Cross-check the published record against the manifest before transferring
            # anything: a mismatch means the two have drifted apart.
            if upmd5 and upmd5 != row["md5"]:
                problems.append(f"{key}: Zenodo md5 {upmd5} != manifest {row['md5']}")
                continue
            if size >= 0 and size != row["bytes"]:
                problems.append(f"{key}: Zenodo size {size} != manifest {row['bytes']}")
                continue
            row["url"] = url
        elif group == "public":
            cands = PUBLIC_SOURCES.get(key) or []
            if not cands:
                problems.append(f"{key}: no public source URL known "
                                f"(manifest says {r['known_public_source'] or 'unknown'})")
                continue
            # Pick by checksum, never by ordering: prefer a candidate whose upstream md5
            # equals the manifest's, then one that declares its own expected bytes
            # (the GEO pair), and only then fall back to a lone candidate.
            src = next((c for c in cands if c.get("md5") and c["md5"] == row["md5"]), None)
            if src is None:
                src = next((c for c in cands if c.get("override_size")), None)
            if src is None and len(cands) == 1 and not cands[0].get("md5"):
                src = cands[0]
            if src is None:
                offered = "; ".join(f"{c['note']} md5={c.get('md5', '?')}" for c in cands)
                problems.append(f"{key}: no source serves the manifest md5 {row['md5']} "
                                f"-- candidates were [{offered}]")
                continue

            row["url"] = src["url"]
            row["gunzip"] = src.get("gunzip", False)
            row["note"] = src.get("note", "")
            # A source may serve a different-but-equivalent artefact than the lab copy the
            # manifest describes (the GEO GSE264667 pair). Verify against the upstream's own
            # size/md5 in that case; validating against the manifest would always fail.
            if src.get("override_size"):
                row["bytes"] = int(src["override_size"])
                row["md5"] = src.get("override_md5") or ""
                row["md5_source"] = "upstream"
                if not row["md5"]:
                    row["verify"] = "size+magic"
        else:
            continue
        plan.append(row)
    return plan, problems


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Records: main=%d  suppl=%d" % (ZENODO_RECORDS["main"], ZENODO_RECORDS["suppl"]))
    ap.add_argument("--data-dir", default=os.environ.get("CIPHER_DATA_DIR"),
                    help="destination (default: $CIPHER_DATA_DIR)")
    ap.add_argument("--record", choices=["all", "main", "suppl", "public"], default="all",
                    help="fetch only one group (default: all)")
    ap.add_argument("--only", action="append", metavar="SUBSTR",
                    help="only files whose name contains this (repeatable)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel downloads (default: 4)")
    ap.add_argument("--force", action="store_true", help="refetch even if present and valid")
    ap.add_argument("--check", action="store_true",
                    help="verify what is present against the manifest; download nothing")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and exit")
    args = ap.parse_args(argv)

    if not args.data_dir:
        sys.exit("ERROR: set $CIPHER_DATA_DIR or pass --data-dir")
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    log(f"CIPHER_DATA_DIR: {args.data_dir}")

    if args.record in ("all", "public") or args.record is None:
        resolve_public_sources()

    plan, problems = build_plan(args)
    if problems:
        log("\nPROBLEMS (these files will not be fetched):")
        for p in problems:
            log(f"  - {p}")

    if not plan:
        log("\nnothing to do.")
        return 1 if problems else 0

    total = sum(r["bytes"] for r in plan)
    by_group = {}
    for r in plan:
        by_group.setdefault(r["group"], []).append(r)
    log(f"\nplan: {len(plan)} files, {total / 1e9:.2f} GB")
    for g in ("main", "suppl", "public"):
        if g in by_group:
            log(f"  {g:7s} {len(by_group[g]):2d} files  "
                f"{sum(r['bytes'] for r in by_group[g]) / 1e9:6.2f} GB")

    if args.dry_run:
        log("\n--dry-run: files that would be fetched")
        for r in sorted(plan, key=lambda r: -r["bytes"]):
            log(f"  {human(r['bytes']):>10}  {r['rel']:70s} {r.get('url', '')[:60]}")
        return 0

    log("")
    results = []
    # Largest first so the long poles start immediately rather than trailing at the end.
    ordered = sorted(plan, key=lambda r: -r["bytes"])
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(provision, r, args): r for r in ordered}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((r["key"], f"FAILED ({e})"))

    width = max(len(n) for n, _ in results)
    log("\n" + "=" * (width + 26))
    log("SUMMARY")
    log("=" * (width + 26))
    for name, status in sorted(results):
        log(f"  {name:<{width}}  {status}")
    bad = [n for n, s in results if not s.startswith("ok")]
    log("=" * (width + 26))

    if bad:
        log(f"\n{len(bad)} file(s) not in place: {', '.join(bad)}")
        if args.check:
            log("Run without --check to fetch them.")
        return 1

    log(f"\nAll {len(results)} file(s) present and md5-verified.")
    if args.record in ("all", "suppl"):
        print_precompute_note()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
