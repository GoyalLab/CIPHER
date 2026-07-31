"""Lightweight logging helper for the supplementary reproduction notebooks.

The pipeline functions in ``notebooks/src`` print a lot of per-dataset progress (and,
historically, absolute file paths). For a clean, shareable notebook we want:

* the **plots** to remain as cell outputs (they render via the display machinery, not
  stdout, so they are unaffected by the stdout redirect below);
* only a **one-line pointer** to a log file per step in the cell output;
* the full verbose log written to a relative ``.log`` file so exact reproduction can still
  be checked;
* **no absolute paths** anywhere in the logs (so allocation IDs / usernames are never
  published) -- absolute paths are rewritten to ``$CIPHER_DATA_DIR`` / ``$SUPPL_OUT`` /
  ``~`` placeholders or collapsed to their basename.

Usage in a notebook cell::

    from src.logutil import log_run
    with log_run("figS7_variant1"):
        R.covcorr_variant1_raw_preserving()

Not part of the installable ``cipher`` package.
"""
from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager

_HOME = os.path.expanduser("~")

# Absolute-path prefixes -> placeholders, longest first so nested prefixes win.
def _prefix_subs():
    subs = []
    suppl_out = os.environ.get("SUPPL_OUT")
    data_dir = os.environ.get("CIPHER_DATA_DIR")
    if suppl_out:
        subs.append((os.path.abspath(suppl_out), "$SUPPL_OUT"))
    if data_dir:
        subs.append((os.path.join(os.path.abspath(data_dir), "suppl"), "$SUPPL"))
        subs.append((os.path.abspath(data_dir), "$CIPHER_DATA_DIR"))
    subs.append((_HOME, "~"))
    return sorted((a, b) for a, b in subs if a)

# Any residual absolute path (that escaped the prefix rewrite) -> ".../<basename>".
_ABS_PATH = re.compile(r"/(?:projects|gpfs|home|scratch|tmp|mnt)/[^\s'\"),;]+")


def sanitize(text: str) -> str:
    """Rewrite absolute paths in ``text`` to placeholders / basenames."""
    for prefix, placeholder in sorted(_prefix_subs(), key=lambda kv: -len(kv[0])):
        text = text.replace(prefix, placeholder)
    text = _ABS_PATH.sub(lambda m: ".../" + os.path.basename(m.group(0).rstrip("/")), text)
    return text


class _SanitizingWriter:
    """File-like object that sanitizes text before writing to the log file."""

    def __init__(self, fh):
        self._fh = fh

    def write(self, s):
        self._fh.write(sanitize(s))
        return len(s)

    def flush(self):
        self._fh.flush()

    def isatty(self):
        return False


@contextmanager
def log_run(name: str, out_dir: str | None = None):
    """Redirect verbose stdout/stderr of the wrapped call to ``<out_dir>/logs/<name>.log``.

    ``out_dir`` defaults to ``$SUPPL_OUT`` (else the current directory). Plots and rich
    ``display(...)`` outputs still render in the cell; the cell shows a one-line pointer to
    the log. Tracebacks are unaffected (they use the kernel's error channel), and stdout/
    stderr are always restored on exit.
    """
    base = out_dir or os.environ.get("SUPPL_OUT") or "."
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    try:
        rel = os.path.relpath(log_path)
    except ValueError:
        rel = log_path
    rel = sanitize(rel)

    real_out, real_err = sys.stdout, sys.stderr
    fh = open(log_path, "w", encoding="utf-8")
    writer = _SanitizingWriter(fh)
    sys.stdout = writer
    sys.stderr = writer
    try:
        yield
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        try:
            fh.flush()
            fh.close()
        except Exception:
            pass
        print(f"[log] {rel}")


# --------------------------------------------------------------------------------------
# Notebook-level routing: install once in the config cell; all subsequent verbose output
# goes to a sanitized .log while concise lines still reach the cell.
# --------------------------------------------------------------------------------------

# High-volume lines that should go to the log only (not the cell).
_VERBOSE_TAG = re.compile(
    r"^\s*\[(?:dataset|folder|sigma|load|gene removal|saved|breadth|ok|skip|"
    r"CELLxGENE[^\]]*|RPE[^\]]*|placeholder|removed genes|log)\]"
)
_SEP = re.compile(r"^\s*(?:={5,}|-{5,}|!{5,}|\*{5,})\s*$")


def _is_verbose_line(line: str) -> bool:
    if not line.strip():
        return True
    if "\r" in line:                      # tqdm in-place progress
        return True
    if re.search(r"\d+%\|", line) or "it/s]" in line or "it/s," in line:
        return True
    if _SEP.match(line):
        return True
    return bool(_VERBOSE_TAG.match(line))


class _TeeWriter:
    """Sanitize everything to the log file; pass only concise lines to the cell."""

    def __init__(self, log_fh, console):
        self._fh = log_fh
        self._console = console
        self._buf = ""

    def write(self, s):
        s = sanitize(s)
        self._fh.write(s)
        self._buf += s
        while True:
            n = self._buf.find("\n")
            r = self._buf.find("\r")
            cut = min([i for i in (n, r) if i >= 0], default=-1)
            if cut < 0:
                break
            line, self._buf = self._buf[: cut + 1], self._buf[cut + 1:]
            if not _is_verbose_line(line):
                self._console.write(line)
        return len(s)

    def flush(self):
        self._fh.flush()
        self._console.flush()

    def isatty(self):
        return False


def _caller_out_dir():
    frame = sys._getframe(2)  # caller of route_logs
    g = frame.f_globals
    for key in ("SUPPL_OUT", "OUTDIR", "REPRO", "OUT_DIR", "OUTBASE", "OUT"):
        v = g.get(key)
        if isinstance(v, str) and v:
            return v
    return os.environ.get("SUPPL_OUT") or "."


def route_logs(name: str, out_dir: str | None = None):
    """Route this notebook's verbose stdout/stderr to ``<out_dir>/logs/<name>.log``.

    Call once in the config cell (after the output directory is defined). ``out_dir``
    defaults to the caller's ``SUPPL_OUT``/``OUTDIR``/``REPRO`` global. Concise lines still
    print to the cell; per-dataset spam, separators, tqdm bars, and absolute paths do not.
    Plots and ``display(...)`` outputs are unaffected. Idempotent within a kernel.
    """
    base = out_dir or _caller_out_dir()
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    try:
        rel = sanitize(os.path.relpath(log_path))
    except ValueError:
        rel = sanitize(log_path)

    # avoid double-wrapping if a cell re-runs
    if isinstance(sys.stdout, _TeeWriter):
        return log_path
    fh = open(log_path, "w", encoding="utf-8")
    # capture the CURRENT streams (under nbconvert these are the kernel's capturing
    # OutStreams, so concise lines still land in the cell output)
    console_out, console_err = sys.stdout, sys.stderr
    sys.stdout = _TeeWriter(fh, console_out)
    sys.stderr = _TeeWriter(fh, console_err)
    console_out.write(f"[logs] verbose output -> {rel}\n")
    return log_path
