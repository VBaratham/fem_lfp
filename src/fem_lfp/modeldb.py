"""Fetch and compile ModelDB cell models on demand.

The example scenarios use published cell models (Hay 2011 = ModelDB
139653, Mainen & Sejnowski 1996 = ModelDB 2488). Those archives aren't
redistributed here — ModelDB shares them under its default reuse policy,
not an explicit FOSS license — so instead of vendoring them we download
on first run. This keeps the scenarios "clone and run": no manual trip to
ModelDB, no by-hand ``nrnivmodl``.

Stdlib only (urllib + zipfile + subprocess); no new dependencies.
"""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

MODELDB_DOWNLOAD = "https://modeldb.science/download/{id}"


def _arch_is_built(mod_dir: Path) -> bool:
    """True if ``nrnivmodl`` output already exists under ``mod_dir``."""
    for pat in ("*/libnrnmech.dylib", "*/.libs/libnrnmech.so",
                "*/libnrnmech.so", "*/special"):
        if any(mod_dir.glob(pat)):
            return True
    return False


def compile_mods(mod_dir: Path) -> Path:
    """Run ``nrnivmodl`` in ``mod_dir`` (idempotent). Returns ``mod_dir``.

    NEURON drops the compiled ``<arch>/libnrnmech`` next to the .mod files,
    which is where the scenarios load it from.
    """
    mod_dir = Path(mod_dir)
    if _arch_is_built(mod_dir):
        return mod_dir
    if shutil.which("nrnivmodl") is None:
        raise RuntimeError(
            "nrnivmodl not on PATH — activate the NEURON/conda env first."
        )
    logger.info(f"[modeldb] compiling NMODL mechanisms in {mod_dir} …")
    res = subprocess.run(["nrnivmodl"], cwd=str(mod_dir), text=True,
                         capture_output=True)
    if res.returncode != 0 or not _arch_is_built(mod_dir):
        raise RuntimeError(
            f"nrnivmodl failed in {mod_dir} (exit {res.returncode}):\n"
            f"{res.stdout[-2000:]}\n{res.stderr[-2000:]}"
        )
    return mod_dir


def fetch(model_id: int, target_dir: Path, *, inner: str) -> Path:
    """Download ModelDB ``model_id`` and place its ``inner`` subtree at
    ``target_dir`` (idempotent — a no-op if ``target_dir`` already exists).

    ``inner`` is the name of the directory inside the archive to keep
    (found anywhere in the extracted tree). Returns ``target_dir``.
    """
    target_dir = Path(target_dir)
    if target_dir.exists():
        return target_dir
    url = MODELDB_DOWNLOAD.format(id=model_id)
    logger.info(f"[modeldb] downloading ModelDB {model_id} from {url} …")
    with urllib.request.urlopen(url) as resp:   # noqa: S310 (trusted host)
        data = resp.read()
    logger.info(f"[modeldb]   {len(data) / 1024:.0f} KiB; extracting …")
    tmp = target_dir.parent / f"_modeldb_{model_id}_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tmp)
        matches = [p for p in tmp.rglob(inner) if p.is_dir()]
        if not matches:
            raise RuntimeError(
                f"no '{inner}' directory inside ModelDB {model_id} archive"
            )
        # Shallowest match, in case the name recurs deeper.
        src = min(matches, key=lambda p: len(p.parts))
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target_dir))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    logger.info(f"[modeldb]   → {target_dir}")
    return target_dir


def ensure_cell(
    model_id: int,
    target_dir: Path,
    *,
    inner: str,
    mod_subdir: str = "",
) -> Path:
    """Download (if needed) and compile a ModelDB cell. Returns ``target_dir``.

    ``mod_subdir`` is where the .mod files live relative to ``target_dir``
    (``""`` = the directory itself, e.g. M&S; ``"mod"`` = Hay 2011).
    """
    fetch(model_id, target_dir, inner=inner)
    compile_mods(Path(target_dir) / mod_subdir if mod_subdir else target_dir)
    return target_dir


if __name__ == "__main__":
    # `python -m fem_lfp.modeldb <id> <target_dir> <inner> [mod_subdir]`
    args = sys.argv[1:]
    if len(args) < 3:
        sys.exit("usage: python -m fem_lfp.modeldb <id> <target_dir> "
                 "<inner> [mod_subdir]")
    ensure_cell(int(args[0]), Path(args[1]), inner=args[2],
                mod_subdir=args[3] if len(args) > 3 else "")
