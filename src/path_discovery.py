"""
path_discovery.py
------------------
Locates the Amazon ML Challenge 2026 (Business Entity Resolution) dataset on the
local machine, without assuming any fixed install location.

Handles three situations, in order of preference:
  1. An already-extracted folder that contains dataset/train/*.tsv and dataset/test/*.tsv
  2. A .zip archive (e.g. "student_resource.zip") sitting anywhere under the search
     roots -- it is peeked into (without full extraction) to confirm it really is the
     challenge dataset, then extracted next to itself.
  3. An explicit --data-root passed on the command line, which is trusted as-is.

Nothing here reads or interprets the *contents* of business records -- it only looks
at file/folder names and a lightweight zip-namelist peek, so it is safe to run before
you have decided anything about the data itself.

Windows note: this module is written with pathlib throughout, so it works unchanged
on Windows, macOS or Linux. No hard-coded path separators.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("ber.path_discovery")

# Filenames we expect to find, relative to the dataset root, per the challenge spec.
EXPECTED_TRAIN_FILES = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
]
EXPECTED_TEST_FILES = [
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
]
OPTIONAL_FILES = [
    "utils/validate_submission.py",
    "Documentation_template.md",
]

# Default places to look, if the caller does not specify --search-root.
# These are common download / project locations on Windows and are skipped
# silently if they do not exist.
DEFAULT_SEARCH_ROOTS = [
    Path.cwd(),
    Path.cwd().parent,
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent.parent,
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    Path.home() / "Documents",
]


@dataclass
class DiscoveredDataset:
    """Resolved, absolute paths to every file the pipeline needs."""

    root: Path
    train_source1: Optional[Path] = None
    train_source2: Optional[Path] = None
    train_source3: Optional[Path] = None
    train_ground_truth: Optional[Path] = None
    test_source1: Optional[Path] = None
    test_source2: Optional[Path] = None
    test_source3: Optional[Path] = None
    validator_script: Optional[Path] = None
    doc_template: Optional[Path] = None
    missing: list = field(default_factory=list)

    def is_train_complete(self) -> bool:
        return all(
            [
                self.train_source1,
                self.train_source2,
                self.train_source3,
                self.train_ground_truth,
            ]
        )

    def is_test_complete(self) -> bool:
        return all([self.test_source1, self.test_source2, self.test_source3])

    def summary(self) -> str:
        lines = [f"Dataset root: {self.root}"]
        for label, p in [
            ("train_source1", self.train_source1),
            ("train_source2", self.train_source2),
            ("train_source3", self.train_source3),
            ("train_ground_truth", self.train_ground_truth),
            ("test_source1", self.test_source1),
            ("test_source2", self.test_source2),
            ("test_source3", self.test_source3),
            ("validator_script", self.validator_script),
            ("doc_template", self.doc_template),
        ]:
            status = str(p) if p else "NOT FOUND"
            lines.append(f"  {label:20s}: {status}")
        if self.missing:
            lines.append("  Missing / not found:")
            for m in self.missing:
                lines.append(f"    - {m}")
        return "\n".join(lines)


def _resolve_from_root(root: Path) -> DiscoveredDataset:
    """Given a folder that is believed to be (or contain) the dataset, resolve
    every expected file path underneath it, searching a couple of plausible
    nesting levels since zip extraction sometimes adds a wrapper folder."""

    candidates = [root] + [p for p in root.iterdir() if p.is_dir()] if root.exists() else [root]

    def find_file(relative: str) -> Optional[Path]:
        rel_path = Path(relative)
        for base in candidates:
            direct = base / rel_path
            if direct.is_file():
                return direct.resolve()
        # Fall back to a recursive search by filename only (handles unexpected nesting).
        matches = list(root.rglob(rel_path.name))
        if matches:
            if len(matches) > 1:
                logger.warning(
                    "Multiple candidates found for %s, using the first: %s",
                    rel_path.name,
                    matches,
                )
            return matches[0].resolve()
        return None

    ds = DiscoveredDataset(root=root.resolve())
    ds.train_source1 = find_file("dataset/train/train_source1.tsv")
    ds.train_source2 = find_file("dataset/train/train_source2.tsv")
    ds.train_source3 = find_file("dataset/train/train_source3.tsv")
    ds.train_ground_truth = find_file("dataset/train/train_ground_truth.tsv")
    ds.test_source1 = find_file("dataset/test/test_source1.tsv")
    ds.test_source2 = find_file("dataset/test/test_source2.tsv")
    ds.test_source3 = find_file("dataset/test/test_source3.tsv")
    ds.validator_script = find_file("utils/validate_submission.py")
    ds.doc_template = find_file("Documentation_template.md")

    for label, p in [
        ("train_source1.tsv", ds.train_source1),
        ("train_source2.tsv", ds.train_source2),
        ("train_source3.tsv", ds.train_source3),
        ("train_ground_truth.tsv", ds.train_ground_truth),
        ("test_source1.tsv", ds.test_source1),
        ("test_source2.tsv", ds.test_source2),
        ("test_source3.tsv", ds.test_source3),
        ("validate_submission.py", ds.validator_script),
        ("Documentation_template.md", ds.doc_template),
    ]:
        if p is None:
            ds.missing.append(label)

    return ds


def _looks_like_challenge_zip(zip_path: Path) -> bool:
    """Peek at a zip's namelist (no extraction) and check for a couple of the
    expected filenames, so we don't extract random unrelated archives."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError) as exc:
        logger.debug("Skipping unreadable zip %s (%s)", zip_path, exc)
        return False

    hits = sum(
        1
        for name in names
        if name.endswith("train_source1.tsv")
        or name.endswith("train_ground_truth.tsv")
        or name.endswith("test_source1.tsv")
    )
    return hits >= 2


def _extract_zip(zip_path: Path) -> Path:
    """Extract a confirmed challenge zip next to itself, into
    '<zipstem>_extracted'. Skips extraction if that folder already exists and
    already looks complete."""
    extract_dir = zip_path.parent / f"{zip_path.stem}_extracted"
    marker = extract_dir / ".extraction_complete"

    if marker.exists():
        logger.info("Using previously extracted dataset at %s", extract_dir)
        return extract_dir

    logger.info("Extracting %s -> %s", zip_path, extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    return extract_dir


def _find_candidate_zips(search_roots: Iterable[Path], max_depth: int = 6) -> list:
    """Walk each search root looking for .zip files, bounded by max_depth so a
    search root like the whole home directory does not take forever."""
    found = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        root_depth = len(root.parts)
        for path in root.rglob("*.zip"):
            depth = len(path.parts) - root_depth
            if depth <= max_depth:
                found.append(path)
    return found


def discover_dataset(
    data_root: Optional[str] = None,
    search_roots: Optional[Iterable[str]] = None,
) -> DiscoveredDataset:
    """Main entry point.

    Parameters
    ----------
    data_root:
        If given, treated as a trusted, already-extracted dataset folder (or a
        zip file, which will be extracted). Skips the search step entirely.
    search_roots:
        Folders to search for a dataset folder or challenge zip. Defaults to
        common download/project locations if not given.

    Returns
    -------
    DiscoveredDataset with every resolved path (fields are None if not found).
    """
    if data_root:
        root_path = Path(data_root).expanduser()
        if root_path.is_file() and root_path.suffix.lower() == ".zip":
            root_path = _extract_zip(root_path)
        logger.info("Using explicit data root: %s", root_path)
        return _resolve_from_root(root_path)

    roots = [Path(r).expanduser() for r in search_roots] if search_roots else DEFAULT_SEARCH_ROOTS
    roots = [r for r in roots if r.exists()]
    logger.info("Searching for dataset under: %s", [str(r) for r in roots])

    # 1) Look for an already-extracted dataset folder first (fast path, no zip work).
    for root in roots:
        for path in root.rglob("dataset"):
            if (path / "train" / "train_source1.tsv").is_file():
                logger.info("Found extracted dataset at %s", path.parent)
                return _resolve_from_root(path.parent)

    # 2) Fall back to locating and extracting a challenge zip.
    zips = _find_candidate_zips(roots)
    logger.info("Found %d zip file(s) to inspect", len(zips))
    for zip_path in zips:
        if _looks_like_challenge_zip(zip_path):
            logger.info("Identified challenge dataset zip: %s", zip_path)
            extracted_root = _extract_zip(zip_path)
            return _resolve_from_root(extracted_root)

    logger.warning(
        "Could not auto-discover the dataset under any search root. "
        "Re-run with --data-root pointing at the extracted folder or the .zip file."
    )
    # Return an empty result rooted at the first search root (or cwd) so callers
    # get a consistent object with a useful "missing" report instead of a crash.
    fallback_root = roots[0] if roots else Path.cwd()
    return _resolve_from_root(fallback_root)