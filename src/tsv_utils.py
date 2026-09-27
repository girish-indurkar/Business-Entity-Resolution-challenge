"""
tsv_utils.py
------------
Memory-efficient helpers for reading the challenge's .tsv files.

Design choices:
- Always sep="\t" explicitly (the challenge spec calls out that commas appear
  inside fields, so a naive read_csv default would silently corrupt columns).
- dtype=str + keep_default_na=False everywhere: entity IDs and the ID-list
  columns must never be silently coerced to numbers/NaN by pandas' type
  inference, and an empty matched_entity_ids / candidate_entity_ids cell is a
  meaningful value ("no match"), not a missing value.
- Chunked reading via an iterator, so multi-GB files are processed in fixed-size
  blocks instead of being loaded whole into memory.
- Encoding fallback: try utf-8 first (with BOM handling), fall back to latin-1
  if that fails, since the noise taxonomy explicitly mentions transliteration
  and mixed-source formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

logger = logging.getLogger("ber.tsv_utils")

DEFAULT_CHUNKSIZE = 100_000


def count_rows_fast(path: Path) -> int:
    """Count data rows (excluding header) without loading the file into pandas.
    Uses a raw binary newline count, which is fast and memory-flat even for
    multi-GB files. Slightly overcounts if the last line has no trailing
    newline in a way that differs from the header convention, so treat this as
    an approximate progress indicator, not a validation-grade count."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return max(count - 1, 0)  # subtract header row


def _open_with_encoding_fallback(path: Path, chunksize: Optional[int], usecols: Optional[List[str]]):
    last_exc = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            reader = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                na_values=[],
                encoding=encoding,
                chunksize=chunksize,
                usecols=usecols,
                engine="c",
            )
            # For chunksize=None this returns a DataFrame directly (no lazy
            # validation of encoding until iterated), so force a peek for
            # chunked mode is not possible without consuming a chunk; instead
            # we trust the C engine to raise UnicodeDecodeError up front for
            # chunksize=None mode, and rely on the per-chunk try/except in
            # iter_tsv_chunks for chunked mode.
            if chunksize is None:
                logger.debug("Read %s fully with encoding=%s", path, encoding)
            return reader, encoding
        except UnicodeDecodeError as exc:
            last_exc = exc
            logger.warning("Encoding %s failed for %s, trying next fallback", encoding, path)
            continue
    raise RuntimeError(f"Could not decode {path} with any known encoding") from last_exc


def iter_tsv_chunks(
    path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    usecols: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Yield successive DataFrames of at most `chunksize` rows from a TSV file.

    Every column is read as `str` and empty cells are kept as empty strings
    rather than NaN, matching the challenge's ID-list semantics.
    """
    reader, encoding = _open_with_encoding_fallback(path, chunksize=chunksize, usecols=usecols)
    try:
        for chunk in reader:
            yield chunk
    except UnicodeDecodeError:
        # The C engine can still fail mid-stream on a bad chunk with utf-8;
        # restart cleanly with latin-1, which accepts any byte sequence.
        logger.warning(
            "Mid-file decode error reading %s with encoding=%s; restarting with latin-1",
            path,
            encoding,
        )
        reader2 = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_values=[],
            encoding="latin-1",
            chunksize=chunksize,
            usecols=usecols,
            engine="c",
        )
        for chunk in reader2:
            yield chunk


def read_tsv_full(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a whole TSV into memory. Only use this for files you know are
    small enough (e.g. train_ground_truth.tsv), not for the large source
    files."""
    df, _ = _open_with_encoding_fallback(path, chunksize=None, usecols=usecols)
    return df


def validate_expected_columns(path: Path, expected: List[str]) -> List[str]:
    """Peek at just the header row and report any expected columns that are
    missing, without reading the rest of the file. Returns the list of
    missing column names (empty list means the header matched)."""
    header = pd.read_csv(path, sep="\t", nrows=0, dtype=str, encoding="utf-8-sig")
    actual_cols = list(header.columns)
    missing = [c for c in expected if c not in actual_cols]
    if missing:
        logger.warning("File %s is missing expected columns: %s (found: %s)", path, missing, actual_cols)
    return missing