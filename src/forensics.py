"""
forensics.py
------------
Phase 0 "dataset forensics" for the Business Entity Resolution challenge.

Everything here is computed in a single streaming pass per file (chunked
reading via tsv_utils), so it works memory-efficiently on large source files.
Nothing assumes a particular noise taxonomy in advance -- it *measures*
candidate noise patterns (legal suffixes, abbreviations, punctuation,
non-ASCII ratios, token reordering proxies, exact/near duplicates) so you can
build a taxonomy from evidence rather than guesswork, per the challenge brief.

Outputs:
- A JSON-serialisable report dict per file (source1/2/3, ground truth).
- A handful of CSV samples (e.g. most-repeated names/addresses, a sample of
  exact-duplicate records) written under <out_dir>/samples/ for manual review.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import tsv_utils

logger = logging.getLogger("ber.forensics")

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Common legal-suffix / entity-type tokens worth counting explicitly. This is a
# starting vocabulary for *measurement*, not a claim about what exists in your
# data -- the report's "unclassified_frequent_tokens" section is what tells you
# what to add here.
LEGAL_SUFFIX_PATTERNS = {
    "corp_corporation": r"\bcorp(?:oration)?\b",
    "ltd_limited": r"\bl(?:t)?d\b|\blimited\b",
    "pvt_private": r"\bpvt\b|\bprivate\b",
    "llc": r"\bllc\b",
    "inc_incorporated": r"\binc(?:orporated)?\b",
    "co_company": r"\bco\b|\bcompany\b",
    "llp": r"\bllp\b",
    "plc": r"\bplc\b",
    "gmbh": r"\bgmbh\b",
    "sa_sarl": r"\bsa\b|\bsarl\b",
    "and_ampersand": r"&|\band\b",
}

ADDRESS_ABBREV_PATTERNS = {
    "rd_road": r"\brd\b|\broad\b",
    "st_street": r"\bst\b|\bstreet\b",
    "ave_avenue": r"\bave\b|\bavenue\b",
    "blvd_boulevard": r"\bblvd\b|\bboulevard\b",
    "near_landmark": r"\bnear\b|\bopp(?:osite)?\b",
    "apt_flat": r"\bapt\b|\bflat\b|\bunit\b",
}

TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass
class RunningLengthStats:
    """Welford's online algorithm for mean/variance, plus min/max, so we never
    need to hold every length value in memory at once."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: Optional[int] = None
    max_val: Optional[int] = None

    def update(self, value: int) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = value if self.min_val is None else min(self.min_val, value)
        self.max_val = value if self.max_val is None else max(self.max_val, value)

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def stdev(self) -> float:
        return self.variance ** 0.5

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": round(self.mean, 3),
            "stdev": round(self.stdev, 3),
            "min": self.min_val,
            "max": self.max_val,
        }


def _normalize_for_grouping(text: str) -> str:
    """Cheap, non-authoritative normalization used only to *measure* how many
    near-duplicate groups a simple normalized-name block would produce -- this
    is diagnostic, not the final blocking strategy."""
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    tokens = [t for t in TOKEN_SPLIT_RE.split(text) if t]
    return " ".join(sorted(tokens))


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)


@dataclass
class SourceFileReport:
    source_label: str
    path: str
    total_rows: int = 0
    missing_columns: List[str] = field(default_factory=list)
    id_prefix_mismatches: int = 0
    duplicate_entity_ids: int = 0
    exact_duplicate_records: int = 0  # rows with identical (name, address, country)
    empty_name_count: int = 0
    empty_address_count: int = 0
    empty_country_count: int = 0
    name_length_chars: dict = field(default_factory=dict)
    name_length_tokens: dict = field(default_factory=dict)
    address_length_chars: dict = field(default_factory=dict)
    address_length_tokens: dict = field(default_factory=dict)
    country_distribution: dict = field(default_factory=dict)
    legal_suffix_hits: dict = field(default_factory=dict)
    address_abbrev_hits: dict = field(default_factory=dict)
    name_non_ascii_ratio_mean: float = 0.0
    address_non_ascii_ratio_mean: float = 0.0
    top_repeated_names: list = field(default_factory=list)
    top_repeated_addresses: list = field(default_factory=list)
    normalized_name_block_size_histogram: dict = field(default_factory=dict)
    largest_normalized_name_blocks: list = field(default_factory=list)


def analyze_source_file(
    path: Path,
    source_label: str,
    chunksize: int = tsv_utils.DEFAULT_CHUNKSIZE,
    top_n: int = 20,
) -> SourceFileReport:
    """Single streaming pass over one source file, accumulating every stat in
    SourceFileReport. Safe for files far larger than available RAM, at the
    cost of holding Counters keyed by unique name/address/country values --
    if your dataset has an extremely high-cardinality address field this can
    still grow large; see the README note on scaling this further with a
    disk-backed counter (e.g. sqlite) if needed."""

    report = SourceFileReport(source_label=source_label, path=str(path))
    report.missing_columns = tsv_utils.validate_expected_columns(path, SOURCE_COLUMNS)
    if report.missing_columns:
        logger.error("Skipping detailed analysis of %s: missing columns %s", path, report.missing_columns)
        return report

    expected_prefix = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}.get(source_label, "")

    name_len_chars = RunningLengthStats()
    name_len_tokens = RunningLengthStats()
    addr_len_chars = RunningLengthStats()
    addr_len_tokens = RunningLengthStats()

    seen_entity_ids: set = set()
    exact_record_counter: Counter = Counter()
    name_counter: Counter = Counter()
    address_counter: Counter = Counter()
    country_counter: Counter = Counter()
    legal_suffix_counter: Counter = Counter({k: 0 for k in LEGAL_SUFFIX_PATTERNS})
    address_abbrev_counter: Counter = Counter({k: 0 for k in ADDRESS_ABBREV_PATTERNS})
    normalized_name_groups: Counter = Counter()

    name_non_ascii_sum = 0.0
    address_non_ascii_sum = 0.0

    compiled_legal = {k: re.compile(v, re.IGNORECASE) for k, v in LEGAL_SUFFIX_PATTERNS.items()}
    compiled_addr = {k: re.compile(v, re.IGNORECASE) for k, v in ADDRESS_ABBREV_PATTERNS.items()}

    rows_processed = 0
    for chunk in tsv_utils.iter_tsv_chunks(path, chunksize=chunksize, usecols=SOURCE_COLUMNS):
        rows_processed += len(chunk)

        for eid in chunk["entity_id"]:
            if expected_prefix and not eid.startswith(expected_prefix):
                report.id_prefix_mismatches += 1
            if eid in seen_entity_ids:
                report.duplicate_entity_ids += 1
            else:
                seen_entity_ids.add(eid)

        for name, addr, country in zip(
            chunk["business_name"], chunk["business_address"], chunk["country"]
        ):
            if name == "":
                report.empty_name_count += 1
            if addr == "":
                report.empty_address_count += 1
            if country == "":
                report.empty_country_count += 1

            name_len_chars.update(len(name))
            name_len_tokens.update(len([t for t in TOKEN_SPLIT_RE.split(name) if t]))
            addr_len_chars.update(len(addr))
            addr_len_tokens.update(len([t for t in TOKEN_SPLIT_RE.split(addr) if t]))

            name_non_ascii_sum += _non_ascii_ratio(name)
            address_non_ascii_sum += _non_ascii_ratio(addr)

            name_counter[name] += 1
            address_counter[addr] += 1
            country_counter[country] += 1
            exact_record_counter[(name, addr, country)] += 1

            for key, pattern in compiled_legal.items():
                if pattern.search(name):
                    legal_suffix_counter[key] += 1
            for key, pattern in compiled_addr.items():
                if pattern.search(addr):
                    address_abbrev_counter[key] += 1

            normalized_name_groups[_normalize_for_grouping(name)] += 1

        logger.info("[%s] processed %d rows so far", source_label, rows_processed)

    report.total_rows = rows_processed
    report.exact_duplicate_records = sum(c - 1 for c in exact_record_counter.values() if c > 1)
    report.name_length_chars = name_len_chars.to_dict()
    report.name_length_tokens = name_len_tokens.to_dict()
    report.address_length_chars = addr_len_chars.to_dict()
    report.address_length_tokens = addr_len_tokens.to_dict()
    report.country_distribution = dict(country_counter.most_common())
    report.legal_suffix_hits = dict(legal_suffix_counter)
    report.address_abbrev_hits = dict(address_abbrev_counter)
    report.name_non_ascii_ratio_mean = round(name_non_ascii_sum / max(rows_processed, 1), 5)
    report.address_non_ascii_ratio_mean = round(address_non_ascii_sum / max(rows_processed, 1), 5)
    report.top_repeated_names = [
        {"business_name": n, "count": c} for n, c in name_counter.most_common(top_n) if c > 1
    ]
    report.top_repeated_addresses = [
        {"business_address": a, "count": c} for a, c in address_counter.most_common(top_n) if c > 1
    ]

    block_sizes = Counter(normalized_name_groups.values())
    report.normalized_name_block_size_histogram = dict(sorted(block_sizes.items()))
    largest_blocks = normalized_name_groups.most_common(top_n)
    report.largest_normalized_name_blocks = [
        {"normalized_name": n, "record_count": c} for n, c in largest_blocks if c > 1
    ]

    return report


@dataclass
class GroundTruthReport:
    path: str
    total_source1_rows: int = 0
    duplicate_source1_rows: int = 0
    singleton_count: int = 0
    match_count_histogram: dict = field(default_factory=dict)
    total_matched_ids: int = 0
    matched_to_source2: int = 0
    matched_to_source3: int = 0
    duplicate_ids_within_a_row: int = 0
    s2_s3_ids_matched_to_multiple_s1: int = 0
    top_multiply_matched_ids: list = field(default_factory=list)


def analyze_ground_truth(path: Path, top_n: int = 20) -> GroundTruthReport:
    report = GroundTruthReport(path=str(path))
    missing = tsv_utils.validate_expected_columns(path, ["source1_entity_id", "matched_entity_ids"])
    if missing:
        logger.error("Ground truth file %s missing expected columns: %s", path, missing)
        return report

    df = tsv_utils.read_tsv_full(path)
    report.total_source1_rows = len(df)
    report.duplicate_source1_rows = int(df["source1_entity_id"].duplicated().sum())

    match_count_counter: Counter = Counter()
    s2_s3_to_s1_count: Dict[str, int] = defaultdict(int)

    for s1_id, matched_raw in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        matched_raw = matched_raw.strip()
        if matched_raw == "":
            ids = []
        else:
            ids = [x.strip() for x in matched_raw.split(",") if x.strip() != ""]

        if len(ids) != len(set(ids)):
            report.duplicate_ids_within_a_row += 1

        n = len(ids)
        match_count_counter[n] += 1
        if n == 0:
            report.singleton_count += 1

        report.total_matched_ids += n
        report.matched_to_source2 += sum(1 for i in ids if i.startswith("S2-"))
        report.matched_to_source3 += sum(1 for i in ids if i.startswith("S3-"))

        for i in ids:
            s2_s3_to_s1_count[i] += 1

    # Bucket the histogram so 0,1,2,3 are exact and "4+" catches the tail.
    bucketed: Dict[str, int] = defaultdict(int)
    for n, c in match_count_counter.items():
        key = str(n) if n < 4 else "4+"
        bucketed[key] += c
    report.match_count_histogram = dict(sorted(bucketed.items(), key=lambda kv: (kv[0] != "4+", kv[0])))

    multiply_matched = {k: v for k, v in s2_s3_to_s1_count.items() if v > 1}
    report.s2_s3_ids_matched_to_multiple_s1 = len(multiply_matched)
    report.top_multiply_matched_ids = [
        {"entity_id": k, "num_source1_matches": v}
        for k, v in sorted(multiply_matched.items(), key=lambda kv: -kv[1])[:top_n]
    ]

    return report


def write_report(report_dict: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forensics_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info("Wrote full forensics report to %s", out_path)
    return out_path


def write_samples(
    reports: Dict[str, SourceFileReport],
    out_dir: Path,
) -> None:
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for label, report in reports.items():
        if report.top_repeated_names:
            pd.DataFrame(report.top_repeated_names).to_csv(
                samples_dir / f"{label}_top_repeated_names.csv", index=False
            )
        if report.top_repeated_addresses:
            pd.DataFrame(report.top_repeated_addresses).to_csv(
                samples_dir / f"{label}_top_repeated_addresses.csv", index=False
            )
        if report.largest_normalized_name_blocks:
            pd.DataFrame(report.largest_normalized_name_blocks).to_csv(
                samples_dir / f"{label}_largest_normalized_name_blocks.csv", index=False
            )
    logger.info("Wrote CSV samples to %s", samples_dir)