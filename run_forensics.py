#!/usr/bin/env python3
"""
run_forensics.py
-----------------
Phase 0 entry point for the Amazon ML Challenge 2026 (Business Entity
Resolution). Run this first, before writing any blocking/matching code.

What it does:
  1. Auto-discovers your dataset (an extracted folder, or a .zip anywhere
     under common download locations) -- or use --data-root to point at it
     directly if auto-discovery does not find it.
  2. Streams through train_source1/2/3.tsv and train_ground_truth.tsv in
     fixed-size chunks (never loading a whole large file into memory).
  3. Prints a human-readable summary to the console as it runs.
  4. Saves a full JSON report plus a few CSV samples for manual review under
     the output directory (default: ./artifacts).

Usage (from Windows PowerShell or cmd, after `pip install -r requirements.txt`):

    python run_forensics.py
    python run_forensics.py --data-root "C:\\Users\\you\\Downloads\\student_resource"
    python run_forensics.py --data-root "C:\\Users\\you\\Downloads\\student_resource.zip"
    python run_forensics.py --search-root "D:\\Hackathon" --chunksize 50000
    python run_forensics.py --out-dir .\\artifacts --top-n 30

Nothing in this script (or the modules it imports) requires or performs any
external/network lookup -- it only reads the local files you point it at.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running this script directly (python run_forensics.py) without having
# the project installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import forensics, path_discovery  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Explicit path to the extracted dataset folder OR a .zip file. "
        "If omitted, auto-discovery searches --search-root locations.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="Folder to search for the dataset/zip (repeatable). "
        "Defaults to Downloads, Desktop, Documents and the current directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts",
        help="Where to write forensics_report.json and samples/ (default: ./artifacts)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per chunk when streaming large TSVs (default: 100000). "
        "Lower this if you hit memory pressure.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many top repeated names/addresses/blocks to report (default: 20).",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip checking for test_source*.tsv presence (useful on day 1 before test data drops).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def setup_logging(out_dir: Path, verbose: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "forensics_run.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    setup_logging(out_dir, args.verbose)
    logger = logging.getLogger("ber.run_forensics")

    print_section("STEP 1 / 4 — Locating dataset")
    dataset = path_discovery.discover_dataset(data_root=args.data_root, search_roots=args.search_root)
    print(dataset.summary())

    if not dataset.is_train_complete():
        logger.error(
            "Training files are incomplete -- cannot run forensics. "
            "Pass --data-root explicitly, e.g.:\n"
            '  python run_forensics.py --data-root "C:\\path\\to\\student_resource"'
        )
        return 1

    if not args.skip_test and not dataset.is_test_complete():
        logger.warning(
            "Test files not found (this is expected on Day 1 before test data drops). "
            "Continuing with training-file forensics only."
        )

    print_section("STEP 2 / 4 — Streaming source-file forensics (this can take a while on large files)")
    source_reports = {}
    for label, path in [
        ("source1", dataset.train_source1),
        ("source2", dataset.train_source2),
        ("source3", dataset.train_source3),
    ]:
        print(f"\n--- Analyzing {label}: {path} ---")
        approx_rows = path_discovery and None  # placeholder to keep lints quiet
        try:
            row_estimate = None
            row_estimate = _safe_count_rows(path)
        except Exception:  # noqa: BLE001 - best-effort progress hint only
            row_estimate = None
        if row_estimate is not None:
            print(f"(approx {row_estimate:,} data rows to process)")

        report = forensics.analyze_source_file(path, label, chunksize=args.chunksize, top_n=args.top_n)
        source_reports[label] = report
        _print_source_report(report)

    print_section("STEP 3 / 4 — Ground truth analysis")
    gt_report = forensics.analyze_ground_truth(dataset.train_ground_truth, top_n=args.top_n)
    _print_ground_truth_report(gt_report)

    print_section("STEP 4 / 4 — Writing artifacts")
    full_report = {
        "dataset_root": str(dataset.root),
        "source_reports": {k: _dataclass_to_dict(v) for k, v in source_reports.items()},
        "ground_truth_report": _dataclass_to_dict(gt_report),
    }
    report_path = forensics.write_report(full_report, out_dir)
    forensics.write_samples(source_reports, out_dir)

    print(f"\nDone. Full JSON report: {report_path}")
    print(f"CSV samples:            {out_dir / 'samples'}")
    print(f"Run log:                {out_dir / 'forensics_run.log'}")
    return 0


def _safe_count_rows(path: Path):
    from src import tsv_utils

    return tsv_utils.count_rows_fast(path)


def _dataclass_to_dict(obj):
    from dataclasses import asdict, is_dataclass

    return asdict(obj) if is_dataclass(obj) else obj


def _print_source_report(report) -> None:
    print(f"Total rows:                  {report.total_rows:,}")
    if report.missing_columns:
        print(f"MISSING COLUMNS:            {report.missing_columns} -- analysis skipped for this file")
        return
    print(f"ID prefix mismatches:        {report.id_prefix_mismatches:,}")
    print(f"Duplicate entity_id rows:    {report.duplicate_entity_ids:,}")
    print(f"Exact duplicate records:     {report.exact_duplicate_records:,}")
    print(f"Empty name / address / country: {report.empty_name_count:,} / {report.empty_address_count:,} / {report.empty_country_count:,}")
    print(f"Country distribution:        {report.country_distribution}")
    print(f"Name length (chars):         {report.name_length_chars}")
    print(f"Address length (chars):      {report.address_length_chars}")
    print(f"Name non-ASCII ratio (mean): {report.name_non_ascii_ratio_mean}")
    print(f"Address non-ASCII ratio (mean): {report.address_non_ascii_ratio_mean}")
    print(f"Legal-suffix token hits:     {report.legal_suffix_hits}")
    print(f"Address-abbreviation hits:   {report.address_abbrev_hits}")
    print(f"Normalized-name block-size histogram (size -> #groups): {report.normalized_name_block_size_histogram}")
    if report.largest_normalized_name_blocks:
        print("Largest normalized-name blocks (collision risk if used as sole blocking key):")
        for row in report.largest_normalized_name_blocks[:5]:
            print(f"    {row}")


def _print_ground_truth_report(report) -> None:
    print(f"Total Source-1 rows:               {report.total_source1_rows:,}")
    print(f"Duplicate source1_entity_id rows:  {report.duplicate_source1_rows:,}")
    print(f"Singletons (no match):             {report.singleton_count:,}")
    print(f"Match-count histogram:              {report.match_count_histogram}")
    print(f"Total matched IDs:                  {report.total_matched_ids:,}")
    print(f"  -> to Source 2:                   {report.matched_to_source2:,}")
    print(f"  -> to Source 3:                    {report.matched_to_source3:,}")
    print(f"Rows with duplicate IDs in one list: {report.duplicate_ids_within_a_row:,}")
    print(f"S2/S3 IDs matched to >1 Source-1:    {report.s2_s3_ids_matched_to_multiple_s1:,}")
    if report.top_multiply_matched_ids:
        print("Top multiply-matched S2/S3 IDs (potential many-to-one behavior):")
        for row in report.top_multiply_matched_ids[:5]:
            print(f"    {row}")


if __name__ == "__main__":
    raise SystemExit(main())