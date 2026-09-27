#!/usr/bin/env python3
"""
run_hard_negatives.py
---------------------
Execution entrypoint for Hard Negatives + Cross-Source S2+S3 Consistency + Threshold Strategy.

Runs using the virtual environment python (.venv/Scripts/python.exe).

Usage:
  python run_hard_negatives.py --data-root "<path_to_dataset>" [--sample-size 5000]
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.path_discovery import discover_dataset
from src.hard_negatives import (
    normalize_string,
    TokenIDFCalculator,
    HardNegativeAnalyzer,
    CrossSourceConsistency,
    AdaptiveThresholdStrategy
)


def load_ground_truth(gt_path: str) -> dict:
    gt_dict = {}
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    df = pd.read_csv(gt_path, sep='\t', header=None, names=['source1_entity_id', 'matched_entity_ids'], dtype=str)
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna('')
    for s1_id, matches_raw in zip(df['source1_entity_id'], df['matched_entity_ids']):
        s1_id_str = str(s1_id).strip()
        targets = set(str(m).strip() for m in str(matches_raw).split(',') if str(m).strip())
        gt_dict[s1_id_str] = targets
    return gt_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hard Negatives, Cross-Source S2+S3 Consistency & Threshold Analysis")
    parser.add_argument("--data-root", type=str, default=None, help="Path to dataset root or zip file")
    parser.add_argument("--sample-size", type=int, default=2000, help="Sample size for Source1 analysis (0 for full dataset)")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    print("======================================================================")
    print(" HARD NEGATIVES + CROSS-SOURCE CONSISTENCY + THRESHOLD STRATEGY")
    print("======================================================================")

    # 1. Discover dataset
    dataset = discover_dataset(data_root=args.data_root)
    if not dataset.is_train_complete():
        print(f"Error: Dataset incomplete under {dataset.root}")
        print(dataset.summary())
        return 1

    nrows = None if (args.sample_size is None or args.sample_size <= 0) else args.sample_size
    mode_str = f"full dataset" if nrows is None else f"sample size: {nrows:,}"
    print(f"\n[1/4] Loading training dataset ({mode_str}) from: {dataset.root}")
    s1_df = pd.read_csv(dataset.train_source1, sep='\t', nrows=nrows)
    gt_dict = load_ground_truth(str(dataset.train_ground_truth))

    s1_ids = set(s1_df['entity_id'].astype(str))
    target_s2_ids = set()
    target_s3_ids = set()

    for s1_id in s1_ids:
        for target in gt_dict.get(s1_id, set()):
            if target.startswith("S2-"):
                target_s2_ids.add(target)
            elif target.startswith("S3-"):
                target_s3_ids.add(target)

    print(f"  Target S2 IDs needed: {len(target_s2_ids):,}, Target S3 IDs needed: {len(target_s3_ids):,}")

    print("  Streaming S2 TSV for target records...")
    s2_rows = []
    for chunk in pd.read_csv(dataset.train_source2, sep='\t', chunksize=100_000):
        matched_chunk = chunk[chunk['entity_id'].astype(str).isin(target_s2_ids)]
        if not matched_chunk.empty:
            s2_rows.append(matched_chunk)
        if sum(len(c) for c in s2_rows) >= len(target_s2_ids):
            break
    s2_df = pd.concat(s2_rows, ignore_index=True) if s2_rows else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])

    print("  Streaming S3 TSV for target records...")
    s3_rows = []
    for chunk in pd.read_csv(dataset.train_source3, sep='\t', chunksize=100_000):
        matched_chunk = chunk[chunk['entity_id'].astype(str).isin(target_s3_ids)]
        if not matched_chunk.empty:
            s3_rows.append(matched_chunk)
        if sum(len(c) for c in s3_rows) >= len(target_s3_ids):
            break
    s3_df = pd.concat(s3_rows, ignore_index=True) if s3_rows else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])

    print(f"  Loaded S1: {len(s1_df):,} rows, S2 Targets: {len(s2_df):,} rows, S3 Targets: {len(s3_df):,} rows")

    # Index S2 and S3 by entity_id
    s2_lookup = s2_df.set_index('entity_id').to_dict('index')
    s3_lookup = s3_df.set_index('entity_id').to_dict('index')

    # 2. Fit Token IDF Calculator
    print("\n[2/4] Fitting Token IDF Calculator across business names...")
    all_names = list(s1_df['business_name'].dropna()) + list(s2_df['business_name'].dropna()) + list(s3_df['business_name'].dropna())
    idf_calc = TokenIDFCalculator().fit(all_names)
    print(f"  Vocabulary size: {len(idf_calc.idf_cache):,} unique normalized tokens")

    hn_analyzer = HardNegativeAnalyzer(idf_calc)
    cross_cons = CrossSourceConsistency(idf_calc)
    threshold_strat = AdaptiveThresholdStrategy()

    print("\n[3/4] Running Hard Negative Analysis & Cross-Source Triangulation...")

    hard_negative_cases = []
    decision_summary = Counter()
    triangulation_summary = Counter()

    # Build candidate evaluation pairs (both GT targets and non-GT candidates for hard negative detection)
    all_s2_records = list(s2_lookup.values())

    
    for idx, s1_row in s1_df.iterrows():
        s1_id = str(s1_row['entity_id'])
        gt_targets = gt_dict.get(s1_id, set())

        s1_rec = {
            'id': s1_id,
            'name': s1_row.get('business_name', ''),
            'address': s1_row.get('business_address', ''),
            'country': s1_row.get('country', '')
        }

        s2_targets = [t for t in gt_targets if t.startswith('S2-')]
        s3_targets = [t for t in gt_targets if t.startswith('S3-')]

        # Evaluate Ground Truth Targets (True Matches)
        for s2_id in s2_targets:
            s2_rec = s2_lookup.get(s2_id)
            if not s2_rec:
                continue
            s2_rec['id'] = s2_id
            
            s3_rec = s3_lookup.get(s3_targets[0]) if s3_targets else None
            if s3_rec:
                s3_rec['id'] = s3_targets[0]

            an_res = hn_analyzer.analyze_pair(s1_rec, s2_rec)
            tri_res = cross_cons.evaluate_triangulation(s1_rec, s2_rec, s3_rec)
            accepted, tier, final_score = threshold_strat.decide(an_res, tri_res)

            decision_summary[f"GT_MATCH_{tier}"] += 1
            if tri_res["is_joint_pair"]:
                triangulation_summary["joint_s2_s3_pair"] += 1

            if an_res["is_hard_negative_risk"]:
                hard_negative_cases.append({
                    "type": "GT_HARD_CASE",
                    "s1_id": s1_id,
                    "target_id": s2_id,
                    "s1_name": s1_rec['name'],
                    "s1_address": s1_rec['address'],
                    "target_name": s2_rec.get('business_name', ''),
                    "target_address": s2_rec.get('business_address', ''),
                    "analysis": an_res,
                    "decision": tier
                })

        # Evaluate Hard Negative Non-GT Candidates (Surface-similar false matches)
        # Sample non-GT S2 candidates to identify hard negatives
        s1_tokens = set(normalize_string(s1_rec['name']).split())
        if s1_tokens:
            for cand_rec in all_s2_records[:100]:
                cand_id = str(cand_rec.get('entity_id', ''))
                if cand_id in gt_targets:
                    continue
                cand_tokens = set(normalize_string(cand_rec.get('business_name', '')).split())
                # If they share at least 1 token (surface similarity)
                if s1_tokens.intersection(cand_tokens):
                    an_res = hn_analyzer.analyze_pair(s1_rec, cand_rec)
                    tri_res = cross_cons.evaluate_triangulation(s1_rec, cand_rec, None)
                    accepted, tier, final_score = threshold_strat.decide(an_res, tri_res)
                    
                    decision_summary[f"NON_GT_{tier}"] += 1
                    
                    if an_res["is_hard_negative_risk"] or not accepted:
                        hard_negative_cases.append({
                            "type": "FALSE_MATCH_HARD_NEGATIVE",
                            "s1_id": s1_id,
                            "cand_id": cand_id,
                            "s1_name": s1_rec['name'],
                            "s1_address": s1_rec['address'],
                            "cand_name": cand_rec.get('business_name', ''),
                            "cand_address": cand_rec.get('business_address', ''),
                            "analysis": an_res,
                            "decision": tier
                        })

    print(f"\n[4/4] Generating Report & Decision Breakdown...")
    print("\n--- DECISION BREAKDOWN (Adaptive Threshold Strategy) ---")
    for tier, count in decision_summary.most_common():
        print(f"  {tier:35s}: {count:,}")

    print(f"\n--- HARD NEGATIVES IDENTIFIED ---")
    print(f"  Identified {len(hard_negative_cases):,} difficult false-match risk cases in sample.")

    os.makedirs(args.out_dir, exist_ok=True)
    report_file = os.path.join(args.out_dir, "hard_negatives_report.json")

    report_data = {
        "sample_size": len(s1_df),
        "total_hard_negatives_detected": len(hard_negative_cases),
        "decision_tier_counts": dict(decision_summary),
        "triangulation_counts": dict(triangulation_summary),
        "sample_hard_negative_cases": hard_negative_cases[:20]
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved report artifact to: {report_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
