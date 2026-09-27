import argparse
import os
import json
import pandas as pd
from src.blocking import Phase0Preprocessor, BlockingConfig, Phase1Blocker
from src.evaluate_blocking import BlockingEvaluator

def load_ground_truth(gt_path):
    gt_dict = {}
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    df = pd.read_csv(gt_path, sep='\t', header=None, names=['source1_entity_id', 'matched_entity_ids'])
    for _, row in df.iterrows():
        s1_id = str(row['source1_entity_id'])
        matches_raw = str(row['matched_entity_ids'])
        targets = set(str(m).strip() for m in matches_raw.split(',') if str(m).strip())
        gt_dict[s1_id] = targets
    return gt_dict

def main():
    parser = argparse.ArgumentParser(description="Run Amazon ML Challenge 2026 Blocking Sweeps")
    parser.add_argument("--data-root", type=str, required=True, help="Path to student resource data root")
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample size for S1")
    parser.add_argument("--sweep-k", type=str, default="1,2,3,4", help="Comma-separated k values")
    args = parser.parse_args()

    sweep_k_list = [int(x.strip()) for x in args.sweep_k.split(",")]
    strong_name_thresholds = [10, 25, 50, 100, 200]

    train_dir = os.path.join(args.data_root, "dataset", "train")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    for p in [s1_path, s2_path, s3_path, gt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required dataset file missing: {p}")

    print(f"Loading training data from {train_dir}...")
    s1_df = pd.read_csv(s1_path, sep='\t').head(args.sample_size)
    s2_df = pd.read_csv(s2_path, sep='\t')
    s3_df = pd.read_csv(s3_path, sep='\t')
    gt_dict = load_ground_truth(gt_path)

    # Dataset schema mapping: entity_id, business_name, business_address, country
    s1_df = s1_df.rename(columns={
        'entity_id': 'id',
        'business_name': 'name',
        'business_address': 'address'
    })
    s2_df = s2_df.rename(columns={
        'entity_id': 'id',
        'business_name': 'name',
        'business_address': 'address'
    })
    s3_df = s3_df.rename(columns={
        'entity_id': 'id',
        'business_name': 'name',
        'business_address': 'address'
    })
    s1_id_col = 'id'
    s2_id_col = 'id'
    s3_id_col = 'id'

    print("Preprocessing training corpus (Phase 0)...")
    corpus_for_fit = pd.concat([
        s2_df,
        s3_df
    ], ignore_index=True)

    preprocessor = Phase0Preprocessor()
    preprocessor.fit(corpus_for_fit)

    s1_records = [preprocessor.transform_record(row, id_col=s1_id_col) for _, row in s1_df.iterrows()]

    s2_ids = set(s2_df[s2_id_col].astype(str))
    s3_ids = set(s3_df[s3_id_col].astype(str))
    all_s1_ids = [str(rec['id']) for rec in s1_records]

    evaluator = BlockingEvaluator(gt_dict, s2_ids, s3_ids, all_s1_ids)

    print("\nBuilding retrieval inverted index ONCE...")
    base_config = BlockingConfig(strong_name_df_threshold=50, k=2, rare_token_mode="precision")
    base_config.freeze_train_stats(preprocessor)
    blocker = Phase1Blocker(base_config)
    blocker.build_index_from_df(corpus_for_fit)

    print("Pre-fetching raw candidate views for all S1 records...")

s1_raw_views_cache = {}

total_s1 = len(s1_records)

for i, s1_rec in enumerate(s1_records, start=1):

    s1_id = str(s1_rec['id'])

    s1_raw_views_cache[s1_id] = (
        blocker.get_raw_candidate_views(s1_rec)
    )

    if i % 250 == 0 or i == total_s1:

        print(
            f"  Prefetched {i:,}/{total_s1:,} "
            f"({i / total_s1 * 100:.1f}%)",
            flush=True
        )

    sweep_records = []

    # 1. Evaluate union_current_config (Raw Union)
    print("\nEvaluating union_current_config (Raw Union)...")
    results_union = {}
    for s1_rec in s1_records:
        s1_id_str = str(s1_rec['id'])
        results_union[s1_id_str] = blocker.get_union_candidates(s1_raw_views_cache[s1_id_str])
    metrics_union = evaluator.evaluate(results_union)
    
    sweep_records.append({
        "mode": "union_current_config",
        "k": 0,
        "strong_name_threshold": 0,
        **metrics_union
    })
    print(f"union_current_config -> ID Recall: {metrics_union['id_recall']:.4f} | Mean Cands: {metrics_union['mean_candidates']:.1f}")

    # 2. Sweeps: Precision and Diagnostic modes
    modes = ["precision", "diagnostic"]
    
    for mode in modes:
        for k in sweep_k_list:
            for s_thresh in strong_name_thresholds:
                cfg = BlockingConfig(strong_name_df_threshold=s_thresh, k=k, rare_token_mode=mode)
                cfg.freeze_train_stats(preprocessor)
                blk = Phase1Blocker(cfg)
                
                res = {}
                for s1_rec in s1_records:
                    s1_id_str = str(s1_rec['id'])
                    res[s1_id_str] = blk.evaluate_candidates(s1_raw_views_cache[s1_id_str], s1_rec)
                    
                met = evaluator.evaluate(res)
                sweep_records.append({
                    "mode": mode,
                    "k": k,
                    "strong_name_threshold": s_thresh,
                    **met
                })
                print(f"Mode: {mode} | k={k} | Threshold={s_thresh} --> ID Recall: {met['id_recall']:.4f} | Mean Cands: {met['mean_candidates']:.1f}")

    os.makedirs("artifacts", exist_ok=True)
    out_path = os.path.join("artifacts", "blocking_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(sweep_records, f, indent=2)
    print(f"\nSaved complete sweep results to {out_path}")

if __name__ == "__main__":
    main()