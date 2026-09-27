import numpy as np

class BlockingEvaluator:
    def __init__(self, ground_truth_dict, s2_ids, s3_ids, all_s1_ids):
        # Ensure all IDs are strings internally
        self.gt_dict = {str(k): set(str(t) for t in v) for k, v in ground_truth_dict.items()}
        self.s2_ids = set(str(x) for x in s2_ids)
        self.s3_ids = set(str(x) for x in s3_ids)
        self.all_s1_ids = [str(x) for x in all_s1_ids]

    def _compute_stats(self, counts):
        if not counts:
            return 0.0, 0.0, 0.0
        return float(np.mean(counts)), float(np.median(counts)), float(np.percentile(counts, 95))

    def evaluate(self, results_dict):
        total_gt_pairs = sum(len(targets) for targets in self.gt_dict.values())
        retrieved_gt_pairs = 0
        
        s2_gt_total = sum(1 for s1_id, targets in self.gt_dict.items() for t in targets if t in self.s2_ids)
        s3_gt_total = sum(1 for s1_id, targets in self.gt_dict.items() for t in targets if t in self.s3_ids)
        s2_retrieved = 0
        s3_retrieved = 0
        
        fully_covered_count = 0
        
        all_counts = []
        gt_counts = []
        zero_gt_counts = []
        
        singleton_counts = 0
        zero_candidate_counts = 0
        
        for s1_id in self.all_s1_ids:
            retrieved = set(str(c) for c in results_dict.get(s1_id, set()))
            gt_targets = self.gt_dict.get(s1_id, set())
            
            num_cands = len(retrieved)
            all_counts.append(num_cands)
            
            if gt_targets:
                gt_counts.append(num_cands)
            else:
                zero_gt_counts.append(num_cands)
            
            if num_cands == 1:
                singleton_counts += 1
            elif num_cands == 0:
                zero_candidate_counts += 1
                
            matched_targets = gt_targets.intersection(retrieved)
            retrieved_gt_pairs += len(matched_targets)
            
            for t in matched_targets:
                if t in self.s2_ids:
                    s2_retrieved += 1
                elif t in self.s3_ids:
                    s3_retrieved += 1
                    
            if gt_targets:
                if gt_targets.issubset(retrieved):
                    fully_covered_count += 1

        id_recall = (retrieved_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 0.0
        s2_recall = (s2_retrieved / s2_gt_total) if s2_gt_total > 0 else 0.0
        s3_recall = (s3_retrieved / s3_gt_total) if s3_gt_total > 0 else 0.0
        
        s1_with_gt = [s1 for s1 in self.all_s1_ids if self.gt_dict.get(s1)]
        fully_covered_rate = (fully_covered_count / len(s1_with_gt)) if s1_with_gt else 0.0
        
        mean_cands, median_cands, p95_cands = self._compute_stats(all_counts)
        mean_gt_cands, median_gt_cands, p95_gt_cands = self._compute_stats(gt_counts)
        mean_zero_cands, median_zero_cands, p95_zero_cands = self._compute_stats(zero_gt_counts)
        
        return {
            "id_recall": id_recall,
            "s2_recall": s2_recall,
            "s3_recall": s3_recall,
            "fully_covered_rate": fully_covered_rate,
            "mean_candidates": mean_cands,
            "median_candidates": median_cands,
            "p95_candidates": p95_cands,
            "mean_candidates_gt": mean_gt_cands,
            "median_candidates_gt": median_gt_cands,
            "p95_candidates_gt": p95_gt_cands,
            "mean_candidates_no_gt": mean_zero_cands,
            "median_candidates_no_gt": median_zero_cands,
            "p95_candidates_no_gt": p95_zero_cands,
            "singleton_count": singleton_counts,
            "zero_candidate_count": zero_candidate_counts
        }