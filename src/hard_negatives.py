"""
src/hard_negatives.py
---------------------
Hard Negatives Identification, Cross-Source S2+S3 Consistency Logic,
and Adaptive Threshold Strategy for Amazon ML Challenge (Business Entity Resolution).

Module capabilities:
1. Difficult False-Match (Hard Negative) Identification:
   - Generic business name token frequency (IDF analysis)
   - Address numeric mismatch detection (house/suite numbers)
   - Cross-country / regional mismatches
   - Coincidental token overlap filtering

2. Cross-Source S2+S3 Consistency Logic:
   - Joint S2 and S3 candidate pair evaluation
   - Triangulation agreement: Sim(S2_cand, S3_cand)
   - Multi-source conflict resolution & deduplication

3. Adaptive Threshold Strategy:
   - Multi-tier confidence scoring (Tier 1: Exact/Strong, Tier 2: Address-Validated, Tier 3: Triangulated)
   - Hard drop rules (Address numeric collision, Country mismatch, Low-IDF generic match)
"""

from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd


def normalize_string(text: Any) -> str:
    """Standardized text normalization."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_numbers(text: Any) -> Set[str]:
    """Extract numeric sequences (e.g. house numbers, street numbers, zip codes)."""
    if pd.isna(text) or text is None:
        return set()
    return set(re.findall(r'\b\d+\b', str(text)))


class TokenIDFCalculator:
    """Computes Inverse Document Frequency (IDF) for tokens across business names."""
    
    def __init__(self, smooth_idf: bool = True):
        self.doc_freq = Counter()
        self.total_docs = 0
        self.smooth_idf = smooth_idf
        self.idf_cache: Dict[str, float] = {}

    def fit(self, names: List[str]) -> TokenIDFCalculator:
        self.total_docs = len(names)
        for name in names:
            tokens = set(normalize_string(name).split())
            self.doc_freq.update(tokens)
            
        n_docs = self.total_docs if not self.smooth_idf else self.total_docs + 1
        for token, df in self.doc_freq.items():
            df_val = df if not self.smooth_idf else df + 1
            self.idf_cache[token] = math.log(n_docs / df_val) + 1.0
        return self

    def get_idf(self, token: str) -> float:
        if token in self.idf_cache:
            return self.idf_cache[token]
        # Out-of-vocabulary rare token gets max IDF
        default_df = 1 if not self.smooth_idf else 2
        n_docs = max(self.total_docs, 1) + (1 if self.smooth_idf else 0)
        return math.log(n_docs / default_df) + 1.0

    def compute_idf_weighted_jaccard(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        """Weighted Jaccard similarity using token IDF weights."""
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        inter_weight = sum(self.get_idf(t) for t in intersection)
        union_weight = sum(self.get_idf(t) for t in union)
        
        return inter_weight / union_weight if union_weight > 0 else 0.0


class HardNegativeAnalyzer:
    """Identifies and analyzes difficult false-match cases (Hard Negatives)."""
    
    def __init__(self, idf_calc: TokenIDFCalculator):
        self.idf_calc = idf_calc

    def analyze_pair(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyzes a candidate pair and identifies hard negative risk factors."""
        name1 = normalize_string(rec1.get('name', ''))
        name2 = normalize_string(rec2.get('name', ''))
        
        addr1 = normalize_string(rec1.get('address', ''))
        addr2 = normalize_string(rec2.get('address', ''))
        
        country1 = str(rec1.get('country', '')).strip().upper()
        country2 = str(rec2.get('country', '')).strip().upper()
        
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())
        
        shared_tokens = tokens1.intersection(tokens2)
        
        nums1 = extract_numbers(rec1.get('address', ''))
        nums2 = extract_numbers(rec2.get('address', ''))
        
        # 1. Address Numeric Mismatch Check
        # If both addresses have numbers but share NO numbers -> Hard Negative Risk
        has_numeric_mismatch = bool(nums1 and nums2 and not (nums1.intersection(nums2)))
        
        # 2. Low-IDF Generic Match Check
        # High token overlap but max IDF among shared tokens is very low -> Generic Business Name
        max_shared_idf = max([self.idf_calc.get_idf(t) for t in shared_tokens], default=0.0)
        is_generic_name_match = (len(shared_tokens) > 0 and max_shared_idf < 3.0)
        
        # 3. Country Mismatch
        has_country_mismatch = bool(country1 and country2 and country1 != country2)
        
        # 4. Weighted Similarity
        weighted_sim = self.idf_calc.compute_idf_weighted_jaccard(tokens1, tokens2)
        
        # Flag as Hard Negative Risk if similarities are superficially high but discriminator flags trip
        is_hard_negative_risk = (
            (weighted_sim > 0.4) and (has_numeric_mismatch or is_generic_name_match or has_country_mismatch)
        )
        
        return {
            "weighted_sim": round(weighted_sim, 4),
            "shared_tokens": list(shared_tokens),
            "max_shared_idf": round(max_shared_idf, 3),
            "has_numeric_mismatch": has_numeric_mismatch,
            "is_generic_name_match": is_generic_name_match,
            "has_country_mismatch": has_country_mismatch,
            "is_hard_negative_risk": is_hard_negative_risk,
            "risk_reasons": [
                r for r, flag in [
                    ("numeric_address_mismatch", has_numeric_mismatch),
                    ("generic_low_idf_match", is_generic_name_match),
                    ("country_mismatch", has_country_mismatch)
                ] if flag
            ]
        }


class CrossSourceConsistency:
    """Designs logic for cross-source S2+S3 joint consistency and conflict resolution."""

    def __init__(self, idf_calc: TokenIDFCalculator):
        self.idf_calc = idf_calc

    def evaluate_triangulation(
        self,
        s1_rec: Dict[str, Any],
        s2_rec: Optional[Dict[str, Any]],
        s3_rec: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates 3-way consistency between S1, S2, and S3 records."""
        tokens1 = set(normalize_string(s1_rec.get('name', '')).split())
        
        sim_s1_s2 = 0.0
        sim_s1_s3 = 0.0
        sim_s2_s3 = 0.0
        
        if s2_rec:
            tokens2 = set(normalize_string(s2_rec.get('name', '')).split())
            sim_s1_s2 = self.idf_calc.compute_idf_weighted_jaccard(tokens1, tokens2)
            
        if s3_rec:
            tokens3 = set(normalize_string(s3_rec.get('name', '')).split())
            sim_s1_s3 = self.idf_calc.compute_idf_weighted_jaccard(tokens1, tokens3)
            
        if s2_rec and s3_rec:
            tokens2 = set(normalize_string(s2_rec.get('name', '')).split())
            tokens3 = set(normalize_string(s3_rec.get('name', '')).split())
            sim_s2_s3 = self.idf_calc.compute_idf_weighted_jaccard(tokens2, tokens3)
            
        # Triangulated score boosts overall confidence if S2 and S3 also agree with each other
        is_joint_pair = bool(s2_rec and s3_rec)
        joint_confidence = (
            (sim_s1_s2 + sim_s1_s3 + sim_s2_s3) / 3.0 if is_joint_pair else max(sim_s1_s2, sim_s1_s3)
        )
        
        return {
            "sim_s1_s2": round(sim_s1_s2, 4),
            "sim_s1_s3": round(sim_s1_s3, 4),
            "sim_s2_s3": round(sim_s2_s3, 4),
            "is_joint_pair": is_joint_pair,
            "triangulated_confidence": round(joint_confidence, 4)
        }


class AdaptiveThresholdStrategy:
    """Executes multi-tier dynamic thresholding and hard drop rules."""

    def __init__(
        self,
        tier1_exact_thresh: float = 0.85,
        tier2_fuzzy_thresh: float = 0.65,
        tier3_triangulated_thresh: float = 0.55
    ):
        self.tier1_thresh = tier1_exact_thresh
        self.tier2_thresh = tier2_fuzzy_thresh
        self.tier3_thresh = tier3_triangulated_thresh

    def decide(
        self,
        analysis_result: Dict[str, Any],
        triangulation_result: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        """Returns (accepted: bool, decision_tier: str, final_score: float)."""
        score = analysis_result["weighted_sim"]
        
        # Hard Drop Filter Rules
        if analysis_result["has_country_mismatch"]:
            return False, "REJECT_COUNTRY_MISMATCH", 0.0
            
        if analysis_result["has_numeric_mismatch"] and score < 0.90:
            return False, "REJECT_NUMERIC_ADDRESS_MISMATCH", score
            
        if analysis_result["is_generic_name_match"] and score < 0.80:
            return False, "REJECT_GENERIC_LOW_IDF", score

        # Multi-Tier Admission Strategy
        # Tier 1: High Similarity / Exact
        if score >= self.tier1_thresh:
            return True, "TIER1_HIGH_CONFIDENCE", score

        # Tier 2: Medium Similarity + Address Validated (No numeric mismatch)
        if score >= self.tier2_thresh and not analysis_result["has_numeric_mismatch"]:
            return True, "TIER2_ADDRESS_VALIDATED", score

        # Tier 3: Triangulated Consistency (S2 and S3 mutually reinforce)
        tri_score = triangulation_result.get("triangulated_confidence", 0.0)
        if triangulation_result.get("is_joint_pair") and tri_score >= self.tier3_thresh:
            return True, "TIER3_TRIANGULATED_CONSISTENT", tri_score

        return False, "REJECT_BELOW_THRESHOLDS", score
