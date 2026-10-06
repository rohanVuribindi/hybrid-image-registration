"""
Multi-Matcher Consensus Ensemble for ZENITH.
Implements multi-model correspondence pooling, cross-model spatial consensus,
contradiction pruning, and confidence-weighted aggregation.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
import cv2

from .common import MatcherResult, BaseMatcher
from .rootsift import RootSIFTAdapter
from .lightglue_matcher import LightGlueMatcher
from .roma_matcher import RoMaMatcher
from .loftr_matcher import LoFTRAdapter
from .xoftr_matcher import XoFTRMatcher


class EnsembleMatcher(BaseMatcher):
    """
    Multi-Matcher Consensus Engine.
    Executes multiple diverse matchers (classical + deep sparse + deep dense) and
    fuses candidate correspondences using cross-model spatial verification.
    """
    def __init__(
        self,
        matchers: Optional[List[BaseMatcher]] = None,
        consensus_radius_px: float = 3.0,
        max_pool_candidates: int = 2500,
        grid_bins: Tuple[int, int] = (16, 16),
        max_pts_per_bin: int = 15
    ):
        super().__init__(matcher_name="MultiMatcherEnsemble")
        self.consensus_radius_px = consensus_radius_px
        self.max_pool_candidates = max_pool_candidates
        self.grid_bins = grid_bins
        self.max_pts_per_bin = max_pts_per_bin
        
        if matchers is None:
            # Default ensemble suite: RootSIFT + LightGlue (ALIKED) + RoMa (Dense) + LoFTR
            self.matchers = [
                RootSIFTAdapter(ratio_threshold=0.75, nfeatures=3000),
                LightGlueMatcher(features="aliked", max_num_keypoints=1200),
                RoMaMatcher(model_type="tiny", sample_pts=1000, certainty_threshold=0.15),
                LoFTRAdapter(confidence_threshold=0.25)
            ]
        else:
            self.matchers = matchers

    def match(self, img_ref: np.ndarray, img_src: np.ndarray) -> MatcherResult:
        """
        Runs all configured matchers, performs cross-model spatial agreement analysis,
        and returns a consensus-weighted MatcherResult.
        """
        results: List[MatcherResult] = []
        timings: Dict[str, float] = {}
        matcher_counts: Dict[str, int] = {}

        for m in self.matchers:
            import time
            t0 = time.perf_counter()
            try:
                res = m.match(img_ref, img_src)
                if not res.is_empty:
                    results.append(res)
                    matcher_counts[res.matcher_name] = len(res)
                else:
                    matcher_counts[m.matcher_name] = 0
            except Exception as e:
                matcher_counts[m.matcher_name] = 0
            timings[m.matcher_name] = round(time.perf_counter() - t0, 4)

        if len(results) == 0:
            return MatcherResult(
                matcher_name=self.matcher_name,
                metadata={
                    "active_matchers": [m.matcher_name for m in self.matchers],
                    "raw_candidates_per_matcher": matcher_counts,
                    "timings": timings,
                    "consensus_rate": 0.0
                }
            )

        # Single matcher fallback if only one succeeded
        if len(results) == 1:
            res = results[0]
            res.metadata["active_matchers"] = [res.matcher_name]
            res.metadata["raw_candidates_per_matcher"] = matcher_counts
            res.metadata["timings"] = timings
            res.metadata["consensus_rate"] = 1.0
            return res

        # ---------------------------------------------------------------------
        # Multi-Matcher Consensus & Spatial Agreement Analysis
        # ---------------------------------------------------------------------
        fused_src = []
        fused_ref = []
        fused_conf = []
        fused_sources = []

        # Build KD-trees for each active model's source points
        trees_src: Dict[int, cKDTree] = {}
        for idx, r in enumerate(results):
            if len(r.source_points) > 0:
                trees_src[idx] = cKDTree(r.source_points)

        total_tested = 0
        total_agreed = 0

        # Evaluate each point in every model against all other models
        for m_idx, r in enumerate(results):
            pts_s = r.source_points
            pts_r = r.reference_points
            confs = r.confidence
            m_name = r.matcher_name

            other_indices = [idx for idx in trees_src.keys() if idx != m_idx]
            num_others = len(other_indices)

            for i in range(len(pts_s)):
                p_s = pts_s[i]
                p_r = pts_r[i]
                c_orig = float(confs[i])
                total_tested += 1

                agreements = 0
                contradictions = 0

                for o_idx in other_indices:
                    tree = trees_src[o_idx]
                    other_pts_r = results[o_idx].reference_points
                    other_pts_s = results[o_idx].source_points

                    # Query nearest neighbor in other model's source space
                    d_s, nn_idx = tree.query(p_s, k=1)
                    if d_s <= self.consensus_radius_px:
                        # Found nearby source point, now inspect corresponding reference point
                        matched_r = other_pts_r[nn_idx]
                        d_r = np.linalg.norm(p_r - matched_r)
                        if d_r <= self.consensus_radius_px:
                            agreements += 1
                        elif d_r > 2.5 * self.consensus_radius_px:
                            contradictions += 1

                if agreements > 0:
                    total_agreed += 1

                # Deterministic consensus formula
                agree_ratio = agreements / max(num_others, 1)
                contra_ratio = contradictions / max(num_others, 1)
                
                # Boost confidence for multi-model agreement, penalize contradictions
                c_final = 0.65 * c_orig + 0.35 * agree_ratio - 0.25 * contra_ratio
                c_final = float(np.clip(c_final, 0.05, 1.0))

                fused_src.append(p_s)
                fused_ref.append(p_r)
                fused_conf.append(c_final)
                fused_sources.append(m_name)

        fused_src = np.array(fused_src, dtype=np.float32)
        fused_ref = np.array(fused_ref, dtype=np.float32)
        fused_conf = np.array(fused_conf, dtype=np.float32)

        # ---------------------------------------------------------------------
        # Spatial Grid Non-Maximum Suppression / Diversity Balancing
        # ---------------------------------------------------------------------
        h_ref, w_ref = img_ref.shape[:2]
        ny, nx = self.grid_bins
        bin_h = max(h_ref / ny, 1.0)
        bin_w = max(w_ref / nx, 1.0)

        grid_buckets: Dict[Tuple[int, int], List[int]] = {}
        for idx in range(len(fused_ref)):
            rx, ry = fused_ref[idx]
            bx = int(np.clip(rx // bin_w, 0, nx - 1))
            by = int(np.clip(ry // bin_h, 0, ny - 1))
            key = (by, bx)
            if key not in grid_buckets:
                grid_buckets[key] = []
            grid_buckets[key].append(idx)

        selected_indices = []
        for key, bucket in grid_buckets.items():
            # Sort points in this cell by confidence descending
            bucket.sort(key=lambda idx: fused_conf[idx], reverse=True)
            # Pick top distinct points
            picked_cell = []
            for b_idx in bucket:
                cand_ref = fused_ref[b_idx]
                # Check spatial separation within cell (e.g. >= 2.0 px)
                too_close = False
                for p_idx in picked_cell:
                    if np.linalg.norm(cand_ref - fused_ref[p_idx]) < 2.0:
                        too_close = True
                        break
                if not too_close:
                    picked_cell.append(b_idx)
                if len(picked_cell) >= self.max_pts_per_bin:
                    break
            selected_indices.extend(picked_cell)

        selected_indices = np.array(selected_indices, dtype=np.int32)
        if len(selected_indices) > self.max_pool_candidates:
            # Sort globally by confidence and cap
            order = np.argsort(fused_conf[selected_indices])[::-1]
            selected_indices = selected_indices[order[:self.max_pool_candidates]]

        final_src = fused_src[selected_indices]
        final_ref = fused_ref[selected_indices]
        final_conf = fused_conf[selected_indices]

        consensus_rate = (total_agreed / max(total_tested, 1)) * 100.0

        return MatcherResult(
            source_points=final_src,
            reference_points=final_ref,
            confidence=final_conf,
            matcher_name="Ensemble_Consensus",
            metadata={
                "active_matchers": [r.matcher_name for r in results],
                "raw_candidates_per_matcher": matcher_counts,
                "total_candidates_pooled": len(fused_src),
                "balanced_pool_size": len(final_src),
                "consensus_agreement_rate_pct": round(consensus_rate, 2),
                "timings": timings
            }
        )

    def match_legacy(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray, np.ndarray]:
        res = self.match(img_ref, img_src)
        return res.to_legacy_tuple()
