"""
Hybrid & Ensemble Matcher implementing:
1. Multi-Matcher Consensus Ensemble ("ensemble" / "auto")
2. Standalone matchers: RootSIFT, LightGlue, RoMa, LoFTR, XoFTR
3. Intelligent Fallback & Escalation logic: RootSIFT -> LoFTR
"""
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2

from .common import MatcherResult, BaseMatcher
from .rootsift import RootSIFTMatcher, RootSIFTAdapter
from .loftr_matcher import LoFTRMatcher, LoFTRAdapter
from .lightglue_matcher import LightGlueMatcher
from .roma_matcher import RoMaMatcher
from .xoftr_matcher import XoFTRMatcher
from .ensemble import EnsembleMatcher
from .router import AdaptiveMatcherRouter


class HybridMatcher:
    """
    Intelligent Matcher supporting multi-matcher ensemble consensus,
    adaptive automatic routing, and classical/learned standalone modes.
    """
    def __init__(
        self,
        mode: str = "ensemble",  # "ensemble", "auto", "fallback", "rootsift_only", "lightglue_only", "roma_only", "loftr_only", "fusion"
        sift_nfeatures: int = 4000,
        sift_ratio: float = 0.75,
        loftr_conf_thresh: float = 0.25,
        fallback_min_candidates: int = 15,
        lightglue_features: str = "aliked",
        roma_model_type: str = "tiny"
    ):
        self.mode = mode.lower()
        self.fallback_min_candidates = fallback_min_candidates
        self.loftr_conf_thresh = loftr_conf_thresh
        self.sift_ratio = sift_ratio
        self.sift_nfeatures = sift_nfeatures
        self.lightglue_features = lightglue_features
        self.roma_model_type = roma_model_type

        # Classical RootSIFT instance (always lightweight)
        self.rootsift = RootSIFTMatcher(ratio_threshold=sift_ratio, nfeatures=sift_nfeatures)
        
        # Lazy instantiated adapters
        self._loftr = None
        self._lightglue = None
        self._roma = None
        self._ensemble = None
        self._router = AdaptiveMatcherRouter()

    def _get_loftr(self) -> LoFTRMatcher:
        if self._loftr is None:
            self._loftr = LoFTRMatcher(confidence_threshold=self.loftr_conf_thresh)
        return self._loftr

    def _get_lightglue(self) -> LightGlueMatcher:
        if self._lightglue is None:
            self._lightglue = LightGlueMatcher(features=self.lightglue_features)
        return self._lightglue

    def _get_roma(self) -> RoMaMatcher:
        if self._roma is None:
            self._roma = RoMaMatcher(model_type=self.roma_model_type)
        return self._roma

    def _get_ensemble(self) -> EnsembleMatcher:
        if self._ensemble is None:
            self._ensemble = EnsembleMatcher()
        return self._ensemble

    def match(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes matching according to configured mode.
        Returns:
            kp_ref, kp_src, matches, pts_ref, pts_src, meta
        """
        m = self.mode

        if m in ["ensemble", "multi_matcher", "consensus"]:
            ens = self._get_ensemble()
            res = ens.match(img_ref, img_src)
            kp_r, kp_s, dmatches, pts_r, pts_s = res.to_legacy_tuple()
            meta = {
                "matcher_used": "Multi-Matcher Consensus Ensemble",
                "escalated": False,
                **res.metadata
            }
            return kp_r, kp_s, dmatches, pts_r, pts_s, meta

        elif m in ["auto", "adaptive", "automatic"]:
            routed_matcher, route_diag = self._router.route(img_ref, img_src, requested_mode="auto")
            res = routed_matcher.match(img_ref, img_src)
            kp_r, kp_s, dmatches, pts_r, pts_s = res.to_legacy_tuple()
            meta = {
                "matcher_used": f"Adaptive Ensemble ({res.matcher_name})",
                "escalated": False,
                "routing_diag": route_diag,
                **res.metadata
            }
            return kp_r, kp_s, dmatches, pts_r, pts_s, meta

        elif m in ["lightglue_only", "lightglue"]:
            lg = self._get_lightglue()
            res = lg.match(img_ref, img_src)
            kp_r, kp_s, dmatches, pts_r, pts_s = res.to_legacy_tuple()
            meta = {"matcher_used": f"LightGlue ({self.lightglue_features.upper()})", "escalated": False, **res.metadata}
            return kp_r, kp_s, dmatches, pts_r, pts_s, meta

        elif m in ["roma_only", "roma"]:
            roma = self._get_roma()
            res = roma.match(img_ref, img_src)
            kp_r, kp_s, dmatches, pts_r, pts_s = res.to_legacy_tuple()
            meta = {"matcher_used": f"RoMa ({self.roma_model_type.capitalize()})", "escalated": False, **res.metadata}
            return kp_r, kp_s, dmatches, pts_r, pts_s, meta

        elif m in ["loftr_only", "loftr"]:
            kp_r, kp_s, dmatches, pts_r, pts_s = self._get_loftr().match(img_ref, img_src)
            meta = {"matcher_used": "LoFTR", "escalated": False}
            return kp_r, kp_s, dmatches, pts_r, pts_s, meta

        elif m in ["rootsift_only", "rootsift"]:
            kp_r, kp_s, dmatches, pts_r, pts_s = self.rootsift.match(img_ref, img_src)
            meta = {"matcher_used": "RootSIFT", "escalated": False}
            return kp_r, kp_s, dmatches, pts_r, pts_s, meta

        elif m == "fusion":
            # Classical + LoFTR Fusion
            kp_ref_s, kp_src_s, matches_s, pts_ref_s, pts_src_s = self.rootsift.match(img_ref, img_src)
            kp_ref_l, kp_src_l, matches_l, pts_ref_l, pts_src_l = self._get_loftr().match(img_ref, img_src)
            
            if len(pts_ref_s) == 0:
                pts_ref, pts_src = pts_ref_l, pts_src_l
                kp_ref, kp_src, matches = kp_ref_l, kp_src_l, matches_l
            elif len(pts_ref_l) == 0:
                pts_ref, pts_src = pts_ref_s, pts_src_s
                kp_ref, kp_src, matches = kp_ref_s, kp_src_s, matches_s
            else:
                pts_ref = np.vstack([pts_ref_s, pts_ref_l])
                pts_src = np.vstack([pts_src_s, pts_src_l])
                kp_ref = list(kp_ref_s) + list(kp_ref_l)
                kp_src = list(kp_src_s) + list(kp_src_l)
                matches = list(matches_s) + list(matches_l)
                
            meta = {
                "matcher_used": "Evidence Fusion (RootSIFT + LoFTR)",
                "sift_candidates": len(pts_ref_s),
                "loftr_candidates": len(pts_ref_l)
            }
            return kp_ref, kp_src, matches, pts_ref, pts_src, meta

        else:  # Default: "fallback"
            # 1. Try RootSIFT first (fast)
            kp_ref, kp_src, matches, pts_ref, pts_src = self.rootsift.match(img_ref, img_src)
            
            # 2. Check candidate adequacy
            if len(pts_ref) < self.fallback_min_candidates:
                # Escalate to LoFTR deep matcher on-demand
                loftr = self._get_loftr()
                kp_ref_l, kp_src_l, matches_l, pts_ref_l, pts_src_l = loftr.match(img_ref, img_src)
                if len(pts_ref_l) > len(pts_ref):
                    meta = {
                        "matcher_used": "LoFTR (Fallback Escalated)",
                        "escalated": True,
                        "initial_sift_candidates": len(pts_ref),
                        "loftr_candidates": len(pts_ref_l)
                    }
                    return kp_ref_l, kp_src_l, matches_l, pts_ref_l, pts_src_l, meta
                    
            meta = {
                "matcher_used": "RootSIFT (Primary)",
                "escalated": False,
                "sift_candidates": len(pts_ref)
            }
            return kp_ref, kp_src, matches, pts_ref, pts_src, meta
