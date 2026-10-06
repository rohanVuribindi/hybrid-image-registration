"""
Adaptive Matcher Routing for ZENITH.
Analyzes image pair characteristics (texture, contrast, feature density, scale gap)
and dynamically routes to the most effective matcher combination with explainable diagnostics.
"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

from .common import BaseMatcher
from .rootsift import RootSIFTAdapter
from .lightglue_matcher import LightGlueMatcher
from .roma_matcher import RoMaMatcher
from .loftr_matcher import LoFTRAdapter
from .xoftr_matcher import XoFTRMatcher
from .ensemble import EnsembleMatcher


def analyze_image_pair_characteristics(img_ref: np.ndarray, img_src: np.ndarray) -> Dict[str, Any]:
    """
    Measures quantifiable physical and radiometric properties of the image pair:
    - Dynamic range & contrast ratio
    - Gradient energy & texture entropy
    - FAST corner density
    - Scale ratio
    """
    h_r, w_r = img_ref.shape[:2]
    h_s, w_s = img_src.shape[:2]
    
    gray_r = img_ref if img_ref.ndim == 2 else cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_s = img_src if img_src.ndim == 2 else cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # 1. Contrast & intensity statistics
    std_r = float(np.std(gray_r))
    std_s = float(np.std(gray_s))
    contrast_ratio = max(std_r / max(std_s, 1e-5), std_s / max(std_r, 1e-5))

    # 2. Gradient energy (Sobel)
    gx_r = cv2.Sobel(gray_r, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(gray_r, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_r = float(np.mean(np.sqrt(gx_r**2 + gy_r**2)))

    # 3. Fast feature density
    fast = cv2.FastFeatureDetector_create(threshold=25)
    kps_r = fast.detect(gray_r, None)
    kps_s = fast.detect(gray_s, None)
    kp_density_r = len(kps_r) / (h_r * w_r / 10000.0)
    kp_density_s = len(kps_s) / (h_s * w_s / 10000.0)

    # 4. Aspect / Dimension Ratio
    dim_ratio = max((h_r * w_r) / max(h_s * w_s, 1), (h_s * w_s) / max(h_r * w_r, 1))

    # Classification
    is_low_texture = (grad_mag_r < 12.0) or (kp_density_r < 15.0 and kp_density_s < 15.0)
    is_high_contrast_diff = (contrast_ratio > 1.8)
    is_large_scale_gap = (dim_ratio > 2.0)

    category = "NOMINAL_TEXTURE"
    if is_low_texture:
        category = "LOW_TEXTURE_SMOOTH"
    elif is_large_scale_gap or is_high_contrast_diff:
        category = "CHALLENGING_WARP_OR_ILLUMINATION"

    return {
        "contrast_ref_std": round(std_r, 2),
        "contrast_src_std": round(std_s, 2),
        "contrast_ratio": round(contrast_ratio, 2),
        "gradient_energy_ref": round(grad_mag_r, 2),
        "kp_count_ref": len(kps_r),
        "kp_count_src": len(kps_s),
        "kp_density_ref_per_10k_px": round(kp_density_r, 2),
        "dimension_ratio": round(dim_ratio, 2),
        "is_low_texture": is_low_texture,
        "is_high_contrast_diff": is_high_contrast_diff,
        "is_large_scale_gap": is_large_scale_gap,
        "recommended_category": category
    }


class AdaptiveMatcherRouter:
    """
    Intelligent routing engine selecting optimal matchers based on pair diagnostics.
    """
    def __init__(self, default_mode: str = "ensemble"):
        self.default_mode = default_mode

    def route(self, img_ref: np.ndarray, img_src: np.ndarray, requested_mode: str = "auto") -> Tuple[BaseMatcher, Dict[str, Any]]:
        """
        Determines and instantiates the optimal matcher.
        """
        diag = analyze_image_pair_characteristics(img_ref, img_src)
        
        req = requested_mode.lower()
        if req == "rootsift_only" or req == "rootsift":
            matcher = RootSIFTAdapter(ratio_threshold=0.75, nfeatures=4000)
            diag["routing_decision"] = "Explicit RootSIFT Selection"
            return matcher, diag

        elif req == "lightglue_only" or req == "lightglue":
            matcher = LightGlueMatcher(features="aliked", max_num_keypoints=1536)
            diag["routing_decision"] = "Explicit LightGlue Selection"
            return matcher, diag

        elif req == "roma_only" or req == "roma":
            matcher = RoMaMatcher(model_type="tiny", sample_pts=1200, certainty_threshold=0.15)
            diag["routing_decision"] = "Explicit RoMa Selection"
            return matcher, diag

        elif req == "loftr_only" or req == "loftr":
            matcher = LoFTRAdapter(confidence_threshold=0.25)
            diag["routing_decision"] = "Explicit LoFTR Selection"
            return matcher, diag

        # Automatic Ensemble Routing
        category = diag["recommended_category"]
        if category == "NOMINAL_TEXTURE":
            # High-speed classical + learned sparse consensus
            matchers = [
                RootSIFTAdapter(ratio_threshold=0.75, nfeatures=3000),
                LightGlueMatcher(features="aliked", max_num_keypoints=1200),
                RoMaMatcher(model_type="tiny", sample_pts=800, certainty_threshold=0.18)
            ]
            diag["routing_decision"] = "Ensemble: RootSIFT + LightGlue(ALIKED) + RoMa (Nominal High Texture)"
            return EnsembleMatcher(matchers=matchers), diag

        elif category == "LOW_TEXTURE_SMOOTH":
            # Dense matching emphasis (RoMa + LoFTR + LightGlue)
            matchers = [
                RoMaMatcher(model_type="tiny", sample_pts=1500, certainty_threshold=0.12),
                LoFTRAdapter(confidence_threshold=0.20),
                LightGlueMatcher(features="superpoint", max_num_keypoints=1200)
            ]
            diag["routing_decision"] = "Ensemble: RoMa (Dense) + LoFTR (Transformer) + SuperPoint/LightGlue (Low Texture Mare)"
            return EnsembleMatcher(matchers=matchers), diag

        else: # CHALLENGING_WARP_OR_ILLUMINATION / DEFAULT FULL ENSEMBLE
            matchers = [
                RoMaMatcher(model_type="tiny", sample_pts=1200, certainty_threshold=0.15),
                LightGlueMatcher(features="aliked", max_num_keypoints=1200),
                LoFTRAdapter(confidence_threshold=0.22),
                RootSIFTAdapter(ratio_threshold=0.78, nfeatures=3000)
            ]
            diag["routing_decision"] = "Full Consensus Ensemble: RoMa + LightGlue + LoFTR + RootSIFT"
            return EnsembleMatcher(matchers=matchers), diag
