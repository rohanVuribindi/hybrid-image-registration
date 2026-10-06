"""
RoMa (Robust Dense Feature Matcher) Matcher & Adapter for ZENITH.
Integrates SOTA dense matching via RoMa (TinyRoMa / RoMa Outdoor).
Provides lazy-loading, certainty-based sampling, and universal MatcherResult contract.
"""
from typing import Tuple, List, Dict, Any, Optional
import os
import warnings
import numpy as np
import cv2
import torch
import PIL.Image

from .common import BaseMatcher, MatcherResult


class RoMaMatcher(BaseMatcher):
    """
    RoMa dense warp-field matcher for difficult viewpoint, illumination, and texture-poor pairs.
    """
    def __init__(
        self,
        model_type: str = "tiny",  # "tiny" (TinyRoMa v1, fast/CPU) or "outdoor" (Full RoMa)
        sample_pts: int = 1200,
        certainty_threshold: float = 0.15,
        device: Optional[str] = None
    ):
        super().__init__(matcher_name=f"RoMa_{model_type.upper()}")
        self.model_type = model_type.lower()
        self.sample_pts = sample_pts
        self.certainty_threshold = certainty_threshold
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = None
        self._is_loaded = False

    def _ensure_loaded(self):
        """Loads RoMa model weights on demand."""
        if not self._is_loaded:
            self._load_model()

    def _load_model(self):
        """Initializes RoMa model safely with cached/hub weights."""
        try:
            import romatch
            
            if self.model_type == "tiny":
                # Ensure xfeat is loaded cleanly with trust_repo=True
                xfeat = torch.hub.load(
                    "verlab/accelerated_features", "XFeat",
                    pretrained=True, top_k=4096, trust_repo=True
                ).net
                self.model = romatch.tiny_roma_v1_outdoor(device=self.device, xfeat=xfeat)
            else:
                self.model = romatch.roma_outdoor(device=self.device)
                
            self.model.eval()
            self._is_loaded = True
        except Exception as e:
            warnings.warn(f"Could not load RoMa model: {e}")
            self.model = None
            self._is_loaded = False

    def match(self, img_ref: np.ndarray, img_src: np.ndarray) -> MatcherResult:
        """
        Runs dense RoMa matching and samples high-certainty pixel correspondences.
        """
        self._ensure_loaded()
        if not self._is_loaded or self.model is None:
            return MatcherResult(matcher_name=self.matcher_name, metadata={"error": "RoMa model not loaded"})

        try:
            # Convert inputs to PIL Images
            if img_ref.ndim == 2:
                im_ref = PIL.Image.fromarray(img_ref).convert("RGB")
            else:
                im_ref = PIL.Image.fromarray(img_ref)

            if img_src.ndim == 2:
                im_src = PIL.Image.fromarray(img_src).convert("RGB")
            else:
                im_src = PIL.Image.fromarray(img_src)

            h_ref, w_ref = img_ref.shape[:2]
            h_src, w_src = img_src.shape[:2]

            with torch.no_grad():
                warp, certainty = self.model.match(im_ref, im_src)
                matches, certainty_pts = self.model.sample(
                    warp, certainty, num=self.sample_pts
                )
                kps_ref_t, kps_src_t = self.model.to_pixel_coordinates(
                    matches, h_ref, w_ref, h_src, w_src
                )

            kps_ref = kps_ref_t.detach().cpu().numpy().astype(np.float32)
            kps_src = kps_src_t.detach().cpu().numpy().astype(np.float32)
            confs = certainty_pts.detach().cpu().numpy().astype(np.float32)

            # Filter by certainty threshold
            valid_mask = (confs >= self.certainty_threshold) & \
                         (kps_ref[:, 0] >= 0) & (kps_ref[:, 0] < w_ref) & \
                         (kps_ref[:, 1] >= 0) & (kps_ref[:, 1] < h_ref) & \
                         (kps_src[:, 0] >= 0) & (kps_src[:, 0] < w_src) & \
                         (kps_src[:, 1] >= 0) & (kps_src[:, 1] < h_src)

            pts_ref = kps_ref[valid_mask]
            pts_src = kps_src[valid_mask]
            filtered_confs = confs[valid_mask]

            if len(pts_ref) == 0:
                return MatcherResult(
                    matcher_name=self.matcher_name,
                    metadata={"candidates": 0, "device": str(self.device)}
                )

            return MatcherResult(
                source_points=pts_src,
                reference_points=pts_ref,
                confidence=filtered_confs,
                matcher_name=self.matcher_name,
                metadata={
                    "candidates": len(pts_ref),
                    "sampled": self.sample_pts,
                    "mean_certainty": float(np.mean(filtered_confs)),
                    "device": str(self.device),
                    "model_type": self.model_type
                }
            )
        except Exception as e:
            warnings.warn(f"RoMa matching failed: {e}")
            return MatcherResult(matcher_name=self.matcher_name, metadata={"error": str(e)})

    def match_legacy(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray, np.ndarray]:
        """Legacy OpenCV tuple return for backward compatibility."""
        res = self.match(img_ref, img_src)
        return res.to_legacy_tuple()
