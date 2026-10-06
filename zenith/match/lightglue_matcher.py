"""
LightGlue (Deep Feature Matcher with Adaptive Pruning) Matcher & Adapter for ZENITH.
Integrates SOTA sparse matching via ALIKED / SuperPoint + LightGlue.
Provides lazy-loading, CPU/CUDA auto-detection, and universal MatcherResult contract.
"""
from typing import Tuple, List, Dict, Any, Optional
import os
import warnings
import numpy as np
import cv2
import torch

from .common import BaseMatcher, MatcherResult


class LightGlueMatcher(BaseMatcher):
    """
    LightGlue neural feature matcher supporting ALIKED and SuperPoint backbones.
    """
    def __init__(
        self,
        features: str = "aliked",  # "aliked", "superpoint", "disk"
        max_num_keypoints: int = 1536,
        filter_threshold: float = 0.10,
        device: Optional[str] = None
    ):
        super().__init__(matcher_name=f"LightGlue_{features.upper()}")
        self.features = features.lower()
        self.max_num_keypoints = max_num_keypoints
        self.filter_threshold = filter_threshold
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.extractor = None
        self.matcher = None
        self._is_loaded = False

    def _ensure_loaded(self):
        """Loads feature extractor and LightGlue weights on demand."""
        if not self._is_loaded:
            self._load_models()

    def _load_models(self):
        """Initializes extractor and LightGlue from official package or kornia."""
        try:
            from lightglue import LightGlue, ALIKED, SuperPoint, DISK, SIFT
            
            if self.features == "aliked":
                self.extractor = ALIKED(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features="aliked", filter_threshold=self.filter_threshold).eval().to(self.device)
            elif self.features == "superpoint":
                self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features="superpoint", filter_threshold=self.filter_threshold).eval().to(self.device)
            elif self.features == "disk":
                self.extractor = DISK(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features="disk", filter_threshold=self.filter_threshold).eval().to(self.device)
            else:
                self.extractor = ALIKED(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features="aliked", filter_threshold=self.filter_threshold).eval().to(self.device)
                
            self._is_loaded = True
        except Exception as e_lg:
            # Fallback to kornia LightGlueMatcher if available
            try:
                import kornia.feature as kf
                self.matcher = kf.LightGlueMatcher(self.features).eval().to(self.device)
                self._is_loaded = True
            except Exception as e_kf:
                warnings.warn(f"Could not load LightGlue weights: {e_lg} / {e_kf}")
                self.extractor = None
                self.matcher = None
                self._is_loaded = False

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Converts image to normalized float32 tensor (1, C, H, W) or (1, 1, H, W)."""
        if img.ndim == 2:
            tensor = torch.from_numpy(img).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif img.ndim == 3:
            # RGB / BGR
            if img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                tensor = torch.from_numpy(gray).float() / 255.0
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            else:
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0)
        else:
            tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def match(self, img_ref: np.ndarray, img_src: np.ndarray) -> MatcherResult:
        """
        Runs LightGlue matching across reference and source images.
        """
        self._ensure_loaded()
        if not self._is_loaded or self.matcher is None:
            return MatcherResult(matcher_name=self.matcher_name, metadata={"error": "Model not loaded"})

        try:
            t0 = self._to_tensor(img_ref)
            t1 = self._to_tensor(img_src)
            
            with torch.no_grad():
                feats0 = self.extractor.extract(t0)
                feats1 = self.extractor.extract(t1)
                
                # Match features
                match_res = self.matcher({"image0": feats0, "image1": feats1})
                
                # Extract indices and scores
                matches_idx = match_res["matches"][0].detach().cpu().numpy()  # (M, 2) [idx0, idx1]
                scores = match_res["scores"][0].detach().cpu().numpy() if "scores" in match_res else np.ones(len(matches_idx), dtype=np.float32)
                
                kpts0 = feats0["keypoints"][0].detach().cpu().numpy()  # (N0, 2) [x, y] in ref
                kpts1 = feats1["keypoints"][0].detach().cpu().numpy()  # (N1, 2) [x, y] in src

            if len(matches_idx) == 0:
                return MatcherResult(
                    matcher_name=self.matcher_name,
                    metadata={"candidates": 0, "kpts_ref": len(kpts0), "kpts_src": len(kpts1), "device": str(self.device)}
                )

            pts_ref = kpts0[matches_idx[:, 0]].astype(np.float32)
            pts_src = kpts1[matches_idx[:, 1]].astype(np.float32)
            confs = scores.astype(np.float32)

            return MatcherResult(
                source_points=pts_src,
                reference_points=pts_ref,
                confidence=confs,
                matcher_name=self.matcher_name,
                metadata={
                    "candidates": len(pts_ref),
                    "kpts_ref": len(kpts0),
                    "kpts_src": len(kpts1),
                    "device": str(self.device),
                    "backbone": self.features
                }
            )
        except Exception as e:
            warnings.warn(f"LightGlue matching failed: {e}")
            return MatcherResult(matcher_name=self.matcher_name, metadata={"error": str(e)})

    def match_legacy(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray, np.ndarray]:
        """Legacy OpenCV tuple return for backward compatibility."""
        res = self.match(img_ref, img_src)
        return res.to_legacy_tuple()
