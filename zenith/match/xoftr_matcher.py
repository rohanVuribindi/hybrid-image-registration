"""
XoFTR (Cross-Modal Feature Matching Transformer) Matcher & Adapter for ZENITH.
Optional specialist adapter for thermal/infrared vs visible cross-modal pairs.
Includes graceful dependency/weight validation and universal MatcherResult contract.
"""
from typing import Tuple, List, Dict, Any, Optional
import os
import warnings
import numpy as np
import cv2
import torch

from .common import BaseMatcher, MatcherResult


class XoFTRMatcher(BaseMatcher):
    """
    XoFTR specialist matcher for cross-modal imagery.
    Loads conditionally if optional dependencies are present, otherwise fails gracefully.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.20,
        device: Optional[str] = None
    ):
        super().__init__(matcher_name="XoFTR")
        self.confidence_threshold = confidence_threshold
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None
        self._is_loaded = False
        self._load_attempted = False

    def _ensure_loaded(self):
        if not self._load_attempted:
            self._load_attempted = True
            self._load_model()

    def _load_model(self):
        try:
            import vismatch
            self.model = vismatch.get_matcher("xoftr", device=str(self.device))
            self._is_loaded = True
        except Exception as e:
            # Graceful warning: XoFTR is an optional specialist
            warnings.warn(f"XoFTR model not available: {e}. RoMa / LoFTR will be utilized instead.")
            self.model = None
            self._is_loaded = False

    def match(self, img_ref: np.ndarray, img_src: np.ndarray) -> MatcherResult:
        self._ensure_loaded()
        if not self._is_loaded or self.model is None:
            return MatcherResult(
                matcher_name=self.matcher_name,
                metadata={"available": False, "reason": "XoFTR optional dependencies / weights not present"}
            )

        try:
            # Run vismatch XoFTR pipeline if available
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f0, \
                 tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
                cv2.imwrite(f0.name, img_ref)
                cv2.imwrite(f1.name, img_src)
                p0, p1 = f0.name, f1.name

            try:
                preds = self.model(p0, p1)
                kps0 = preds.get("keypoints0", np.empty((0, 2)))
                kps1 = preds.get("keypoints1", np.empty((0, 2)))
                confs = preds.get("confidence", np.ones(len(kps0), dtype=np.float32))
            finally:
                if os.path.exists(p0):
                    os.remove(p0)
                if os.path.exists(p1):
                    os.remove(p1)

            if len(kps0) == 0:
                return MatcherResult(matcher_name=self.matcher_name, metadata={"candidates": 0})

            return MatcherResult(
                source_points=np.asarray(kps1, dtype=np.float32),
                reference_points=np.asarray(kps0, dtype=np.float32),
                confidence=np.asarray(confs, dtype=np.float32),
                matcher_name=self.matcher_name,
                metadata={"candidates": len(kps0), "device": str(self.device)}
            )
        except Exception as e:
            warnings.warn(f"XoFTR execution failed: {e}")
            return MatcherResult(matcher_name=self.matcher_name, metadata={"error": str(e)})

    def match_legacy(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray, np.ndarray]:
        res = self.match(img_ref, img_src)
        return res.to_legacy_tuple()
