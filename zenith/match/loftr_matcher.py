"""
LoFTR (Detector-Free Local Feature Matching with Transformers) Matcher.
Provides deep transformer-based feature correspondences using Kornia.
"""
from typing import Tuple, List, Dict, Any, Optional
import os
import numpy as np
import cv2
import torch
import warnings


class LoFTRMatcher:
    """
    LoFTR transformer-based dense correspondence matcher.
    """
    def __init__(
        self,
        pretrained: str = "outdoor",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None
    ):
        self.pretrained = pretrained
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = None
        # Lazy loading: do not load/download weights during __init__

    def _ensure_loaded(self):
        """Loads LoFTR model weights on demand."""
        if self.model is None:
            self._load_model()

    def _load_model(self):
        """Initializes LoFTR model and loads weights safely via HuggingFace hub if needed."""
        import kornia.feature as kf
        
        try:
            # First try default kornia loading
            self.model = kf.LoFTR(pretrained=self.pretrained).to(self.device).eval()
        except Exception:
            # Fallback to HuggingFace hub download
            try:
                from huggingface_hub import hf_hub_download
                ckpt_filename = f"loftr_{self.pretrained}.ckpt"
                ckpt_path = hf_hub_download(repo_id="kornia/loftr", filename=ckpt_filename)
                
                # Instantiate uninitialized LoFTR model and load state_dict
                self.model = kf.LoFTR(pretrained=None).to(self.device)
                state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
            except Exception as e:
                warnings.warn(f"Could not load LoFTR weights: {e}")
                self.model = None

    def match(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray, np.ndarray]:
        """
        Runs LoFTR transformer matching across reference and source images.
        """
        self._ensure_loaded()
        if self.model is None:
            return [], [], [], np.empty((0, 2)), np.empty((0, 2))

        # Convert images to float32 grayscale tensors normalized to [0, 1]
        def to_tensor(img):
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            # LoFTR expects dimensions divisible by 8
            h, w = gray.shape[:2]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                gray = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
            return t.to(self.device)

        t_ref = to_tensor(img_ref)
        t_src = to_tensor(img_src)

        input_dict = {"image0": t_ref, "image1": t_src}
        
        with torch.no_grad():
            corr = self.model(input_dict)

        pts0 = corr["keypoints0"].cpu().numpy()  # Reference points
        pts1 = corr["keypoints1"].cpu().numpy()  # Source points
        confs = corr["confidence"].cpu().numpy()

        # Filter by confidence threshold
        mask = confs >= self.confidence_threshold
        pts_ref = pts0[mask]
        pts_src = pts1[mask]
        filtered_confs = confs[mask]

        if len(pts_ref) == 0:
            return [], [], [], np.empty((0, 2)), np.empty((0, 2))

        # Build OpenCV KeyPoints and DMatch objects for API compatibility
        kp_ref = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0) for pt in pts_ref]
        kp_src = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0) for pt in pts_src]
        matches = [
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=float(1.0 - filtered_confs[i]))
            for i in range(len(pts_ref))
        ]

        return kp_ref, kp_src, matches, pts_ref.astype(np.float32), pts_src.astype(np.float32)


class LoFTRAdapter:
    """
    Adapter wrapping LoFTRMatcher to conform to the universal BaseMatcher & MatcherResult interface.
    """
    def __init__(
        self,
        pretrained: str = "outdoor",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None
    ):
        self.matcher_name = "LoFTR"
        self.inner = LoFTRMatcher(
            pretrained=pretrained,
            confidence_threshold=confidence_threshold,
            device=device
        )

    def match(self, img_ref: np.ndarray, img_src: np.ndarray):
        from .common import MatcherResult
        kp_ref, kp_src, matches, pts_ref, pts_src = self.inner.match(img_ref, img_src)
        
        if len(pts_ref) == 0:
            return MatcherResult(
                source_points=np.empty((0, 2), dtype=np.float32),
                reference_points=np.empty((0, 2), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                matcher_name=self.matcher_name,
                metadata={"candidates": 0, "device": str(self.inner.device)}
            )

        confidences = np.array([1.0 - m.distance for m in matches], dtype=np.float32)
        return MatcherResult(
            source_points=pts_src,
            reference_points=pts_ref,
            confidence=confidences,
            matcher_name=self.matcher_name,
            metadata={"candidates": len(pts_ref), "device": str(self.inner.device)}
        )

