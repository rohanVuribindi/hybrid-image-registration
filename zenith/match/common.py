"""
Common correspondence data structures and abstract base matcher interface for ZENITH.
Provides the universal MatcherResult contract across all classical, learned, and dense matchers.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class MatcherResult:
    """
    Standardized result contract returned by every ZENITH matcher.
    
    Attributes:
        source_points: (N, 2) array of coordinates [x, y] in source image space.
        reference_points: (N, 2) array of coordinates [x, y] in reference image space.
        confidence: (N,) array of correspondence confidence values in [0.0, 1.0].
        matcher_name: Identifier of the generating matcher (e.g. 'RootSIFT', 'LightGlue', 'RoMa').
        metadata: Detailed diagnostics, timing, keypoint counts, or model-specific outputs.
    """
    source_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    reference_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    confidence: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    matcher_name: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.source_points) > 0:
            self.source_points = np.asarray(self.source_points, dtype=np.float32)
            self.reference_points = np.asarray(self.reference_points, dtype=np.float32)
            if len(self.confidence) == 0:
                self.confidence = np.ones(len(self.source_points), dtype=np.float32)
            else:
                self.confidence = np.asarray(self.confidence, dtype=np.float32)
        else:
            self.source_points = np.empty((0, 2), dtype=np.float32)
            self.reference_points = np.empty((0, 2), dtype=np.float32)
            self.confidence = np.empty((0,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.source_points)

    @property
    def is_empty(self) -> bool:
        return len(self.source_points) == 0

    def to_legacy_tuple(self) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray, np.ndarray]:
        """
        Converts to legacy OpenCV (kp_ref, kp_src, matches, pts_ref, pts_src) format for backward compatibility.
        """
        if self.is_empty:
            return [], [], [], np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        kp_ref = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0) for pt in self.reference_points]
        kp_src = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0) for pt in self.source_points]
        matches = [
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=float(1.0 - self.confidence[i]))
            for i in range(len(self.source_points))
        ]
        return kp_ref, kp_src, matches, self.reference_points, self.source_points


class BaseMatcher:
    """
    Abstract base class that all ZENITH matcher adapters must implement.
    """
    def __init__(self, matcher_name: str = "BaseMatcher"):
        self.matcher_name = matcher_name

    def match(self, img_ref: np.ndarray, img_src: np.ndarray) -> MatcherResult:
        """
        Extracts correspondences between reference and source images.
        
        Args:
            img_ref: Reference image (grayscale or RGB, uint8 or float32)
            img_src: Source image (grayscale or RGB, uint8 or float32)
            
        Returns:
            MatcherResult containing source/reference points, confidence, and metadata.
        """
        raise NotImplementedError("Subclasses must implement match()")
