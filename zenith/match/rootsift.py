"""
RootSIFT feature extractor and matcher.
Implements the Arandjelović & Zisserman (2012) Hellinger kernel / L1-sqrt transformation on SIFT descriptors.
"""
from typing import Tuple, List, Optional
import numpy as np
import cv2


def extract_rootsift_features(
    image: np.ndarray,
    nfeatures: int = 4000,
    contrast_threshold: float = 0.03,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
    eps: float = 1e-7
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Extracts SIFT keypoints and transforms descriptors into RootSIFT (L1-norm + square root + L2-norm).
    """
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )
    
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None or len(descriptors) == 0:
        return keypoints, None
    
    # RootSIFT formulation:
    # 1. L1 normalization across feature dimensions
    l1_norm = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True)
    descriptors_l1 = descriptors / (l1_norm + eps)
    
    # 2. Element-wise square root (Hellinger kernel embedding)
    descriptors_rootsift = np.sqrt(descriptors_l1)
    
    # 3. L2 normalization of the square-rooted vectors
    l2_norm = np.linalg.norm(descriptors_rootsift, ord=2, axis=1, keepdims=True)
    descriptors_rootsift = descriptors_rootsift / (l2_norm + eps)
    
    return keypoints, descriptors_rootsift.astype(np.float32)


class RootSIFTMatcher:
    """
    Candidate correspondence matcher using RootSIFT and Lowe's ratio test.
    """
    def __init__(
        self,
        ratio_threshold: float = 0.75,
        cross_check: bool = False,
        nfeatures: int = 4000
    ):
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
        self.nfeatures = nfeatures
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
    def match(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray, np.ndarray]:
        """
        Detects RootSIFT features on both reference and source images, then performs k-NN matching
        with Lowe's second-nearest neighbor distance ratio test.
        
        Returns:
            kp_ref: Keypoints in reference image
            kp_src: Keypoints in source image
            good_matches: List of cv2.DMatch passing ratio test
            pts_ref: (N, 2) array of reference coordinates [x, y]
            pts_src: (N, 2) array of source coordinates [x, y]
        """
        kp_ref, desc_ref = extract_rootsift_features(img_ref, nfeatures=self.nfeatures)
        kp_src, desc_src = extract_rootsift_features(img_src, nfeatures=self.nfeatures)
        
        if desc_ref is None or desc_src is None or len(desc_ref) < 4 or len(desc_src) < 4:
            return kp_ref, kp_src, [], np.empty((0, 2)), np.empty((0, 2))
        
        # 2-NN matching for Lowe's ratio test
        raw_matches = self.matcher.knnMatch(desc_ref, desc_src, k=2)
        
        good_matches = []
        for m_tuple in raw_matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(m_tuple) == 1:
                good_matches.append(m_tuple[0])
                
        if len(good_matches) == 0:
            return kp_ref, kp_src, [], np.empty((0, 2)), np.empty((0, 2))
            
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches])
        pts_src = np.float32([kp_src[m.trainIdx].pt for m in good_matches])
        
        return kp_ref, kp_src, good_matches, pts_ref, pts_src


class RootSIFTAdapter:
    """
    Adapter wrapping RootSIFTMatcher to conform to the universal BaseMatcher & MatcherResult interface.
    """
    def __init__(
        self,
        ratio_threshold: float = 0.75,
        nfeatures: int = 4000
    ):
        self.matcher_name = "RootSIFT"
        self.inner = RootSIFTMatcher(ratio_threshold=ratio_threshold, nfeatures=nfeatures)

    def match(self, img_ref: np.ndarray, img_src: np.ndarray):
        from .common import MatcherResult
        kp_ref, kp_src, good_matches, pts_ref, pts_src = self.inner.match(img_ref, img_src)
        
        if len(pts_ref) == 0:
            return MatcherResult(
                source_points=np.empty((0, 2), dtype=np.float32),
                reference_points=np.empty((0, 2), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                matcher_name=self.matcher_name,
                metadata={"candidates": 0, "kp_ref_count": len(kp_ref), "kp_src_count": len(kp_src)}
            )

        # Confidence from Lowe's ratio distance
        distances = np.array([m.distance for m in good_matches], dtype=np.float32)
        max_d = max(float(np.max(distances)), 1.0)
        confidences = np.clip(1.0 - (distances / max_d), 0.1, 1.0)

        return MatcherResult(
            source_points=pts_src,
            reference_points=pts_ref,
            confidence=confidences,
            matcher_name=self.matcher_name,
            metadata={
                "candidates": len(pts_ref),
                "kp_ref_count": len(kp_ref),
                "kp_src_count": len(kp_src)
            }
        )

