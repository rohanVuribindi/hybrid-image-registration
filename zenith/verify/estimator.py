"""
Geometric verification module using robust estimators (USAC_MAGSAC / MAGSAC++).
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class GeometricVerifier:
    """
    Robust geometric transformation estimator.
    Accurately identifies and names the estimator algorithm (e.g. USAC_MAGSAC / MAGSAC++).
    """
    def __init__(
        self,
        estimator_type: str = "USAC_MAGSAC",
        reproj_threshold: float = 3.0,
        max_iters: int = 5000,
        confidence: float = 0.999
    ):
        self.estimator_type = estimator_type
        self.reproj_threshold = reproj_threshold
        self.max_iters = max_iters
        self.confidence = confidence

    def verify(
        self,
        pts_src: np.ndarray,
        pts_ref: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
        """
        Estimates homography H mapping source points to reference coordinate frame: p_ref ~ H * p_src.
        
        Returns:
            H_est: (3, 3) estimated homography matrix, or None if estimation fails
            inlier_mask: boolean 1D array of shape (N,)
            diagnostics: dictionary with estimator name, inlier counts, residual stats
        """
        num_pts = len(pts_src)
        if num_pts < 4:
            return None, np.zeros(num_pts, dtype=bool), {
                "estimator_name": "None (Insufficient Points)",
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "status": "FAILED_INSUFFICIENT_POINTS"
            }

        estimator_flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") and self.estimator_type == "USAC_MAGSAC" else cv2.RANSAC
        estimator_name = "USAC_MAGSAC (MAGSAC++)" if estimator_flag == cv2.USAC_MAGSAC else "RANSAC"

        try:
            H_est, mask = cv2.findHomography(
                pts_src.reshape(-1, 1, 2),
                pts_ref.reshape(-1, 1, 2),
                method=estimator_flag,
                ransacReprojThreshold=self.reproj_threshold,
                maxIters=self.max_iters,
                confidence=self.confidence
            )
        except Exception as e:
            # Fallback to standard RANSAC if USAC_MAGSAC throws an unexpected exception
            H_est, mask = cv2.findHomography(
                pts_src.reshape(-1, 1, 2),
                pts_ref.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.reproj_threshold,
                maxIters=self.max_iters
            )
            estimator_name = f"RANSAC (Fallback from {self.estimator_type} due to {e})"

        if H_est is None or mask is None:
            return None, np.zeros(num_pts, dtype=bool), {
                "estimator_name": estimator_name,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "status": "FAILED_ESTIMATION"
            }

        inlier_mask = mask.ravel().astype(bool)
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count / num_pts) if num_pts > 0 else 0.0

        # Compute point-wise reprojection residuals under H_est: || p_ref - H_est(p_src) ||
        pts_src_homo = np.column_stack([pts_src, np.ones(num_pts)])
        projected = (H_est @ pts_src_homo.T).T
        projected_pts = projected[:, :2] / (projected[:, 2:3] + 1e-12)
        residuals = np.linalg.norm(pts_ref - projected_pts, axis=1)

        diagnostics = {
            "estimator_name": estimator_name,
            "total_candidates": num_pts,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "reproj_threshold": self.reproj_threshold,
            "mean_inlier_residual_px": float(np.mean(residuals[inlier_mask])) if inlier_count > 0 else 0.0,
            "max_inlier_residual_px": float(np.max(residuals[inlier_mask])) if inlier_count > 0 else 0.0,
            "residuals": residuals,
            "status": "VERIFIED" if inlier_count >= 4 else "FAILED_LOW_INLIERS"
        }

        return H_est, inlier_mask, diagnostics


def verify_correspondences(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    estimator_type: str = "USAC_MAGSAC",
    reproj_threshold: float = 3.0
) -> Tuple[Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    verifier = GeometricVerifier(estimator_type=estimator_type, reproj_threshold=reproj_threshold)
    return verifier.verify(pts_src, pts_ref)
