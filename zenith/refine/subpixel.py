"""
Sub-pixel refinement and point-level confidence/residual estimation.
Strictly executed only on correspondences that have passed VERIFY and spatial balancing.
"""
from typing import Tuple, Dict, Any, List
import numpy as np
import cv2


def refine_subpixel_coordinates(
    image: np.ndarray,
    points: np.ndarray,
    win_size: Tuple[int, int] = (5, 5),
    zero_zone: Tuple[int, int] = (-1, -1),
    max_iters: int = 40,
    epsilon: float = 0.001
) -> np.ndarray:
    """
    Refines feature point coordinates to sub-pixel accuracy using gradient structure optimization.
    """
    if len(points) == 0:
        return points.copy()

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure float32 (N, 1, 2) format for OpenCV subpixel refinement
    pts_f32 = np.ascontiguousarray(points.reshape(-1, 1, 2), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, epsilon)

    try:
        refined = cv2.cornerSubPix(
            gray,
            pts_f32,
            winSize=win_size,
            zeroZone=zero_zone,
            criteria=criteria
        )
        return refined.reshape(-1, 2)
    except Exception:
        # Fallback to original positions if cornerSubPix fails on non-standard local gradient
        return points.copy()


def compute_point_confidences(
    pts_ref: np.ndarray,
    pts_src: np.ndarray,
    H_est: np.ndarray,
    residual_scale_sigma: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes exact geometric residual (px) and normalized confidence [0.0, 1.0] for every surviving point.
    
    Residual: r_i = || p_ref,i - H_est(p_src,i) ||_2
    Confidence: c_i = exp(- r_i^2 / (2 * sigma^2))
    """
    n_pts = len(pts_ref)
    if n_pts == 0 or H_est is None:
        return np.empty(0), np.empty(0)

    # Project source points to reference frame under H_est
    src_homo = np.column_stack([pts_src, np.ones(n_pts)])
    projected = (H_est @ src_homo.T).T
    projected_pts = projected[:, :2] / (projected[:, 2:3] + 1e-12)

    residuals = np.linalg.norm(pts_ref - projected_pts, axis=1)
    
    # Gaussian confidence model centered at 0 residual error
    confidences = np.exp(-(residuals ** 2) / (2.0 * (residual_scale_sigma ** 2)))
    confidences = np.clip(confidences, 0.0, 1.0)

    return residuals, confidences
