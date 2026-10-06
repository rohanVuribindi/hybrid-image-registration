"""
REFINE module for Zenith pipeline.
Provides spatial balancing, sub-pixel feature refinement, and per-point residual/confidence attachment.
"""
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2

from .spatial_balancer import balance_correspondences_spatially
from .subpixel import refine_subpixel_coordinates, compute_point_confidences


class PointRefiner:
    """
    Executes post-verification refinement:
    Surviving Inliers -> Spatial Balancing -> Sub-pixel Refinement -> Point-level Residual & Confidence.
    """
    def __init__(
        self,
        grid_size: Tuple[int, int] = (8, 8),
        max_pts_per_bin: int = 5,
        subpix_win_size: Tuple[int, int] = (5, 5),
        subpix_max_iters: int = 30
    ):
        self.grid_size = grid_size
        self.max_pts_per_bin = max_pts_per_bin
        self.subpix_win_size = subpix_win_size
        self.subpix_max_iters = subpix_max_iters

    def refine(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray,
        pts_ref: np.ndarray,
        pts_src: np.ndarray,
        inlier_mask: np.ndarray,
        H_est: Optional[np.ndarray],
        raw_residuals: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes refinement strictly on inlier correspondences.
        """
        if H_est is None or np.sum(inlier_mask) < 4:
            return {
                "pts_ref_refined": np.empty((0, 2)),
                "pts_src_refined": np.empty((0, 2)),
                "residuals": np.empty(0),
                "confidences": np.empty(0),
                "H_refined": H_est,
                "balanced_count": 0,
                "mean_refined_residual_px": 0.0
            }

        # 1. Select strictly surviving inliers
        inlier_pts_ref = pts_ref[inlier_mask]
        inlier_pts_src = pts_src[inlier_mask]
        inlier_residuals = raw_residuals[inlier_mask] if len(raw_residuals) == len(pts_ref) else np.zeros(len(inlier_pts_ref))

        # 2. Spatial Balancing (prevent crater clustering)
        pts_ref_bal, pts_src_bal, res_bal, bal_indices = balance_correspondences_spatially(
            pts_ref=inlier_pts_ref,
            pts_src=inlier_pts_src,
            residuals=inlier_residuals,
            image_shape=img_ref.shape,
            grid_size=self.grid_size,
            max_pts_per_bin=self.max_pts_per_bin
        )

        # 3. Sub-pixel Refinement on reference and source patches
        pts_ref_subpix = refine_subpixel_coordinates(
            image=img_ref,
            points=pts_ref_bal,
            win_size=self.subpix_win_size,
            max_iters=self.subpix_max_iters
        )
        pts_src_subpix = refine_subpixel_coordinates(
            image=img_src,
            points=pts_src_bal,
            win_size=self.subpix_win_size,
            max_iters=self.subpix_max_iters
        )

        # 4. Re-estimate transformation on refined coordinates via Least Squares (Levenberg-Marquardt)
        H_refined, _ = cv2.findHomography(
            pts_src_subpix.reshape(-1, 1, 2),
            pts_ref_subpix.reshape(-1, 1, 2),
            method=cv2.LMEDS
        )
        if H_refined is None:
            H_refined = H_est

        # 5. Compute point-level residuals and confidence
        residuals, confidences = compute_point_confidences(
            pts_ref=pts_ref_subpix,
            pts_src=pts_src_subpix,
            H_est=H_refined
        )

        mean_res = float(np.mean(residuals)) if len(residuals) > 0 else 0.0

        return {
            "pts_ref_refined": pts_ref_subpix,
            "pts_src_refined": pts_src_subpix,
            "residuals": residuals,
            "confidences": confidences,
            "H_refined": H_refined,
            "balanced_count": len(pts_ref_subpix),
            "mean_refined_residual_px": mean_res
        }


__all__ = [
    "balance_correspondences_spatially",
    "refine_subpixel_coordinates",
    "compute_point_confidences",
    "PointRefiner"
]
