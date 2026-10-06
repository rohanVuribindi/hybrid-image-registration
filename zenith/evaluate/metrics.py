"""
Evaluation metrics computation: RMSE, Inlier Ratio, Spatial Coverage, and Error Statistics.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial import ConvexHull


def compute_spatial_coverage(
    points: np.ndarray,
    image_shape: Tuple[int, int],
    grid_bins: Tuple[int, int] = (4, 4)
) -> Dict[str, float]:
    """
    Measures how evenly distributed correspondences are across the image area,
    preventing false confidence from clusters concentrated in a single crater.
    
    Returns:
        grid_coverage_ratio: Fraction of grid cells containing >= 1 point (0.0 to 1.0).
        convex_hull_ratio: Ratio of convex hull area of inliers to total image area (0.0 to 1.0).
        coverage_score: Weighted composite coverage score.
    """
    if len(points) == 0:
        return {
            "grid_coverage_ratio": 0.0,
            "convex_hull_ratio": 0.0,
            "coverage_score": 0.0,
            "active_bins": 0,
            "total_bins": grid_bins[0] * grid_bins[1]
        }

    h, w = image_shape[:2]
    total_bins = grid_bins[0] * grid_bins[1]
    
    # 1. Grid occupancy
    x_bins = np.clip((points[:, 0] / w * grid_bins[1]).astype(int), 0, grid_bins[1] - 1)
    y_bins = np.clip((points[:, 1] / h * grid_bins[0]).astype(int), 0, grid_bins[0] - 1)
    unique_cells = set(zip(y_bins, x_bins))
    active_bins = len(unique_cells)
    grid_coverage_ratio = float(active_bins / total_bins)

    # 2. Convex hull area ratio
    convex_hull_ratio = 0.0
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            convex_hull_ratio = float(min(1.0, hull.volume / (w * h)))  # In 2D, hull.volume is the 2D area
        except Exception:
            convex_hull_ratio = 0.0

    # Composite coverage score
    coverage_score = float(0.6 * grid_coverage_ratio + 0.4 * convex_hull_ratio)

    return {
        "grid_coverage_ratio": grid_coverage_ratio,
        "convex_hull_ratio": convex_hull_ratio,
        "coverage_score": coverage_score,
        "active_bins": active_bins,
        "total_bins": total_bins
    }


def compute_ground_truth_rmse(
    H_est: np.ndarray,
    H_gt: np.ndarray,
    image_shape: Tuple[int, int],
    num_eval_points: int = 400
) -> Dict[str, float]:
    """
    Calculates Point Transfer RMSE across a dense evaluation grid in the reference image
    against known ground-truth homography H_gt (ref -> src) and H_est (src -> ref).
    """
    if H_est is None or H_gt is None:
        return {
            "ground_truth_available": False,
            "rmse_px": -1.0,
            "mean_error_px": -1.0,
            "max_error_px": -1.0
        }

    h, w = image_shape[:2]
    # Generate uniform grid of test points across reference image
    grid_y, grid_x = np.meshgrid(
        np.linspace(10, h - 10, int(np.sqrt(num_eval_points))),
        np.linspace(10, w - 10, int(np.sqrt(num_eval_points)))
    )
    pts_ref = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    num_pts = len(pts_ref)

    # Transfer pts_ref -> pts_src via H_gt
    ref_homo = np.column_stack([pts_ref, np.ones(num_pts)])
    src_gt_homo = (H_gt @ ref_homo.T).T
    pts_src_gt = src_gt_homo[:, :2] / (src_gt_homo[:, 2:3] + 1e-12)

    # Reproject pts_src_gt back to reference frame via estimated H_est
    src_eval_homo = np.column_stack([pts_src_gt, np.ones(num_pts)])
    ref_est_homo = (H_est @ src_eval_homo.T).T
    pts_ref_est = ref_est_homo[:, :2] / (ref_est_homo[:, 2:3] + 1e-12)

    # Calculate Euclidean errors across dense evaluation grid
    errors = np.linalg.norm(pts_ref - pts_ref_est, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    # Genuinely compute 4-corner transfer RMSE
    corners_ref = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    c_homo = np.column_stack([corners_ref, np.ones(4)])
    corners_src_gt = (H_gt @ c_homo.T).T
    corners_src_gt = corners_src_gt[:, :2] / (corners_src_gt[:, 2:3] + 1e-12)
    s_homo = np.column_stack([corners_src_gt, np.ones(4)])
    corners_ref_est = (H_est @ s_homo.T).T
    corners_ref_est = corners_ref_est[:, :2] / (corners_ref_est[:, 2:3] + 1e-12)
    corner_errors = np.linalg.norm(corners_ref - corners_ref_est, axis=1)
    corner_rmse = float(np.sqrt(np.mean(corner_errors ** 2)))

    return {
        "ground_truth_available": True,
        "rmse_px": rmse,
        "corner_rmse_px": corner_rmse,
        "corner_rmse": corner_rmse,
        "mean_error_px": mean_err,
        "max_error_px": max_err
    }
