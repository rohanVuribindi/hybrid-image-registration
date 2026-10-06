"""
Spatial balancing module.
Prevents correspondence clustering in single high-contrast craters by enforcing
uniform spatial distribution across spatial grid buckets.
"""
from typing import Tuple, List, Dict, Any
import numpy as np


def balance_correspondences_spatially(
    pts_ref: np.ndarray,
    pts_src: np.ndarray,
    residuals: np.ndarray,
    image_shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (8, 8),
    max_pts_per_bin: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatially balances verified inlier correspondences.
    
    Args:
        pts_ref: (N, 2) Reference image coordinates of verified inliers.
        pts_src: (N, 2) Source image coordinates of verified inliers.
        residuals: (N,) Inlier reprojection residuals (lower is better).
        image_shape: (H, W) Reference image dimensions.
        grid_size: (rows, cols) Spatial binning grid.
        max_pts_per_bin: Maximum points to retain per spatial cell.
        
    Returns:
        pts_ref_balanced, pts_src_balanced, residuals_balanced, selected_indices
    """
    n_pts = len(pts_ref)
    if n_pts <= max_pts_per_bin * 2:
        return pts_ref, pts_src, residuals, np.arange(n_pts)

    h, w = image_shape[:2]
    n_rows, n_cols = grid_size
    
    bin_x = np.clip((pts_ref[:, 0] / w * n_cols).astype(int), 0, n_cols - 1)
    bin_y = np.clip((pts_ref[:, 1] / h * n_rows).astype(int), 0, n_rows - 1)
    
    bins: Dict[Tuple[int, int], List[int]] = {}
    for idx in range(n_pts):
        key = (bin_y[idx], bin_x[idx])
        if key not in bins:
            bins[key] = []
        bins[key].append(idx)
        
    selected_indices = []
    for key, indices in bins.items():
        # Sort by residual (smallest residual = highest geometric accuracy)
        indices_sorted = sorted(indices, key=lambda i: residuals[i])
        selected_indices.extend(indices_sorted[:max_pts_per_bin])
        
    selected_indices = np.array(sorted(selected_indices), dtype=int)
    
    return (
        pts_ref[selected_indices],
        pts_src[selected_indices],
        residuals[selected_indices],
        selected_indices
    )
