"""
Warping, correspondence visualization, and artifact generation for registered outputs.
"""
from typing import Tuple, List, Dict, Any, Optional
import os
import csv
import json
import numpy as np
import cv2


def warp_image_to_reference(
    image_src: np.ndarray,
    H_est: np.ndarray,
    ref_shape: Tuple[int, int],
    interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """
    Warps the source image into the reference frame using the estimated homography H_est (src -> ref).
    Defaults to high-fidelity Lanczos-4 reconstruction.
    """
    h_ref, w_ref = ref_shape[:2]
    warped = cv2.warpPerspective(
        image_src,
        H_est,
        (w_ref, h_ref),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped


def create_side_by_side(img1: np.ndarray, img2: np.ndarray, title1: str = "Reference", title2: str = "Source") -> np.ndarray:
    """Combines two images side-by-side with labels."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    
    # Ensure 3-channel BGR for drawing
    c1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()
    c2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2.copy()
    
    canvas = np.zeros((h + 40, w, 3), dtype=np.uint8)
    canvas[40:40+h1, 0:w1] = c1
    canvas[40:40+h2, w1:w1+w2] = c2
    
    # Labels
    cv2.putText(canvas, title1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, title2, (w1 + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return canvas


def create_match_visualization(
    img_ref: np.ndarray,
    img_src: np.ndarray,
    pts_ref: np.ndarray,
    pts_src: np.ndarray,
    inlier_mask: np.ndarray,
    max_draw: int = 150
) -> np.ndarray:
    """
    Draws candidate matches (inliers in green, outliers in red) across side-by-side images.
    """
    h_ref, w_ref = img_ref.shape[:2]
    h_src, w_src = img_src.shape[:2]
    h = max(h_ref, h_src)
    
    c_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if img_ref.ndim == 2 else img_ref.copy()
    c_src = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR) if img_src.ndim == 2 else img_src.copy()
    
    canvas = np.zeros((h + 50, w_ref + w_src, 3), dtype=np.uint8)
    canvas[50:50+h_ref, 0:w_ref] = c_ref
    canvas[50:50+h_src, w_ref:w_ref+w_src] = c_src
    
    num_pts = len(pts_ref)
    if num_pts == 0:
        return canvas
        
    indices = np.arange(num_pts)
    inlier_indices = indices[inlier_mask]
    outlier_indices = indices[~inlier_mask]
    
    if len(indices) > max_draw:
        np.random.seed(42)
        if len(inlier_indices) > max_draw // 2:
            sel_inliers = np.random.choice(inlier_indices, max_draw // 2, replace=False)
        else:
            sel_inliers = inlier_indices
        rem = max_draw - len(sel_inliers)
        sel_outliers = np.random.choice(outlier_indices, min(rem, len(outlier_indices)), replace=False) if len(outlier_indices) > 0 else np.array([])
        draw_indices = np.concatenate([sel_inliers, sel_outliers]).astype(int)
    else:
        draw_indices = indices

    # Draw matches
    for idx in draw_indices:
        is_inlier = bool(inlier_mask[idx])
        color = (0, 230, 0) if is_inlier else (0, 60, 240)  # Green for inliers, Red for outliers
        thickness = 2 if is_inlier else 1
        
        pt1 = (int(pts_ref[idx, 0]), int(pts_ref[idx, 1] + 50))
        pt2 = (int(pts_src[idx, 0] + w_ref), int(pts_src[idx, 1] + 50))
        
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.circle(canvas, pt2, 4, color, -1)

    # Top summary text
    inlier_count = int(np.sum(inlier_mask))
    cv2.putText(
        canvas,
        f"Matches: {inlier_count} Inliers (Green) / {num_pts - inlier_count} Outliers (Red) - Total: {num_pts}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return canvas


def create_overlay_blend(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Creates a false-color overlay (Ref in Green, Warped Source in Magenta)
    to visually inspect registration alignment quality.
    """
    c_ref = img_ref.astype(np.float32)
    c_warp = img_warped.astype(np.float32)
    
    # Create false-color RGB: R=warped, G=ref, B=warped (Ref green, Warped magenta)
    # Perfect alignment produces neutral monochrome where both overlap
    overlay = np.zeros((img_ref.shape[0], img_ref.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = np.clip(c_warp, 0, 255).astype(np.uint8)          # Blue = Warped
    overlay[:, :, 1] = np.clip(c_ref, 0, 255).astype(np.uint8)           # Green = Ref
    overlay[:, :, 2] = np.clip(c_warp, 0, 255).astype(np.uint8)          # Red = Warped
    
    return overlay


def save_registration_artifacts(
    output_dir: str,
    img_ref: np.ndarray,
    img_src: np.ndarray,
    img_warped: np.ndarray,
    pts_ref: np.ndarray,
    pts_src: np.ndarray,
    inlier_mask: np.ndarray,
    diagnostics: Dict[str, Any],
    H_est: np.ndarray,
    prefix: str = "mvp1",
    img_enhanced: Optional[np.ndarray] = None
) -> Dict[str, str]:
    """
    Saves raw and enhanced registered images, match visualization, overlay,
    checkerboard blend, correspondences CSV, and metadata JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    from zenith.enhance import (
        enhance_registered_image,
        create_false_color_overlay,
        create_blended_overlay,
        calculate_quality_statistics
    )

    # 1. Raw Registered warped image (AUTHORITATIVE)
    warped_path = os.path.join(output_dir, f"{prefix}_registered_warped.png")
    cv2.imwrite(warped_path, img_warped)
    saved_files["registered_image"] = warped_path

    # 2. Enhanced Registered warped image (OPTIONAL VISUALIZATION ONLY)
    if img_enhanced is None and img_warped is not None and img_warped.size > 0:
        img_enhanced = enhance_registered_image(img_warped, method="conservative")
    
    if img_enhanced is not None:
        enhanced_path = os.path.join(output_dir, f"{prefix}_enhanced_warped.png")
        cv2.imwrite(enhanced_path, img_enhanced)
        saved_files["enhanced_image"] = enhanced_path

    # 3. Input pair
    pair_vis = create_side_by_side(img_ref, img_src, "Reference (Fixed)", "Source (Transformed)")
    pair_path = os.path.join(output_dir, f"{prefix}_input_pair.png")
    cv2.imwrite(pair_path, pair_vis)
    saved_files["input_pair"] = pair_path

    # 4. Match visualization
    match_vis = create_match_visualization(img_ref, img_src, pts_ref, pts_src, inlier_mask)
    match_path = os.path.join(output_dir, f"{prefix}_matches.png")
    cv2.imwrite(match_path, match_vis)
    saved_files["matches_visualization"] = match_path

    # 5. Raw Overlay alignment
    overlay_vis = create_overlay_blend(img_ref, img_warped)
    overlay_path = os.path.join(output_dir, f"{prefix}_alignment_overlay.png")
    cv2.imwrite(overlay_path, overlay_vis)
    saved_files["alignment_overlay"] = overlay_path

    # 6. Enhanced False-Color Alignment Overlay
    enh_overlay_vis = create_false_color_overlay(img_ref, img_warped, enhance=True)
    enh_overlay_path = os.path.join(output_dir, f"{prefix}_enhanced_alignment_overlay.png")
    cv2.imwrite(enh_overlay_path, enh_overlay_vis)
    saved_files["enhanced_alignment_overlay"] = enh_overlay_path

    # 7. Checkerboard Blend Alignment Verification
    checker_vis = create_blended_overlay(img_ref, img_warped, blend_mode="checkerboard", checker_size=48)
    checker_path = os.path.join(output_dir, f"{prefix}_checkerboard_blend.png")
    cv2.imwrite(checker_path, checker_vis)
    saved_files["checkerboard_blend"] = checker_path

    # 8. Compute and attach non-registration Image Quality statistics
    q_stats = calculate_quality_statistics(img_warped, img_enhanced)
    diagnostics["image_quality_statistics"] = q_stats

    # 9. Correspondence CSV with confidence, residual, and coordinates
    csv_path = os.path.join(output_dir, f"{prefix}_correspondences.csv")
    residuals = diagnostics.get("residuals", np.zeros(len(pts_ref)))
    confidences = diagnostics.get("confidences", np.ones(len(pts_ref)))
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "ref_x", "ref_y", "src_x", "src_y", "is_inlier", "reproj_residual_px", "point_confidence"
        ])
        for idx in range(len(pts_ref)):
            writer.writerow([
                idx,
                round(float(pts_ref[idx, 0]), 3),
                round(float(pts_ref[idx, 1]), 3),
                round(float(pts_src[idx, 0]), 3),
                round(float(pts_src[idx, 1]), 3),
                int(inlier_mask[idx]) if idx < len(inlier_mask) else 0,
                round(float(residuals[idx]), 4) if idx < len(residuals) else 0.0,
                round(float(confidences[idx]), 4) if idx < len(confidences) else 0.0
            ])
    saved_files["correspondences_csv"] = csv_path

    # 10. Diagnostics metadata JSON
    meta_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    serializable_diag = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in diagnostics.items()
        if k != "residuals"
    }
    serializable_diag["H_est"] = H_est.tolist() if H_est is not None else None
    with open(meta_path, mode="w", encoding="utf-8") as f:
        json.dump(serializable_diag, f, indent=2)
    saved_files["metrics_json"] = meta_path

    return saved_files

