"""
Runner script for ZENITH MVP2: REPRESENT Module (Phase Congruency + Gradient + Pyramid) verification.
"""
import os
import sys
import json
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenith.adapt import generate_synthetic_lunar_pair
from zenith.represent import (
    compute_phase_congruency,
    compute_gradient_representation,
    ImagePyramid,
    StructuralRepresenter
)
from zenith.pipeline import ZenithPipeline


def apply_illumination_shift(image: np.ndarray, slope: float = 0.7, non_linear: bool = True) -> np.ndarray:
    """
    Simulates cross-sensor lunar illumination gap (severe shadow terminator gradient and gamma distortion).
    """
    h, w = image.shape[:2]
    # Linear ramp across diagonal (sun angle variation)
    y_grad, x_grad = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
    gradient = (x_grad * 0.8 + y_grad * 0.2)
    
    img_f = image.astype(np.float32) / 255.0
    # Apply illumination field and non-linear contrast distortion
    distorted = img_f * (0.35 + 0.65 * gradient)
    if non_linear:
        distorted = np.power(distorted, 1.45) # Gamma change
    
    return np.clip(distorted * 255.0, 0, 255).astype(np.uint8)


def main():
    print("=" * 75)
    print("  ZENITH CROSS-MODAL REGISTRATION - MVP2 (REPRESENT VERIFICATION)")
    print("=" * 75)

    # 1. ADAPT: Generate synthetic pair
    print("\n[Step 1: ADAPT] Generating base synthetic lunar surface with ground truth...")
    img_ref, img_src_clean, H_gt, meta = generate_synthetic_lunar_pair(
        rotation_deg=22.0,
        scale=1.18,
        translation=(32.0, -20.0),
        perspective_shear=(0.00010, -0.00010),
        output_size=(800, 800),
        seed=101
    )
    
    # Introduce severe illumination gradient and non-linear sensor response to source image
    img_src = apply_illumination_shift(img_src_clean)
    print(f"  Applied simulated cross-sensor illumination gradient & gamma shift to source image.")

    output_dir = os.path.join(os.getcwd(), "zenith_outputs", "mvp2")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Test Multi-resolution pyramid
    print("\n[Step 2: REPRESENT - Multi-Resolution Pyramid Verification]")
    pyramid_engine = ImagePyramid(num_levels=3, scale_factor=0.5)
    pyr_ref = pyramid_engine.build_pyramid(img_ref)
    pyr_src = pyramid_engine.build_pyramid(img_src)
    for lvl in pyr_ref:
        print(f"  Pyramid Level {lvl['level']}: Shape {lvl['shape']}, Scale {lvl['scale']:.2f}x")

    # 3. Test Phase Congruency vs Raw Under Illumination Gap
    print("\n[Step 3: REPRESENT - Phase Congruency Execution]")
    pipeline_pc = ZenithPipeline(
        matcher_mode="rootsift_only",
        representation_mode="phase_congruency",
        pc_nscale=3,
        pc_norient=4,
        estimator_type="USAC_MAGSAC",
        sift_nfeatures=4000,
        sift_ratio=0.75,
        reproj_threshold=3.0,
        output_dir=output_dir
    )

    print("  Running Phase Congruency cross-modal registration...")
    res_pc = pipeline_pc.run_pair(
        img_ref=img_ref,
        img_src=img_src,
        H_gt=H_gt,
        experiment_id="mvp2_phase_congruency"
    )

    # Save Phase Congruency representation maps for visual verification
    rep_ref, rep_src = res_pc["rep_images"]
    cv2.imwrite(os.path.join(output_dir, "mvp2_pc_representation_ref.png"), rep_ref)
    cv2.imwrite(os.path.join(output_dir, "mvp2_pc_representation_src.png"), rep_src)

    # Also test Gradient Representation mode for comparison
    pipeline_grad = ZenithPipeline(
        matcher_mode="rootsift_only",
        representation_mode="gradient",
        estimator_type="USAC_MAGSAC",
        sift_nfeatures=4000,
        sift_ratio=0.75,
        reproj_threshold=3.0,
        output_dir=output_dir
    )
    res_grad = pipeline_grad.run_pair(
        img_ref=img_ref,
        img_src=img_src,
        H_gt=H_gt,
        experiment_id="mvp2_gradient"
    )

    # 4. Print Verification Table
    print("\n" + "=" * 75)
    print("  MVP2 RUN RESULTS (Measured from actual execution)")
    print("=" * 75)
    print(f"{'Representation Mode':<22} | {'Candidates':<10} | {'Inliers':<8} | {'Inlier %':<9} | {'Corner RMSE (px)':<16} | {'Status':<10}")
    print("-" * 75)
    for res in [res_pc, res_grad]:
        gt_info = res.get('gt_rmse_data') or res.get('gt_comparison') or {}
        if gt_info.get('ground_truth_available', False):
            val = gt_info.get('corner_rmse_px', gt_info.get('corner_rmse', gt_info.get('rmse_px', 0.0)))
            rmse_str = f"{val:.4f}"
        else:
            rmse_str = "N/A"
        print(f"{res['representation_mode']:<22} | {res['total_candidates']:<10} | {res['inlier_count']:<8} | {res['inlier_ratio']*100:>6.2f}%  | {rmse_str:<16} | {res['status']:<10}")
    print("=" * 75)
    print(f"\nPhase Congruency Timing: Total {res_pc['timings_sec']['total']:.3f}s (Represent: {res_pc['timings_sec']['represent']:.3f}s, Match: {res_pc['timings_sec']['match']:.3f}s, Verify: {res_pc['timings_sec']['verify']:.3f}s)")
    print(f"Artifacts saved in: {output_dir}")

    # Write summary json
    summary_file = os.path.join(output_dir, "mvp2_run_summary.json")
    summary_data = {
        "phase_congruency_run": {k: v for k, v in res_pc.items() if k not in ["H_est", "rep_images"]},
        "gradient_run": {k: v for k, v in res_grad.items() if k not in ["H_est", "rep_images"]}
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved: {summary_file}")


if __name__ == "__main__":
    main()
