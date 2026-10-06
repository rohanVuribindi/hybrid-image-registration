"""
Runner script for ZENITH MVP3: Comprehensive Evaluation & Confidence Layer Verification.
Executes test cases across varying difficulty levels and validates REGISTERED / LOW CONFIDENCE / FAILED states.
"""
import os
import sys
import json
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenith.adapt import generate_synthetic_lunar_pair
from zenith.pipeline import ZenithPipeline


def main():
    print("=" * 80)
    print("  ZENITH CROSS-MODAL REGISTRATION - MVP3 (EVALUATION & CONFIDENCE LAYER)")
    print("=" * 80)

    output_dir = os.path.join(os.getcwd(), "zenith_outputs", "mvp3")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = ZenithPipeline(
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

    test_cases = []

    # -------------------------------------------------------------
    # Case 1: Standard Cross-Sensor Lunar Pair (Nominal / Clean)
    # -------------------------------------------------------------
    print("\n[Case 1: Nominal Synthetic Lunar Pair]")
    ref_1, src_1, H_gt_1, meta_1 = generate_synthetic_lunar_pair(
        rotation_deg=14.0,
        scale=1.08,
        translation=(20.0, -15.0),
        perspective_shear=(0.00008, -0.00006),
        seed=10
    )
    res_1 = pipeline.run_pair(ref_1, src_1, H_gt_1, experiment_id="mvp3_case1_nominal")
    test_cases.append(("Case 1 (Nominal)", res_1))

    # -------------------------------------------------------------
    # Case 2: Severe Terminator Shadow & Illumination Gradient
    # -------------------------------------------------------------
    print("\n[Case 2: Severe Cross-Sensor Illumination Ramp]")
    ref_2, src_clean_2, H_gt_2, meta_2 = generate_synthetic_lunar_pair(
        rotation_deg=25.0,
        scale=1.20,
        translation=(35.0, -25.0),
        perspective_shear=(0.00015, -0.00010),
        seed=20
    )
    # Apply severe diagonal illumination ramp + gamma distortion
    h, w = src_clean_2.shape[:2]
    y_g, x_g = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
    src_2 = np.clip(np.power((src_clean_2 / 255.0) * (0.3 + 0.7 * (0.7*x_g + 0.3*y_g)), 1.5) * 255.0, 0, 255).astype(np.uint8)
    
    res_2 = pipeline.run_pair(ref_2, src_2, H_gt_2, experiment_id="mvp3_case2_illumination")
    test_cases.append(("Case 2 (Illumination)", res_2))

    # -------------------------------------------------------------
    # Case 3: Extreme Scale Gap / Low Overlap (Controlled Stress Test)
    # -------------------------------------------------------------
    print("\n[Case 3: Extreme Transformation / Low Overlap Stress Test]")
    ref_3, src_3, H_gt_3, meta_3 = generate_synthetic_lunar_pair(
        rotation_deg=115.0, # Extreme rotation
        scale=2.85,         # Severe scale mismatch
        translation=(180.0, -150.0),
        perspective_shear=(0.00030, -0.00025),
        seed=30
    )
    res_3 = pipeline.run_pair(ref_3, src_3, H_gt_3, experiment_id="mvp3_case3_extreme_stress")
    test_cases.append(("Case 3 (Extreme Stress)", res_3))

    # -------------------------------------------------------------
    # Print Evaluation Matrix with Actual Measured Numbers
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  ZENITH MVP3 EVALUATION MATRIX (Measured Values from Live Execution)")
    print("=" * 90)
    header = f"{'Case':<22} | {'Candidates':<10} | {'Inliers':<8} | {'Inlier %':<9} | {'Coverage':<9} | {'RMSE (px)':<10} | {'Status':<15}"
    print(header)
    print("-" * 90)
    
    for case_name, res in test_cases:
        c_score = res["spatial_coverage"]["coverage_score"]
        rmse_val = res["gt_rmse_data"].get("rmse_px")
        rmse_str = f"{rmse_val:.3f}" if rmse_val is not None and rmse_val >= 0 else "N/A"
        row = (
            f"{case_name:<22} | "
            f"{res['total_candidates']:<10} | "
            f"{res['inlier_count']:<8} | "
            f"{res['inlier_ratio']*100:>6.2f}%  | "
            f"{c_score:>6.2f}    | "
            f"{rmse_str:<10} | "
            f"{res['status']:<15}"
        )
        print(row)
    print("=" * 90)

    # Print Explanations
    print("\n[Status Justifications]")
    for case_name, res in test_cases:
        print(f"\n* {case_name}:")
        print(f"  - Status: {res['status']} (Confidence Score: {res['confidence_score']:.2f})")
        print(f"  - Summary: {res['summary_reason']}")
        if res.get("detailed_reasons"):
            for dr in res["detailed_reasons"]:
                print(f"    * {dr}")

    # Save summary json
    summary_file = os.path.join(output_dir, "mvp3_evaluation_matrix.json")
    summary_dict = {
        name: {k: v for k, v in r.items() if k not in ["H_est", "rep_images"]}
        for name, r in test_cases
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\nEvaluation Matrix JSON saved to: {summary_file}")


if __name__ == "__main__":
    main()
