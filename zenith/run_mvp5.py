"""
Runner script for ZENITH MVP5: LoFTR Transformer Matching & Intelligent Fallback Logic.
Tests primary RootSIFT -> LoFTR fallback escalation and fusion.
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
    print("=" * 85)
    print("  ZENITH CROSS-MODAL REGISTRATION - MVP5 (LoFTR MATCHING & FALLBACK LOGIC)")
    print("=" * 85)

    output_dir = os.path.join(os.getcwd(), "zenith_outputs", "mvp5")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Test 1: Easy/Moderate Case -> Handled by RootSIFT Primary (Fast)
    # -------------------------------------------------------------------------
    print("\n[Test 1: Standard Lunar Surface -> Fallback Pipeline]")
    ref_1, src_1, H_gt_1, meta_1 = generate_synthetic_lunar_pair(
        rotation_deg=12.0,
        scale=1.06,
        translation=(15.0, -10.0),
        seed=101
    )
    pipe_fallback = ZenithPipeline(
        matcher_mode="fallback",
        representation_mode="phase_congruency",
        output_dir=output_dir
    )
    res_1 = pipe_fallback.run_pair(ref_1, src_1, H_gt_1, experiment_id="mvp5_test1_rootsift_primary")

    # -------------------------------------------------------------------------
    # Test 2: Low-Texture / Severe Cross-Sensor Warp -> Escalation to LoFTR
    # -------------------------------------------------------------------------
    print("\n[Test 2: Low-Texture Smooth Mare Basin -> Fallback Escalation to LoFTR]")
    # Generate low-texture mare region with few craters
    ref_2_raw, src_2_raw, H_gt_2, meta_2 = generate_synthetic_lunar_pair(
        rotation_deg=28.0,
        scale=1.22,
        translation=(40.0, -30.0),
        perspective_shear=(0.00018, -0.00015),
        seed=505
    )
    # Smooth/blur heavily to simulate smooth lunar mare / low feature density
    ref_2 = cv2.GaussianBlur(ref_2_raw, (9, 9), 2.5)
    src_2 = cv2.GaussianBlur(src_2_raw, (9, 9), 2.5)

    res_2 = pipe_fallback.run_pair(ref_2, src_2, H_gt_2, experiment_id="mvp5_test2_loftr_fallback")

    # -------------------------------------------------------------------------
    # Test 3: Standalone LoFTR Transformer
    # -------------------------------------------------------------------------
    print("\n[Test 3: Standalone LoFTR Transformer on Challenging Pair]")
    pipe_loftr = ZenithPipeline(
        matcher_mode="loftr_only",
        representation_mode="raw",  # LoFTR operates natively on normalized intensity patches
        loftr_conf_thresh=0.20,
        output_dir=output_dir
    )
    res_3 = pipe_loftr.run_pair(ref_2, src_2, H_gt_2, experiment_id="mvp5_test3_loftr_standalone")

    # -------------------------------------------------------------------------
    # Test 4: Evidence Fusion (RootSIFT + LoFTR)
    # -------------------------------------------------------------------------
    print("\n[Test 4: Evidence Fusion (RootSIFT + LoFTR)]")
    pipe_fusion = ZenithPipeline(
        matcher_mode="fusion",
        representation_mode="phase_congruency",
        loftr_conf_thresh=0.20,
        output_dir=output_dir
    )
    res_4 = pipe_fusion.run_pair(ref_1, src_1, H_gt_1, experiment_id="mvp5_test4_evidence_fusion")

    # -------------------------------------------------------------------------
    # Results Matrix from Actual Executions
    # -------------------------------------------------------------------------
    results_list = [
        ("Test 1 (Standard)", res_1),
        ("Test 2 (Low Texture)", res_2),
        ("Test 3 (LoFTR Standalone)", res_3),
        ("Test 4 (Fusion)", res_4)
    ]

    print("\n" + "=" * 95)
    print("  MVP5 MEASURED EXECUTION RESULTS (Live Runs Only)")
    print("=" * 95)
    header = f"{'Test Case':<22} | {'Matcher Used':<26} | {'Inliers':<8} | {'Inlier %':<9} | {'RMSE (px)':<10} | {'Status':<12}"
    print(header)
    print("-" * 95)
    for name, r in results_list:
        rmse_val = r["gt_rmse_data"].get("rmse_px")
        rmse_str = f"{rmse_val:.3f}" if rmse_val is not None and rmse_val >= 0 else "N/A"
        print(f"{name:<22} | {r['matcher_used']:<26} | {r['inlier_count']:<8} | {r['inlier_ratio']*100:>6.2f}%  | {rmse_str:<10} | {r['status']:<12}")
    print("=" * 95)

    # Save summary json
    summary_file = os.path.join(output_dir, "mvp5_run_summary.json")
    summary_data = {
        name: {k: v for k, v in r.items() if k not in ["H_est", "rep_images"]}
        for name, r in results_list
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nRun summary written to: {summary_file}")


if __name__ == "__main__":
    main()
