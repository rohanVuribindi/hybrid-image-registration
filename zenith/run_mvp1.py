"""
Runner script for ZENITH MVP1: End-to-end core registration loop verification.
"""
import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenith.adapt import generate_synthetic_lunar_pair
from zenith.pipeline import ZenithPipeline


def main():
    print("=" * 70)
    print("  ZENITH CROSS-MODAL REGISTRATION - MVP1 VERIFICATION")
    print("=" * 70)

    # 1. Generate procedural lunar surface with known ground truth transform
    print("\n[Step 1: ADAPT] Generating synthetic lunar surface with known ground truth...")
    img_ref, img_src, H_gt, meta = generate_synthetic_lunar_pair(
        rotation_deg=18.5,
        scale=1.12,
        translation=(28.0, -16.0),
        perspective_shear=(0.00012, -0.00008),
        output_size=(800, 800),
        seed=42
    )
    print(f"  Reference Image Shape: {img_ref.shape}")
    print(f"  Source Image Shape:    {img_src.shape}")
    print(f"  Ground Truth Rotation: {meta['rotation_deg']} deg, Scale: {meta['scale']}x")
    print(f"  Ground Truth Translation: {meta['translation']} px")

    # 2. Instantiate and run pipeline (Strict classical MVP1: RootSIFT + USAC_MAGSAC)
    output_dir = os.path.join(os.getcwd(), "zenith_outputs", "mvp1")
    pipeline = ZenithPipeline(
        matcher_mode="rootsift_only",
        representation_mode="raw",
        estimator_type="USAC_MAGSAC",
        sift_nfeatures=4000,
        sift_ratio=0.75,
        reproj_threshold=3.0,
        output_dir=output_dir
    )

    print("\n[Step 2: MATCH + VERIFY + OUTPUT] Running Zenith Pipeline...")
    results = pipeline.run_pair(
        img_ref=img_ref,
        img_src=img_src,
        H_gt=H_gt,
        experiment_id="mvp1_synthetic_lunar"
    )

    # 3. Print verification results
    print("\n" + "=" * 70)
    print("  MVP1 RUN RESULTS (Measured values from actual execution)")
    print("=" * 70)
    print(f"  Estimator Name:           {results['estimator_name']}")
    print(f"  Status:                   {results['status']}")
    print(f"  Total Candidate Matches:  {results['total_candidates']}")
    print(f"  Verified Inliers:         {results['inlier_count']}")
    print(f"  Inlier Ratio:             {results['inlier_ratio'] * 100:.2f}%")
    print(f"  Mean Inlier Residual:     {results['mean_inlier_residual_px']:.3f} px")
    
    gt_info = results.get('gt_rmse_data') or results.get('gt_comparison') or {}
    if gt_info.get('ground_truth_available', False):
        rmse_val = gt_info.get('rmse_px', gt_info.get('corner_rmse_px', 0.0))
        max_err = gt_info.get('max_error_px', gt_info.get('max_corner_error_px', 0.0))
        print(f"  Ground Truth RMSE:        {rmse_val:.4f} px")
        print(f"  Max Point Transfer Error: {max_err:.4f} px")
        
    print(f"  Execution Time:           {results['timings_sec']['total']:.3f}s (Match: {results['timings_sec']['match']:.3f}s, Verify: {results['timings_sec']['verify']:.3f}s)")
    print("\n  Generated Artifacts:")
    for k, v in results['saved_files'].items():
        print(f"    - {k}: {v}")
    print("=" * 70)

    # Save summary report
    summary_file = os.path.join(output_dir, "mvp1_run_summary.json")
    clean_results = {
        k: v for k, v in results.items() if k not in ["H_est", "rep_images"]
    }
    clean_results["H_est_matrix"] = results["H_est"].tolist() if results["H_est"] is not None else None
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nRun summary written to: {summary_file}")


if __name__ == "__main__":
    main()
