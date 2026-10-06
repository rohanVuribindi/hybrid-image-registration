"""
Runner script for ZENITH MVP4: REFINE Module (Spatial Balancing + Sub-Pixel Refinement).
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
    print("=" * 80)
    print("  ZENITH CROSS-MODAL REGISTRATION - MVP4 (REFINE VERIFICATION)")
    print("=" * 80)

    output_dir = os.path.join(os.getcwd(), "zenith_outputs", "mvp4")
    os.makedirs(output_dir, exist_ok=True)

    # 1. ADAPT: Create realistic lunar cratered pair
    print("\n[Step 1: ADAPT] Generating synthetic lunar image pair with known ground truth...")
    ref_img, src_img, H_gt, meta = generate_synthetic_lunar_pair(
        rotation_deg=16.5,
        scale=1.10,
        translation=(24.0, -18.0),
        perspective_shear=(0.00010, -0.00008),
        output_size=(800, 800),
        seed=42
    )

    # 2. Execute Zenith Pipeline with REFINE
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

    print("\n[Step 2: EXECUTE] Running REPRESENT -> MATCH -> VERIFY -> REFINE -> EVALUATE -> OUTPUT...")
    res = pipeline.run_pair(
        img_ref=ref_img,
        img_src=src_img,
        H_gt=H_gt,
        experiment_id="mvp4_refined_lunar"
    )

    # 3. Print verification and measured statistics
    print("\n" + "=" * 80)
    print("  MVP4 MEASURED EXECUTION METRICS (From actual run)")
    print("=" * 80)
    print(f"  Estimator Algorithm:             {res['estimator_name']}")
    print(f"  Total Candidates (RootSIFT):     {res['total_candidates']}")
    print(f"  Verified Inliers (MAGSAC++):     {res['inlier_count']} ({res['inlier_ratio']*100:.2f}%)")
    print(f"  Spatially Balanced Inliers:      {res['balanced_count']}")
    print(f"  Sub-pixel Mean Residual:         {res['mean_inlier_residual_px']:.4f} px")
    print(f"  Spatial Coverage Score:          {res['spatial_coverage']['coverage_score']:.3f} (Grid: {res['spatial_coverage']['grid_coverage_ratio']*100:.1f}%)")
    if res['gt_rmse_data']['ground_truth_available']:
        print(f"  Ground-Truth Transfer RMSE:      {res['gt_rmse_data']['rmse_px']:.4f} px")
        print(f"  Max Transfer Error:              {res['gt_rmse_data']['max_error_px']:.4f} px")
    print(f"  Assigned Registration Status:    {res['status']} (Confidence: {res['confidence_score']:.2f})")
    print(f"  Summary Reason:                  {res['summary_reason']}")
    print(f"  Timings Breakdown:               Total: {res['timings_sec']['total']:.3f}s (Represent: {res['timings_sec']['represent']:.3f}s, Match: {res['timings_sec']['match']:.3f}s, Verify: {res['timings_sec']['verify']:.3f}s, Refine: {res['timings_sec']['refine']:.3f}s)")
    print("=" * 80)

    # 4. Inspect Correspondence CSV generated with attached confidence & residuals
    csv_path = res['saved_files']['correspondences_csv']
    print(f"\nVerifying correspondence CSV generated at: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [f.readline().strip() for _ in range(6)]
    print("  Sample CSV records:")
    for line in lines:
        print(f"    {line}")

    # Save summary
    summary_file = os.path.join(output_dir, "mvp4_run_summary.json")
    clean_res = {k: v for k, v in res.items() if k not in ["H_est", "rep_images"]}
    clean_res["H_refined_matrix"] = res["H_est"].tolist() if res["H_est"] is not None else None
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(clean_res, f, indent=2)
    print(f"\nSaved summary to: {summary_file}")


if __name__ == "__main__":
    main()
