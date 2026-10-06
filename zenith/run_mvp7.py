"""
ZENITH MVP7: Final Evaluation Matrix & Controlled Honest Failure Benchmark.
Executes Easy, Difficult, and Hardest benchmark scenarios to produce verified, un-manipulated metrics.
"""
import os
import sys
import json
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenith.adapt import generate_synthetic_lunar_pair, LunarDatasetAdapter
from zenith.pipeline import ZenithPipeline


def main():
    print("=" * 110)
    print("  ZENITH CROSS-MODAL REGISTRATION - MVP7 FINAL EVALUATION & BENCHMARK SUITE")
    print("=" * 110)

    output_dir = os.path.join(os.getcwd(), "zenith_outputs", "mvp7")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = ZenithPipeline(
        matcher_mode="fallback",
        representation_mode="phase_congruency",
        output_dir=output_dir
    )

    evaluation_records = []

    # -------------------------------------------------------------------------
    # Scenario 1: Easy Case (Nominal Synthetic / OHRC High-Resolution Optical)
    # -------------------------------------------------------------------------
    print("\n[Scenario 1: Easy Case - Nominal Cross-Sensor Optical Lunar Surface]")
    ref_1, src_1, H_gt_1, meta_1 = generate_synthetic_lunar_pair(
        rotation_deg=12.0,
        scale=1.08,
        translation=(18.0, -12.0),
        perspective_shear=(0.00006, -0.00005),
        seed=111
    )
    res_1 = pipeline.run_pair(ref_1, src_1, H_gt_1, experiment_id="mvp7_case1_easy")
    evaluation_records.append({
        "Case": "Easy (Nominal)",
        "Scale Ratio": "1.08x",
        "Illumination": "Standard (Nominal)",
        "result": res_1
    })

    # -------------------------------------------------------------------------
    # Scenario 2: Difficult Case (Cross-Modal Real OHRC vs TMC-2 + Illumination Gradient)
    # -------------------------------------------------------------------------
    print("\n[Scenario 2: Difficult Case - Chandrayaan-2 OHRC vs TMC-2 with Severe Shadow Gradient]")
    ref_2_clean, src_2_clean, H_gt_2, meta_2 = LunarDatasetAdapter.load_or_create_lunar_sensor_pair(
        sensor_ref="OHRC",
        sensor_src="TMC-2",
        region_name="South_Pole_Aitken_Basin",
        scale_ratio=1.24,
        rotation_deg=22.0,
        translation=(35.0, -26.0),
        seed=222
    )
    # Add non-linear lighting ramp across terminator
    h, w = src_2_clean.shape[:2]
    yg, xg = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
    src_2 = np.clip(np.power(src_2_clean / 255.0, 1.4) * (0.35 + 0.65 * (0.6*xg + 0.4*yg)) * 255.0, 0, 255).astype(np.uint8)

    res_2 = pipeline.run_pair(ref_2_clean, src_2, H_gt_2, experiment_id="mvp7_case2_difficult")
    evaluation_records.append({
        "Case": "Difficult (Cross-Modal)",
        "Scale Ratio": "1.24x",
        "Illumination": "Terminator Ramp (gamma=1.4)",
        "result": res_2
    })

    # -------------------------------------------------------------------------
    # Scenario 3: Hardest Case (Extreme Scale Mismatch + Smooth Featureless Mare)
    # -------------------------------------------------------------------------
    print("\n[Scenario 3: Hardest Case - Extreme Scale Ratio 3.2x / Low-Texture Smooth Mare Basin]")
    ref_3_raw, src_3_raw, H_gt_3, meta_3 = generate_synthetic_lunar_pair(
        rotation_deg=85.0,
        scale=3.20,
        translation=(140.0, -110.0),
        perspective_shear=(0.00035, -0.00030),
        seed=333
    )
    # Smooth heavily to simulate smooth low-contrast basaltic mare with low feature density
    ref_3 = cv2.GaussianBlur(ref_3_raw, (11, 11), 3.0)
    src_3 = cv2.GaussianBlur(src_3_raw, (11, 11), 3.0)

    res_3 = pipeline.run_pair(ref_3, src_3, H_gt_3, experiment_id="mvp7_case3_hardest_failure")
    evaluation_records.append({
        "Case": "Hardest (Stress Failure)",
        "Scale Ratio": "3.20x",
        "Illumination": "Extreme Degraded Mare",
        "result": res_3
    })

    # -------------------------------------------------------------------------
    # Final Handbook Evaluation Matrix (Populated Exclusively from Real Executions)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  ZENITH FINAL EVALUATION MATRIX (Handbook Deliverable - Measured Runs Only)")
    print("=" * 115)
    header = f"{'Case':<26} | {'Scale Ratio':<12} | {'Illumination':<24} | {'Matcher':<20} | {'RMSE (px)':<10} | {'Inlier %':<9} | {'Coverage':<9} | {'Status':<15}"
    print(header)
    print("-" * 115)

    matrix_rows = []
    for entry in evaluation_records:
        r = entry["result"]
        rmse_val = r["gt_rmse_data"].get("rmse_px")
        rmse_str = f"{rmse_val:.3f}" if rmse_val is not None and rmse_val >= 0 else ""
        cov_score = r["spatial_coverage"]["coverage_score"]
        inlier_pct = r["inlier_ratio"] * 100.0
        
        row_str = (
            f"{entry['Case']:<26} | "
            f"{entry['Scale Ratio']:<12} | "
            f"{entry['Illumination']:<24} | "
            f"{r['matcher_used']:<20} | "
            f"{rmse_str:<10} | "
            f"{inlier_pct:>6.2f}%  | "
            f"{cov_score:>6.2f}    | "
            f"{r['status']:<15}"
        )
        print(row_str)
        
        matrix_rows.append({
            "Case": entry["Case"],
            "Scale Ratio": entry["Scale Ratio"],
            "Illumination": entry["Illumination"],
            "Matcher": r["matcher_used"],
            "RMSE": rmse_val if rmse_val is not None and rmse_val >= 0 else None,
            "Inlier %": round(inlier_pct, 2),
            "Coverage": round(cov_score, 2),
            "Status": r["status"],
            "Summary Reason": r["summary_reason"]
        })

    print("=" * 115)

    # -------------------------------------------------------------------------
    # Print Diagnostics & Explanations for Demo/PPT
    # -------------------------------------------------------------------------
    print("\n[Detailed Case Diagnostics]")
    for entry in evaluation_records:
        r = entry["result"]
        print(f"\n* Scenario: {entry['Case']}")
        print(f"  - Status Assigned:   {r['status']} (Confidence Score: {r['confidence_score']:.2f})")
        print(f"  - Matcher Selected:  {r['matcher_used']}")
        print(f"  - Total Candidates:  {r['total_candidates']}")
        print(f"  - Verified Inliers:  {r['inlier_count']} ({r['inlier_ratio']*100:.2f}%)")
        print(f"  - Balanced Inliers:  {r['balanced_count']}")
        print(f"  - Spatial Coverage:  {r['spatial_coverage']['coverage_score']:.3f} (Grid Occupancy: {r['spatial_coverage']['grid_coverage_ratio']*100:.1f}%)")
        if r['gt_rmse_data'].get('ground_truth_available'):
            print(f"  - Ground Truth RMSE: {r['gt_rmse_data']['rmse_px']:.4f} px")
        print(f"  - Causal Reason:     {r['summary_reason']}")

    # Save comprehensive final JSON report
    final_report_path = os.path.join(output_dir, "zenith_mvp7_final_report.json")
    final_report = {
        "pipeline_version": "1.0.0",
        "evaluation_matrix": matrix_rows,
        "scenarios": {
            entry["Case"]: {
                k: v for k, v in entry["result"].items() if k not in ["H_est", "rep_images"]
            }
            for entry in evaluation_records
        }
    }
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nFinal Zenith Report JSON saved to: {final_report_path}")


if __name__ == "__main__":
    main()
