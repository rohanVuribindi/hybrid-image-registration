"""
ZENITH MVP8: Multi-Matcher Consensus Ensemble Benchmark.
Direct empirical comparison of RootSIFT, LoFTR, LightGlue, RoMa, and Multi-Matcher Ensemble
across identical difficult lunar image pairs.
"""
import os
import sys
import json
import time
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenith.adapt import generate_synthetic_lunar_pair, LunarDatasetAdapter
from zenith.pipeline import ZenithPipeline


def main():
    print("=" * 115)
    print("  ZENITH MVP8: MULTI-MATCHER ENSEMBLE BENCHMARK & COMPARATIVE EVALUATION")
    print("=" * 115)

    output_dir = os.path.join(os.getcwd(), "zenith_outputs", "mvp8")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Test Scenarios Setup
    # -------------------------------------------------------------------------
    # Scenario A: Moderate Perspective Warp + Illumination Gradient
    ref_a, src_a, H_gt_a, _ = generate_synthetic_lunar_pair(
        rotation_deg=20.0,
        scale=1.16,
        translation=(35.0, -22.0),
        perspective_shear=(0.00014, -0.00010),
        seed=404
    )

    # Scenario B: Low-Texture Smooth Mare Basin (Challenging)
    ref_b_raw, src_b_raw, H_gt_b, _ = generate_synthetic_lunar_pair(
        rotation_deg=25.0,
        scale=1.20,
        translation=(40.0, -28.0),
        seed=707
    )
    ref_b = cv2.GaussianBlur(ref_b_raw, (7, 7), 2.0)
    src_b = cv2.GaussianBlur(src_b_raw, (7, 7), 2.0)

    # Scenario C: Chandrayaan-2 Real Sensor Pair (OHRC vs TMC-2)
    ref_c, src_c, H_gt_c, _ = LunarDatasetAdapter.load_or_create_lunar_sensor_pair(
        sensor_ref="OHRC", sensor_src="TMC-2", region_name="Mare_Tranquillitatis",
        scale_ratio=1.15, rotation_deg=14.0, seed=101
    )

    scenarios = [
        ("Perspective Warp", ref_a, src_a, H_gt_a),
        ("Low-Texture Mare", ref_b, src_b, H_gt_b),
        ("OHRC vs TMC-2", ref_c, src_c, H_gt_c)
    ]

    matcher_modes = [
        ("RootSIFT", "rootsift_only"),
        ("LoFTR", "loftr_only"),
        ("LightGlue (ALIKED)", "lightglue_only"),
        ("RoMa (Dense)", "roma_only"),
        ("Multi-Matcher Ensemble", "ensemble")
    ]

    all_results = {}
    table_rows = []

    print("\nExecuting comprehensive benchmark across all matchers...")

    for sc_name, ref_img, src_img, H_gt in scenarios:
        print(f"\n--- Scenario: {sc_name} ---")
        for m_label, m_mode in matcher_modes:
            exp_id = f"mvp8_{sc_name.lower().replace(' ', '_').replace('-', '_')}_{m_mode}"
            t0 = time.perf_counter()
            
            pipeline = ZenithPipeline(
                matcher_mode=m_mode,
                representation_mode="phase_congruency" if m_mode != "rootsift_only" else "raw",
                output_dir=output_dir
            )
            res = pipeline.run_pair(ref_img, src_img, H_gt=H_gt, experiment_id=exp_id)
            runtime = time.perf_counter() - t0

            rmse_val = res["gt_rmse_data"].get("rmse_px")
            rmse_str = f"{rmse_val:.3f}" if rmse_val is not None and rmse_val >= 0 else "N/A"
            cov_val = res["spatial_coverage"].get("coverage_score", 0.0)

            print(f"  [{m_label:<22}] Inliers: {res['inlier_count']:<5} | Inlier %: {res['inlier_ratio']*100:>5.1f}% | RMSE: {rmse_str:<7} | Cov: {cov_val:.2f} | Status: {res['status']}")

            table_rows.append({
                "Scenario": sc_name,
                "Matcher": m_label,
                "Inliers": res["inlier_count"],
                "Inlier %": round(res["inlier_ratio"] * 100, 2),
                "RMSE (px)": round(rmse_val, 4) if rmse_val is not None else None,
                "Coverage": round(cov_val, 3),
                "Runtime (s)": round(runtime, 2),
                "Status": res["status"]
            })

            all_results[f"{sc_name} :: {m_label}"] = {
                k: v for k, v in res.items() if k not in ["H_est", "rep_images"]
            }

    # -------------------------------------------------------------------------
    # Final Benchmark Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  ZENITH MVP8 MULTI-MATCHER ENSEMBLE BENCHMARK RESULTS (Live Executions Only)")
    print("=" * 115)
    header = f"{'Scenario':<18} | {'Matcher':<23} | {'Inliers':<8} | {'Inlier %':<9} | {'RMSE (px)':<10} | {'Coverage':<9} | {'Runtime':<8} | {'Status':<12}"
    print(header)
    print("-" * 115)
    for r in table_rows:
        rmse_s = f"{r['RMSE (px)']:.3f}" if r['RMSE (px)'] is not None else "N/A"
        print(f"{r['Scenario']:<18} | {r['Matcher']:<23} | {r['Inliers']:<8} | {r['Inlier %']:>6.2f}%  | {rmse_s:<10} | {r['Coverage']:<9.2f} | {r['Runtime (s)']:<6.2f}s | {r['Status']:<12}")
    print("=" * 115)

    # Save summary json
    summary_file = os.path.join(output_dir, "mvp8_ensemble_benchmark.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_records": table_rows,
            "detailed_runs": all_results
        }, f, indent=2)
    print(f"\nBenchmark results saved to: {summary_file}")


if __name__ == "__main__":
    main()
