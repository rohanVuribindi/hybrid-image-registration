from typing import Tuple, Dict, Any, Optional
import time
import numpy as np

from zenith.adapt import generate_synthetic_lunar_pair
from zenith.represent import StructuralRepresenter
from zenith.match import HybridMatcher
from zenith.verify import GeometricVerifier
from zenith.refine import PointRefiner
from zenith.evaluate import compute_spatial_coverage, compute_ground_truth_rmse, ConfidenceEvaluator
from zenith.output import warp_image_to_reference, save_registration_artifacts


class ZenithPipeline:
    """
    Zenith Cross-Modal Image Registration Pipeline.
    Modular Architecture: ADAPT -> REPRESENT -> MATCH -> VERIFY -> REFINE -> EVALUATE -> OUTPUT
    """
    def __init__(
        self,
        representation_mode: str = "phase_congruency",
        matcher_mode: str = "fallback",
        pc_nscale: int = 3,
        pc_norient: int = 4,
        estimator_type: str = "USAC_MAGSAC",
        sift_nfeatures: int = 4000,
        sift_ratio: float = 0.75,
        loftr_conf_thresh: float = 0.25,
        lightglue_features: str = "aliked",
        roma_model_type: str = "tiny",
        reproj_threshold: float = 3.0,
        output_dir: str = "zenith_outputs"
    ):
        self.representer = StructuralRepresenter(
            mode=representation_mode,
            pc_nscale=pc_nscale,
            pc_norient=pc_norient
        )
        self.matcher = HybridMatcher(
            mode=matcher_mode,
            sift_nfeatures=sift_nfeatures,
            sift_ratio=sift_ratio,
            loftr_conf_thresh=loftr_conf_thresh,
            lightglue_features=lightglue_features,
            roma_model_type=roma_model_type
        )
        self.verifier = GeometricVerifier(estimator_type=estimator_type, reproj_threshold=reproj_threshold)
        self.refiner = PointRefiner(grid_size=(8, 8), max_pts_per_bin=5)
        self.evaluator = ConfidenceEvaluator(
            high_min_inliers=15,
            high_min_inlier_ratio=0.30,
            high_min_coverage=0.25,
            high_max_residual=2.5,
            high_max_rmse=3.0
        )
        self.output_dir = output_dir

    def run_pair(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray,
        H_gt: Optional[np.ndarray] = None,
        experiment_id: str = "mvp5_test"
    ) -> Dict[str, Any]:
        """
        Executes registration end-to-end through REPRESENT -> MATCH -> VERIFY -> REFINE -> EVALUATE -> OUTPUT.
        """
        start_time = time.perf_counter()
        
        # 1. REPRESENT: Transform raw radiometric images into invariant structural maps
        t_rep_start = time.perf_counter()
        rep_ref, rep_ref_meta = self.representer.transform(img_ref)
        rep_src, rep_src_meta = self.representer.transform(img_src)
        rep_time = time.perf_counter() - t_rep_start

        # 2. MATCH: Hybrid matching (RootSIFT with automatic LoFTR fallback / escalation)
        t_match_start = time.perf_counter()
        kp_ref, kp_src, matches, pts_ref, pts_src, match_meta = self.matcher.match(rep_ref, rep_src)
        match_time = time.perf_counter() - t_match_start

        # 3. VERIFY: Robust estimation via USAC_MAGSAC (src -> ref)
        t_verify_start = time.perf_counter()
        H_est, inlier_mask, verify_diag = self.verifier.verify(pts_src, pts_ref)
        verify_time = time.perf_counter() - t_verify_start

        # 4. REFINE: Spatial balancing + Sub-pixel refinement (surviving inliers ONLY)
        t_refine_start = time.perf_counter()
        refine_res = self.refiner.refine(
            img_ref=img_ref,
            img_src=img_src,
            pts_ref=pts_ref,
            pts_src=pts_src,
            inlier_mask=inlier_mask,
            H_est=H_est,
            raw_residuals=verify_diag.get("residuals", np.zeros(len(pts_ref)))
        )
        H_final = refine_res["H_refined"] if refine_res["H_refined"] is not None else H_est
        refine_time = time.perf_counter() - t_refine_start

        # 5. EVALUATE: Compute metrics, spatial coverage, and confidence status
        t_eval_start = time.perf_counter()
        eval_pts = refine_res["pts_ref_refined"] if len(refine_res["pts_ref_refined"]) > 0 else (pts_ref[inlier_mask] if inlier_mask is not None and len(inlier_mask) > 0 else np.empty((0, 2)))
        spatial_cov = compute_spatial_coverage(eval_pts, img_ref.shape)
        
        gt_rmse_data = {}
        if H_gt is not None and H_final is not None:
            gt_rmse_data = compute_ground_truth_rmse(H_final, H_gt, img_ref.shape)
        else:
            gt_rmse_data = {"ground_truth_available": False}

        inlier_count = verify_diag.get("inlier_count", 0)
        inlier_ratio = verify_diag.get("inlier_ratio", 0.0)
        mean_res = refine_res.get("mean_refined_residual_px", verify_diag.get("mean_inlier_residual_px", 0.0))

        eval_decision = self.evaluator.evaluate(
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            spatial_coverage=spatial_cov,
            mean_residual_px=mean_res,
            gt_rmse_data=gt_rmse_data,
            H_est=H_final
        )
        eval_time = time.perf_counter() - t_eval_start
        total_time = time.perf_counter() - start_time

        # 6. OUTPUT: Warp source to reference frame (Lanczos-4 high-fidelity resampling)
        if H_final is not None:
            img_warped = warp_image_to_reference(img_src, H_final, img_ref.shape)
        else:
            img_warped = np.zeros_like(img_ref)

        # 7. ENHANCE (Optional Visualization Post-Processing - Never alters H or geometry)
        from zenith.enhance import enhance_registered_image, calculate_quality_statistics
        img_enhanced = enhance_registered_image(img_warped, method="conservative")
        quality_stats = calculate_quality_statistics(img_warped, img_enhanced)

        # Build point confidences array for all candidates to export to CSV
        full_residuals = np.zeros(len(pts_ref))
        full_confidences = np.zeros(len(pts_ref))
        if len(refine_res["residuals"]) > 0:
            full_residuals[:len(refine_res['residuals'])] = refine_res['residuals']
            full_confidences[:len(refine_res['confidences'])] = refine_res['confidences']

        # Save artifacts
        saved_files = save_registration_artifacts(
            output_dir=self.output_dir,
            img_ref=img_ref,
            img_src=img_src,
            img_warped=img_warped,
            pts_ref=pts_ref,
            pts_src=pts_src,
            inlier_mask=inlier_mask,
            diagnostics={
                **verify_diag,
                "matcher_meta": match_meta,
                "residuals": full_residuals,
                "confidences": full_confidences,
                "refinement": {
                    "balanced_count": refine_res["balanced_count"],
                    "mean_refined_residual_px": refine_res["mean_refined_residual_px"]
                },
                "spatial_coverage": spatial_cov,
                "ground_truth_eval": gt_rmse_data,
                "confidence_decision": eval_decision,
                "image_quality_statistics": quality_stats,
                "representation": {
                    "mode": self.representer.mode,
                    "ref_meta": rep_ref_meta,
                    "src_meta": rep_src_meta
                },
                "timings": {
                    "represent_sec": rep_time,
                    "match_sec": match_time,
                    "verify_sec": verify_time,
                    "refine_sec": refine_time,
                    "eval_sec": eval_time,
                    "total_sec": total_time
                }
            },
            H_est=H_final,
            prefix=experiment_id,
            img_enhanced=img_enhanced
        )

        return {
            "experiment_id": experiment_id,
            "status": eval_decision["status"],
            "confidence_score": eval_decision["confidence_score"],
            "summary_reason": eval_decision["summary_reason"],
            "detailed_reasons": eval_decision["detailed_reasons"],
            "matcher_meta": match_meta,
            "matcher_used": match_meta.get("matcher_used", "Unknown"),
            "representation_mode": self.representer.mode,
            "estimator_name": verify_diag["estimator_name"],
            "total_candidates": len(pts_ref),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "balanced_count": refine_res["balanced_count"],
            "mean_inlier_residual_px": mean_res,
            "spatial_coverage": spatial_cov,
            "gt_rmse_data": gt_rmse_data,
            "gt_comparison": gt_rmse_data,
            "image_quality_statistics": quality_stats,
            "timings_sec": {
                "represent": round(rep_time, 4),
                "match": round(match_time, 4),
                "verify": round(verify_time, 4),
                "refine": round(refine_time, 4),
                "eval": round(eval_time, 4),
                "total": round(total_time, 4)
            },
            "rep_images": (rep_ref, rep_src),
            "H_est": H_final,
            "saved_files": saved_files
        }
