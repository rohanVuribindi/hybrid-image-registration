"""
Confidence scoring and status classification engine.
Maps validation metrics to REGISTERED / LOW CONFIDENCE / FAILED with explicit justifications.
"""
from typing import Dict, Any, List, Optional
import numpy as np


class ConfidenceEvaluator:
    """
    Evaluates registration quality and assigns a categorical confidence status.
    
    Documented Thresholds:
      High Confidence (REGISTERED):
        - Inlier count >= 15
        - Inlier ratio >= 30% (0.30)
        - Spatial Coverage Score >= 0.25
        - Mean inlier residual <= 2.5 px
        - Ground truth RMSE <= 3.0 px (if GT available)
        
      Low Confidence (LOW CONFIDENCE):
        - Inlier count >= 4 (minimal solvable homography)
        - Inlier ratio >= 12% (0.12)
        - Spatial Coverage Score >= 0.08
        - Mean inlier residual <= 5.0 px
        
      Failed (FAILED):
        - Any condition below Low Confidence thresholds or non-invertible transformation.
    """
    def __init__(
        self,
        high_min_inliers: int = 15,
        high_min_inlier_ratio: float = 0.30,
        high_min_coverage: float = 0.25,
        high_max_residual: float = 2.5,
        high_max_rmse: float = 3.0,
        low_min_inliers: int = 4,
        low_min_inlier_ratio: float = 0.12,
        low_min_coverage: float = 0.08,
        low_max_residual: float = 5.0
    ):
        self.high_min_inliers = high_min_inliers
        self.high_min_inlier_ratio = high_min_inlier_ratio
        self.high_min_coverage = high_min_coverage
        self.high_max_residual = high_max_residual
        self.high_max_rmse = high_max_rmse
        
        self.low_min_inliers = low_min_inliers
        self.low_min_inlier_ratio = low_min_inlier_ratio
        self.low_min_coverage = low_min_coverage
        self.low_max_residual = low_max_residual

    def evaluate(
        self,
        inlier_count: int,
        inlier_ratio: float,
        spatial_coverage: Dict[str, float],
        mean_residual_px: float,
        gt_rmse_data: Optional[Dict[str, float]] = None,
        H_est: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Computes composite confidence score [0.0 - 1.0] and assigns explicit categorical status.
        """
        reasons: List[str] = []
        coverage_score = spatial_coverage.get("coverage_score", 0.0)
        grid_cov = spatial_coverage.get("grid_coverage_ratio", 0.0)
        
        # Check basic solvability
        if H_est is None or inlier_count < self.low_min_inliers:
            return {
                "status": "FAILED",
                "confidence_score": 0.0,
                "reasons": [f"Insufficient verified inliers ({inlier_count} < {self.low_min_inliers}) to estimate transformation."],
                "thresholds_met": {"high": False, "low": False}
            }

        # Check conditioning of H_est
        try:
            det = np.linalg.det(H_est)
            if abs(det) < 1e-7 or abs(det) > 1e7 or np.isnan(det):
                return {
                    "status": "FAILED",
                    "confidence_score": 0.05,
                    "reasons": [f"Estimated transformation is ill-conditioned (determinant: {det:.2e})."],
                    "thresholds_met": {"high": False, "low": False}
                }
        except Exception:
            return {
                "status": "FAILED",
                "confidence_score": 0.0,
                "reasons": ["Estimated transformation matrix is singular or invalid."],
                "thresholds_met": {"high": False, "low": False}
            }

        # Evaluate High Confidence criteria
        high_passes = True
        
        if inlier_count < self.high_min_inliers:
            high_passes = False
            reasons.append(f"Inlier count ({inlier_count}) below high-confidence threshold ({self.high_min_inliers}).")
            
        if inlier_ratio < self.high_min_inlier_ratio:
            high_passes = False
            reasons.append(f"Inlier ratio ({inlier_ratio*100:.1f}%) below high-confidence threshold ({self.high_min_inlier_ratio*100:.1f}%).")
            
        if coverage_score < self.high_min_coverage:
            high_passes = False
            reasons.append(f"Spatial coverage score ({coverage_score:.2f}) indicates localized clustering below threshold ({self.high_min_coverage:.2f}).")
            
        if mean_residual_px > self.high_max_residual:
            high_passes = False
            reasons.append(f"Mean reprojection residual ({mean_residual_px:.2f}px) exceeds threshold ({self.high_max_residual:.2f}px).")

        gt_rmse = -1.0
        if gt_rmse_data and gt_rmse_data.get("ground_truth_available", False):
            gt_rmse = gt_rmse_data.get("rmse_px", -1.0)
            if gt_rmse > self.high_max_rmse:
                high_passes = False
                reasons.append(f"Ground-truth transfer RMSE ({gt_rmse:.2f}px) exceeds tolerance ({self.high_max_rmse:.2f}px).")

        # Evaluate Low Confidence criteria
        low_passes = True
        if inlier_count < self.low_min_inliers:
            low_passes = False
        if inlier_ratio < self.low_min_inlier_ratio:
            low_passes = False
            reasons.append(f"Inlier ratio ({inlier_ratio*100:.1f}%) below acceptable minimum ({self.low_min_inlier_ratio*100:.1f}%).")
        if coverage_score < self.low_min_coverage:
            low_passes = False
            reasons.append(f"Spatial coverage ({coverage_score:.2f}) critically low (< {self.low_min_coverage:.2f}).")
        if mean_residual_px > self.low_max_residual:
            low_passes = False
            reasons.append(f"Mean residual ({mean_residual_px:.2f}px) exceeds allowable error ({self.low_max_residual:.2f}px).")

        # Continuous confidence metric calculation (0.0 to 1.0)
        w_inliers = min(1.0, inlier_count / 50.0) * 0.25
        w_ratio = min(1.0, inlier_ratio / 0.8) * 0.25
        w_cov = min(1.0, coverage_score / 0.6) * 0.25
        w_res = max(0.0, 1.0 - (mean_residual_px / 4.0)) * 0.25
        confidence_score = float(np.clip(w_inliers + w_ratio + w_cov + w_res, 0.0, 1.0))

        if high_passes:
            status = "REGISTERED"
            summary_reason = f"High confidence registration achieved: {inlier_count} inliers ({inlier_ratio*100:.1f}%), spatial coverage {coverage_score:.2f}, mean residual {mean_residual_px:.2f}px."
        elif low_passes:
            status = "LOW CONFIDENCE"
            summary_reason = f"Low confidence registration: Marginal fit ({'; '.join(reasons)})."
        else:
            status = "FAILED"
            summary_reason = f"Registration failed verification criteria ({'; '.join(reasons)})."

        return {
            "status": status,
            "confidence_score": round(confidence_score, 4),
            "summary_reason": summary_reason,
            "detailed_reasons": reasons,
            "thresholds_met": {
                "high": high_passes,
                "low": low_passes
            },
            "metrics": {
                "inlier_count": inlier_count,
                "inlier_ratio": round(inlier_ratio, 4),
                "coverage_score": round(coverage_score, 4),
                "grid_coverage_ratio": round(grid_cov, 4),
                "mean_residual_px": round(mean_residual_px, 4),
                "gt_rmse_px": round(gt_rmse, 4) if gt_rmse >= 0 else None
            }
        }
