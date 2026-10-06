"""
EVALUATE module for Zenith pipeline.
Provides RMSE computation, spatial coverage metrics, and status classification.
"""
from .metrics import compute_spatial_coverage, compute_ground_truth_rmse
from .confidence import ConfidenceEvaluator

__all__ = [
    "compute_spatial_coverage",
    "compute_ground_truth_rmse",
    "ConfidenceEvaluator"
]
