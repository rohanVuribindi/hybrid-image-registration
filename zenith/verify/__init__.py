"""
VERIFY module for Zenith pipeline.
Provides robust geometric estimation (USAC_MAGSAC / RANSAC) for inlier verification.
"""
from .estimator import GeometricVerifier, verify_correspondences

__all__ = ["GeometricVerifier", "verify_correspondences"]
