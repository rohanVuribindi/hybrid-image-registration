"""
OUTPUT module for Zenith pipeline.
Provides warping, registration overlays, correspondence visualization, and artifact generation.
"""
from .warp import (
    warp_image_to_reference,
    create_side_by_side,
    create_match_visualization,
    create_overlay_blend,
    save_registration_artifacts
)

__all__ = [
    "warp_image_to_reference",
    "create_side_by_side",
    "create_match_visualization",
    "create_overlay_blend",
    "save_registration_artifacts"
]
