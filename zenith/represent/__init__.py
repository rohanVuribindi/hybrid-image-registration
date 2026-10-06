"""
REPRESENT module for Zenith pipeline.
Provides Phase Congruency, structural gradient filtering, and multi-resolution pyramids.
"""
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

from .phase_congruency import compute_phase_congruency
from .structural import compute_gradient_representation, apply_adaptive_contrast_enhancement
from .pyramid import ImagePyramid, scale_coordinates_to_level


class StructuralRepresenter:
    """
    Transforms raw radiometric sensor images into contrast/illumination-invariant structural representations.
    """
    def __init__(
        self,
        mode: str = "phase_congruency",
        pc_nscale: int = 3,
        pc_norient: int = 4,
        pyramid_levels: int = 1
    ):
        self.mode = mode
        self.pc_nscale = pc_nscale
        self.pc_norient = pc_norient
        self.pyramid = ImagePyramid(num_levels=pyramid_levels)

    def transform(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Transforms single image into structural representation.
        """
        if self.mode == "phase_congruency":
            M, m, meta = compute_phase_congruency(
                image,
                nscale=self.pc_nscale,
                norient=self.pc_norient,
                normalize_uint8=True
            )
            return M, {"representation_mode": "phase_congruency", **meta}
            
        elif self.mode == "gradient":
            mag, ori = compute_gradient_representation(image)
            return mag, {"representation_mode": "gradient"}
            
        elif self.mode == "clahe":
            enhanced = apply_adaptive_contrast_enhancement(image)
            return enhanced, {"representation_mode": "clahe"}
            
        elif self.mode == "raw":
            return image.copy(), {"representation_mode": "raw"}
            
        else:
            raise ValueError(f"Unknown representation mode: {self.mode}")

    def build_pyramid(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Builds multi-resolution pyramid of representation images.
        """
        return self.pyramid.build_pyramid(image)


__all__ = [
    "compute_phase_congruency",
    "compute_gradient_representation",
    "apply_adaptive_contrast_enhancement",
    "ImagePyramid",
    "scale_coordinates_to_level",
    "StructuralRepresenter"
]
