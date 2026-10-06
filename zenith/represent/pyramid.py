"""
Multi-resolution image pyramid representation.
Enables coarse-to-fine registration across significant resolution / scale gaps.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2


class ImagePyramid:
    """
    Constructs and manages a multi-scale Gaussian or structural pyramid.
    """
    def __init__(self, num_levels: int = 3, scale_factor: float = 0.5):
        self.num_levels = num_levels
        self.scale_factor = scale_factor

    def build_pyramid(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Builds pyramid levels from Level 0 (original resolution) down to Level N-1 (coarsest).
        
        Returns a list of dicts:
            [
               {"level": 0, "scale": 1.0, "image": np.ndarray, "shape": (H, W)},
               {"level": 1, "scale": 0.5, "image": np.ndarray, "shape": (H/2, W/2)},
               ...
            ]
        """
        pyramid = []
        current_img = image.copy()
        current_scale = 1.0

        for lvl in range(self.num_levels):
            pyramid.append({
                "level": lvl,
                "scale": current_scale,
                "image": current_img,
                "shape": current_img.shape[:2]
            })
            
            if lvl < self.num_levels - 1:
                # Downsample with Gaussian smoothing
                h, w = current_img.shape[:2]
                new_w = max(int(w * self.scale_factor), 16)
                new_h = max(int(h * self.scale_factor), 16)
                
                # Smooth before downsampling to prevent aliasing
                smoothed = cv2.GaussianBlur(current_img, (5, 5), sigmaX=1.2, sigmaY=1.2)
                current_img = cv2.resize(smoothed, (new_w, new_h), interpolation=cv2.INTER_AREA)
                current_scale *= self.scale_factor

        return pyramid


def scale_coordinates_to_level(
    coords: np.ndarray,
    from_scale: float,
    to_scale: float
) -> np.ndarray:
    """
    Transforms point coordinates from one pyramid scale level to another.
    """
    if len(coords) == 0:
        return coords
    factor = to_scale / from_scale
    return coords * factor
