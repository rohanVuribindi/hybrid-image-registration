"""
Structural and gradient representation filters for illumination invariance.
"""
from typing import Tuple
import numpy as np
import cv2


def compute_gradient_representation(
    image: np.ndarray,
    ksize: int = 3,
    use_scharr: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes normalized gradient magnitude and gradient orientation map.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    gray_f = gray.astype(np.float32)
    
    if use_scharr:
        gx = cv2.Scharr(gray_f, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray_f, cv2.CV_32F, 0, 1)
    else:
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=ksize)
        
    mag = cv2.magnitude(gx, gy)
    ori = cv2.phase(gx, gy, angleInDegrees=True)
    
    # Normalize magnitude to uint8
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ori_norm = ((ori / 360.0) * 255.0).astype(np.uint8)
    
    return mag_norm, ori_norm


def apply_adaptive_contrast_enhancement(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to balance severe lunar shadow-terminator illumination gradients.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)
