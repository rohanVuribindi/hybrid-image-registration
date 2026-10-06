"""
Synthetic pair generator and image adapter for Zenith pipeline.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def generate_procedural_lunar_surface(
    width: int = 800,
    height: int = 800,
    num_craters: int = 60,
    seed: int = 42
) -> np.ndarray:
    """
    Generates a procedural grayscale image resembling a cratered lunar surface.
    """
    rng = np.random.default_rng(seed)
    
    # Base lunar terrain: fractal perlin-like noise using Gaussian blurs at multiple scales
    terrain = np.zeros((height, width), dtype=np.float32)
    for scale in [128, 64, 32, 16, 8]:
        noise = rng.standard_normal((height // scale + 4, width // scale + 4)).astype(np.float32)
        resized_noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        terrain += resized_noise * (scale / 128.0)
    
    # Normalize base terrain to 0.3 - 0.7 range
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-8)
    img = (terrain * 120 + 60).astype(np.float32)
    
    # Add craters of varying sizes with rim highlights and shadow gradients
    # Simulated sun illumination from upper-left (-1, -1)
    sun_dir = np.array([-0.707, -0.707], dtype=np.float32)
    
    for _ in range(num_craters):
        cx = rng.uniform(0.1 * width, 0.9 * width)
        cy = rng.uniform(0.1 * height, 0.9 * height)
        radius = rng.uniform(10, 90)
        depth = rng.uniform(40, 110)
        
        # Grid around crater
        y_min = max(0, int(cy - radius * 1.5))
        y_max = min(height, int(cy + radius * 1.5))
        x_min = max(0, int(cx - radius * 1.5))
        x_max = min(width, int(cx + radius * 1.5))
        
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        r_sq = radius ** 2
        
        # Interior bowl
        interior_mask = dist_sq <= r_sq
        bowl = depth * (1.0 - dist_sq / r_sq)
        
        # Directional crater shadow/illumination
        dx = (xx - cx) / (radius + 1e-5)
        dy = (yy - cy) / (radius + 1e-5)
        dot_sun = dx * sun_dir[0] + dy * sun_dir[1]
        
        img[y_min:y_max, x_min:x_max][interior_mask] -= (bowl * (1.0 - 0.5 * dot_sun))[interior_mask]
        
        # Raised crater rim
        rim_mask = (dist_sq > r_sq) & (dist_sq <= (radius * 1.35) ** 2)
        rim_intensity = (depth * 0.45) * np.exp(-((np.sqrt(dist_sq) - radius) ** 2) / (2 * (radius * 0.15) ** 2))
        img[y_min:y_max, x_min:x_max][rim_mask] += (rim_intensity * (0.8 + 0.6 * dot_sun))[rim_mask]

    # Add realistic sensor shot noise & micro-roughness
    shot_noise = rng.normal(0, 3.0, (height, width))
    img = np.clip(img + shot_noise, 0, 255).astype(np.uint8)
    return img


def generate_synthetic_lunar_pair(
    base_image: Optional[np.ndarray] = None,
    rotation_deg: float = 18.5,
    scale: float = 1.15,
    translation: Tuple[float, float] = (35.0, -25.0),
    perspective_shear: Tuple[float, float] = (0.00015, -0.00012),
    output_size: Tuple[int, int] = (800, 800),
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Creates reference and source images with a precisely known ground truth Homography (H_gt).
    
    H_gt transforms reference points to source points: p_src = H_gt * p_ref.
    """
    if base_image is None:
        base_image = generate_procedural_lunar_surface(
            width=output_size[0], height=output_size[1], seed=seed
        )
    
    h, w = base_image.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    # Build 3x3 Ground Truth Homography:
    # 1. Translate center to origin
    T_to_orig = np.array([
        [1.0, 0.0, -center[0]],
        [0.0, 1.0, -center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # 2. Similarity (Rotation & Scale)
    rad = np.deg2rad(rotation_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R_S = np.array([
        [scale * cos_a, -scale * sin_a, 0.0],
        [scale * sin_a,  scale * cos_a, 0.0],
        [0.0,            0.0,           1.0]
    ], dtype=np.float64)
    
    # 3. Perspective distortion + translation back to center + offset
    T_back = np.array([
        [1.0, 0.0, center[0] + translation[0]],
        [0.0, 1.0, center[1] + translation[1]],
        [perspective_shear[0], perspective_shear[1], 1.0]
    ], dtype=np.float64)
    
    # Composite ground truth homography: H_gt: ref -> src
    H_gt = T_back @ R_S @ T_to_orig
    H_gt = H_gt / H_gt[2, 2]
    
    # Warp base_image (ref) to create source image
    # cv2.warpPerspective takes H transforming dst coords to src coords, or src to dst directly:
    # cv2.warpPerspective(src_img, M, (w, h)) maps src_img to dst using M.
    image_ref = base_image.copy()
    image_src = cv2.warpPerspective(
        image_ref,
        H_gt,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    metadata = {
        "rotation_deg": rotation_deg,
        "scale": scale,
        "translation": translation,
        "perspective_shear": perspective_shear,
        "H_gt": H_gt.tolist(),
        "shape_ref": image_ref.shape,
        "shape_src": image_src.shape,
    }
    
    return image_ref, image_src, H_gt, metadata


def load_and_preprocess_image(path: str) -> np.ndarray:
    """Loads an image from path and converts to single-channel uint8 grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img
