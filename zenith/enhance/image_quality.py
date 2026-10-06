"""
ZENITH Image Enhancement & Output Quality Post-Processing Module.
Scientifically conservative, deterministic image enhancement for registered outputs.
Includes edge-preserving denoising, CLAHE contrast enhancement, multi-scale unsharp masking,
checkerboard/false-color alignment overlays, and non-registration image quality metrics.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def calculate_quality_statistics(
    image: np.ndarray,
    image_enhanced: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes objective non-registration image quality metrics:
    - Sharpness via Variance of Laplacian (higher indicates crisper high-frequency textures)
    - RMS Contrast (standard deviation of normalized pixel intensities)
    - Dynamic Range & Shannon Entropy
    
    IMPORTANT: These metrics reflect visual readability and NEVER substitute for registration RMSE.
    """
    def _compute_single(img: np.ndarray) -> Dict[str, float]:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Avoid zero/background region skew if masked
        valid_mask = gray > 0
        if np.sum(valid_mask) < 50:
            return {
                "laplacian_sharpness": 0.0,
                "rms_contrast": 0.0,
                "michelson_contrast": 0.0,
                "entropy_bits": 0.0,
                "dynamic_range": 0.0
            }
        
        valid_pixels = gray[valid_mask].astype(np.float32)
        
        # 1. Sharpness: Variance of Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_var = float(np.var(lap[valid_mask]))
        
        # 2. RMS Contrast
        norm_pixels = valid_pixels / 255.0
        rms_cont = float(np.std(norm_pixels))
        
        # 3. Michelson Contrast
        p_min = float(np.percentile(valid_pixels, 1))
        p_max = float(np.percentile(valid_pixels, 99))
        mich_cont = float((p_max - p_min) / max(p_max + p_min, 1e-5))
        
        # 4. Shannon Entropy
        hist, _ = np.histogram(valid_pixels, bins=256, range=(0, 256), density=True)
        hist_nonzero = hist[hist > 0]
        entropy_val = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))
        
        return {
            "laplacian_sharpness": round(lap_var, 2),
            "rms_contrast": round(rms_cont, 4),
            "michelson_contrast": round(mich_cont, 4),
            "entropy_bits": round(entropy_val, 3),
            "dynamic_range": round(p_max - p_min, 1)
        }

    stats_raw = _compute_single(image)
    if image_enhanced is not None:
        stats_enh = _compute_single(image_enhanced)
        return {
            "raw": stats_raw,
            "enhanced": stats_enh,
            "sharpness_delta_pct": round(
                ((stats_enh["laplacian_sharpness"] - stats_raw["laplacian_sharpness"]) / max(stats_raw["laplacian_sharpness"], 1e-3)) * 100.0, 1
            ),
            "contrast_delta_pct": round(
                ((stats_enh["rms_contrast"] - stats_raw["rms_contrast"]) / max(stats_raw["rms_contrast"], 1e-4)) * 100.0, 1
            )
        }
    return {"raw": stats_raw}


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to reveal shadowed craters and subtle ejecta morphology without blowout.
    """
    if clip_limit <= 0.0:
        return image.copy()
        
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    elif image.ndim == 3:
        # Convert to LAB, apply CLAHE to L channel only
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image.copy()


def edge_preserving_denoise(
    image: np.ndarray,
    d: int = 5,
    sigma_color: float = 20.0,
    sigma_space: float = 20.0
) -> np.ndarray:
    """
    Applies bilateral filtering to suppress sensor noise while preserving sharp crater boundaries.
    """
    if sigma_color <= 0.0 or sigma_space <= 0.0:
        return image.copy()
    return cv2.bilateralFilter(image, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))


def sharpen_multiscale(
    image: np.ndarray,
    strength: float = 0.35,
    sigma_fine: float = 1.0,
    sigma_medium: float = 2.5
) -> np.ndarray:
    """
    Multi-scale unsharp masking that accentuates fine topological relief.
    """
    if strength <= 0.0:
        return image.copy()

    img_f = image.astype(np.float32)
    blur_fine = cv2.GaussianBlur(img_f, (0, 0), sigma_fine)
    blur_med = cv2.GaussianBlur(img_f, (0, 0), sigma_medium)
    
    detail_fine = img_f - blur_fine
    detail_med = blur_fine - blur_med
    
    # Combined controlled enhancement
    sharpened = img_f + (strength * 0.7 * detail_fine) + (strength * 0.3 * detail_med)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_registered_image(
    image: np.ndarray,
    method: str = "conservative",  # "conservative", "contrast_only", "denoise_only", "none"
    contrast_strength: float = 1.5,
    denoise_strength: float = 1.0,
    sharpen_strength: float = 0.30
) -> np.ndarray:
    """
    Complete post-processing pipeline for registered warped imagery.
    
    Methods:
    - 'conservative': Bilateral Denoise -> CLAHE Contrast -> Mild Multi-scale Sharpening.
    - 'contrast_only': CLAHE Contrast Enhancement only.
    - 'denoise_only': Edge-Preserving Bilateral Denoise only.
    - 'none': Returns raw copy.
    """
    if method == "none" or image is None or image.size == 0:
        return image.copy() if image is not None else np.empty((0, 0), dtype=np.uint8)

    # Maintain background zero mask so warped empty areas stay clean black
    mask_valid = (image > 0)

    if method == "denoise_only":
        res = edge_preserving_denoise(image, sigma_color=20.0 * denoise_strength, sigma_space=20.0 * denoise_strength)
    elif method == "contrast_only":
        res = enhance_contrast(image, clip_limit=contrast_strength * 1.5)
    else:  # "conservative" (Default recommended)
        denoised = edge_preserving_denoise(image, sigma_color=18.0 * denoise_strength, sigma_space=18.0 * denoise_strength)
        contrasted = enhance_contrast(denoised, clip_limit=contrast_strength * 1.4)
        res = sharpen_multiscale(contrasted, strength=sharpen_strength)

    # Re-apply strict valid border mask to avoid border interpolation artifacts
    res[~mask_valid] = 0
    return res


def create_false_color_overlay(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    alpha: float = 0.5,
    enhance: bool = False
) -> np.ndarray:
    """
    Creates a scientifically interpretable false-color alignment overlay:
    - Reference Frame -> Green channel
    - Warped Source Frame -> Magenta (Red + Blue channels)
    - Result: High-accuracy registered features converge toward neutral monochrome/white.
    - Misalignments or geometric warps produce vivid green or magenta color fringes.
    """
    ref_gray = img_ref if img_ref.ndim == 2 else cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    warp_gray = img_warped if img_warped.ndim == 2 else cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    if enhance:
        ref_proc = enhance_contrast(ref_gray, clip_limit=1.5)
        warp_proc = enhance_contrast(warp_gray, clip_limit=1.5)
    else:
        ref_proc = ref_gray
        warp_proc = warp_gray

    h, w = ref_proc.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Scale blend with alpha
    ref_f = ref_proc.astype(np.float32)
    warp_f = warp_proc.astype(np.float32)

    # Magenta (B + R) for Warped Source, Green (G) for Reference
    overlay[:, :, 0] = np.clip(warp_f * (1.0 - alpha * 0.2), 0, 255).astype(np.uint8)  # Blue
    overlay[:, :, 1] = np.clip(ref_f * (1.0 - (1.0 - alpha) * 0.2), 0, 255).astype(np.uint8)   # Green
    overlay[:, :, 2] = np.clip(warp_f * (1.0 - alpha * 0.2), 0, 255).astype(np.uint8)  # Red

    return overlay


def create_blended_overlay(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    blend_mode: str = "checkerboard",  # "checkerboard", "alpha", "difference"
    alpha: float = 0.5,
    checker_size: int = 64
) -> np.ndarray:
    """
    Creates specialized alignment inspection visualizations:
    - 'checkerboard': Alternating squares of reference and warped source to verify continuous crater rims.
    - 'alpha': Weighted linear blend.
    - 'difference': Normalized absolute radiometric difference.
    """
    ref_gray = img_ref if img_ref.ndim == 2 else cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    warp_gray = img_warped if img_warped.ndim == 2 else cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape[:2]

    if blend_mode == "alpha":
        blend = cv2.addWeighted(ref_gray, alpha, warp_gray, 1.0 - alpha, 0.0)
        return cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)

    elif blend_mode == "difference":
        # Absolute difference highlighted with colormap
        diff = cv2.absdiff(ref_gray, warp_gray)
        # Mask out unwarped background
        diff[warp_gray == 0] = 0
        diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        diff_color[warp_gray == 0] = [0, 0, 0]
        return diff_color

    else:  # "checkerboard"
        canvas = np.zeros((h, w), dtype=np.uint8)
        y_indices, x_indices = np.indices((h, w))
        checker_mask = ((y_indices // checker_size) + (x_indices // checker_size)) % 2 == 0
        canvas[checker_mask] = ref_gray[checker_mask]
        canvas[~checker_mask] = warp_gray[~checker_mask]
        
        # Highlight checkerboard boundaries slightly
        out_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for y in range(0, h, checker_size):
            cv2.line(out_bgr, (0, y), (w, y), (40, 40, 40), 1)
        for x in range(0, w, checker_size):
            cv2.line(out_bgr, (x, 0), (x, h), (40, 40, 40), 1)
        return out_bgr
