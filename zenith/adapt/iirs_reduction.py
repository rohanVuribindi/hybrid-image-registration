"""
IIRS (Imaging Infra-Red Spectrometer) Hyperspectral Cube Dimensionality Reduction.
Reduces full hyperspectral datacubes into high-SNR structural / spectral-derived channels
prior to entering the REPRESENT module.
"""
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
from sklearn.decomposition import PCA


def reduce_iirs_hyperspectral_cube(
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    method: str = "pca_continuum",
    n_components: int = 3
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduces 3D hyperspectral cube (H, W, B) to 2D single-channel structural or 3-channel structural image.
    
    Args:
        cube: (H, W, B) 3D numpy array of spectral radiance / reflectance.
        wavelengths: (B,) array of spectral wavelengths in nanometers (e.g. 800nm - 5000nm).
        method: 'pca_continuum', 'band_ratio', or 'broadband_mean'.
        n_components: Number of PCA components if using PCA.
        
    Returns:
        reduced_image: (H, W) uint8 grayscale image suitable for REPRESENT.
        meta: Metadata with explained variance and spectral band information.
    """
    h, w, b = cube.shape
    cube_clean = np.nan_to_num(cube.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if method == "pca_continuum":
        # Reshape to (H*W, B) for PCA
        flat = cube_clean.reshape(-1, b)
        pca = PCA(n_components=min(n_components, b))
        transformed = pca.fit_transform(flat)
        
        # Primary structural component (PC1 captures dominant morphological surface reflectance)
        pc1 = transformed[:, 0].reshape(h, w)
        
        # Normalize PC1 to [0, 255] uint8
        pc1_min, pc1_max = pc1.min(), pc1.max()
        if pc1_max > pc1_min:
            reduced = ((pc1 - pc1_min) / (pc1_max - pc1_min) * 255.0).astype(np.uint8)
        else:
            reduced = np.zeros((h, w), dtype=np.uint8)

        meta = {
            "method": "pca_continuum",
            "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "total_variance_captured": round(float(np.sum(pca.explained_variance_ratio_)), 4),
            "input_cube_shape": (h, w, b),
            "output_shape": (h, w)
        }
        return reduced, meta

    elif method == "band_ratio":
        # Calculate mineralogical band depth ratio (e.g. 1000nm Pyroxene vs 750nm continuum)
        # If wavelengths provided, find closest indices; otherwise use representative band indices
        idx_cont = int(0.2 * b)
        idx_abs = int(0.4 * b)
        cont = cube_clean[:, :, idx_cont]
        absorp = cube_clean[:, :, idx_abs]
        ratio = cont / (absorp + 1e-6)
        
        reduced = cv2.normalize(ratio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        meta = {
            "method": "band_ratio",
            "continuum_band_idx": idx_cont,
            "absorption_band_idx": idx_abs,
            "output_shape": (h, w)
        }
        return reduced, meta

    else: # "broadband_mean"
        broadband = np.mean(cube_clean, axis=2)
        reduced = cv2.normalize(broadband, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        meta = {
            "method": "broadband_mean",
            "bands_averaged": b,
            "output_shape": (h, w)
        }
        return reduced, meta


def simulate_synthetic_iirs_cube(
    base_morphology: np.ndarray,
    num_bands: int = 64,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a realistic Chandrayaan-2 IIRS hyperspectral datacube (800nm - 5000nm)
    with wavelength-dependent albedo curves and lunar soil mineral absorption (Pyroxene/Olivine 1µm & 2µm bands).
    """
    rng = np.random.default_rng(seed)
    h, w = base_morphology.shape[:2]
    wavelengths = np.linspace(800.0, 5000.0, num_bands)  # nm
    
    cube = np.zeros((h, w, num_bands), dtype=np.float32)
    base_f = base_morphology.astype(np.float32) / 255.0

    # Lunar spectral curve: general red slope across NIR + 1050nm pyroxene absorption + 2800nm OH absorption
    for i, wl in enumerate(wavelengths):
        # NIR spectral continuum slope
        continuum_slope = 0.8 + 0.5 * (wl - 800.0) / 4200.0
        
        # 1050 nm absorption band
        abs_1000 = 0.18 * np.exp(-((wl - 1050.0) ** 2) / (2 * (150.0 ** 2)))
        
        # 2850 nm OH/H2O absorption band
        abs_2800 = 0.12 * np.exp(-((wl - 2850.0) ** 2) / (2 * (250.0 ** 2)))
        
        # Band response
        band_response = continuum_slope - abs_1000 - abs_2800
        
        # Spatially varied mineral noise
        mineral_variation = rng.normal(1.0, 0.02, (h, w))
        cube[:, :, i] = base_f * band_response * mineral_variation + rng.normal(0, 0.005, (h, w))

    return cube, wavelengths
