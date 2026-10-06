"""
Phase Congruency structural representation using phasepack.
Provides contrast- and illumination-invariant edge and feature maps.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import warnings

# Filter harmless pyfftw warning if present
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import phasepack


def compute_phase_congruency(
    image: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wave_length: int = 3,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    normalize_uint8: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes contrast-invariant Phase Congruency edge map (M) and corner map (m).
    
    Args:
        image: Single-channel 2D uint8 or float32 image.
        nscale: Number of wavelet scales (default 4).
        norient: Number of filter orientations (default 6).
        min_wave_length: Wavelength of smallest scale filter.
        mult: Scaling factor between successive filters.
        sigma_on_f: Ratio of Gaussian standard deviation to center frequency.
        normalize_uint8: If True, scales outputs to [0, 255] uint8.
        
    Returns:
        M_norm: Maximum moment of phase congruency covariance (Edge strength map)
        m_norm: Minimum moment (Corner / junction feature map)
        meta: Processing metadata
    """
    if image.ndim == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image.copy()
        
    img_float = img_gray.astype(np.float64) / 255.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M, m, ori, ft, PC, EO, T = phasepack.phasecong(
            img_float,
            nscale=nscale,
            norient=norient,
            minWaveLength=min_wave_length,
            mult=mult,
            sigmaOnf=sigma_on_f
        )

    if normalize_uint8:
        # Scale M (edge map) to [0, 255] uint8
        M_valid = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        M_min, M_max = M_valid.min(), M_valid.max()
        if M_max > M_min:
            M_out = ((M_valid - M_min) / (M_max - M_min) * 255.0).astype(np.uint8)
        else:
            M_out = np.zeros_like(img_gray, dtype=np.uint8)
            
        m_valid = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m_min, m_max = m_valid.min(), m_valid.max()
        if m_max > m_min:
            m_out = ((m_valid - m_min) / (m_max - m_min) * 255.0).astype(np.uint8)
        else:
            m_out = np.zeros_like(img_gray, dtype=np.uint8)
    else:
        M_out = M.astype(np.float32)
        m_out = m.astype(np.float32)

    meta = {
        "nscale": nscale,
        "norient": norient,
        "minWaveLength": min_wave_length,
        "mult": mult,
        "threshold_T": float(T) if isinstance(T, (int, float, np.floating)) else 0.0
    }

    return M_out, m_out, meta
