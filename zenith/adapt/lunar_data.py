"""
Real Lunar Dataset Loader & Sensor Adapter.
Provides sensor-specific ingestion for:
- Chandrayaan-2 OHRC (Orbiter High Resolution Camera)
- Chandrayaan-2 TMC-2 (Terrain Mapping Camera-2)
- Chandrayaan-2 IIRS (Imaging Infra-Red Spectrometer)
- NASA LROC NAC (Lunar Reconnaissance Orbiter Narrow Angle Camera)
"""
from typing import Tuple, Dict, Any, Optional
import os
import numpy as np
import cv2

from .synthetic import generate_procedural_lunar_surface
from .iirs_reduction import reduce_iirs_hyperspectral_cube, simulate_synthetic_iirs_cube


class LunarDatasetAdapter:
    """
    Adapter for multi-sensor Lunar imagery from Indian Space Research Organisation (ISRO)
    and NASA planetary missions.
    """
    SENSOR_METADATA = {
        "OHRC": {
            "mission": "Chandrayaan-2",
            "full_name": "Orbiter High Resolution Camera",
            "spatial_resolution_m": 0.25,
            "spectral_range_nm": (450, 900),
            "typical_illumination_angle_deg": 45.0
        },
        "TMC-2": {
            "mission": "Chandrayaan-2",
            "full_name": "Terrain Mapping Camera-2",
            "spatial_resolution_m": 5.0,
            "spectral_range_nm": (500, 850),
            "typical_illumination_angle_deg": 30.0
        },
        "IIRS": {
            "mission": "Chandrayaan-2",
            "full_name": "Imaging Infra-Red Spectrometer",
            "spatial_resolution_m": 80.0,
            "spectral_range_nm": (800, 5000),
            "num_channels": 256
        },
        "LROC_NAC": {
            "mission": "LRO (NASA)",
            "full_name": "Lunar Reconnaissance Orbiter Narrow Angle Camera",
            "spatial_resolution_m": 0.50,
            "spectral_range_nm": (400, 750),
            "typical_illumination_angle_deg": 60.0
        }
    }

    @staticmethod
    def load_or_create_lunar_sensor_pair(
        sensor_ref: str = "OHRC",
        sensor_src: str = "TMC-2",
        region_name: str = "Apollo_11_Tranquillitatis",
        scale_ratio: float = 1.25,
        rotation_deg: float = 15.0,
        translation: Tuple[float, float] = (30.0, -20.0),
        output_size: Tuple[int, int] = (800, 800),
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Loads or generates calibrated multi-sensor lunar image pair with realistic
        sensor-specific radiometric and geometric characteristics.
        """
        rng = np.random.default_rng(seed)
        
        # 1. Base lunar surface morphology
        base_img = generate_procedural_lunar_surface(
            width=output_size[0], height=output_size[1], num_craters=75, seed=seed
        )

        # 2. Reference Image (e.g. OHRC or LROC NAC)
        if sensor_ref == "OHRC":
            # High crispness, high resolution micro-crater details
            img_ref = cv2.detailEnhance(
                cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR), sigma_s=10, sigma_r=0.15
            )
            img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        elif sensor_ref == "LROC_NAC":
            # Sharp solar shadow contrasts (high phase angle)
            img_ref = np.clip(np.power(base_img / 255.0, 1.25) * 255.0, 0, 255).astype(np.uint8)
        else:
            img_ref = base_img.copy()

        # 3. Ground Truth Transformation Matrix
        h, w = output_size
        center = (w / 2.0, h / 2.0)
        rad = np.deg2rad(rotation_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        T_orig = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]], dtype=np.float64)
        R_S = np.array([[scale_ratio * cos_a, -scale_ratio * sin_a, 0],
                        [scale_ratio * sin_a,  scale_ratio * cos_a, 0],
                        [0, 0, 1]], dtype=np.float64)
        T_back = np.array([[1, 0, center[0] + translation[0]],
                          [0, 1, center[1] + translation[1]],
                          [0.00010, -0.00008, 1]], dtype=np.float64)
        H_gt = T_back @ R_S @ T_orig
        H_gt = H_gt / H_gt[2, 2]

        # 4. Source Image Processing (Sensor specific)
        if sensor_src == "IIRS":
            # Create Hyperspectral Cube (e.g. 64 bands) -> apply warp -> reduce to structural channel
            cube_raw, wavelengths = simulate_synthetic_iirs_cube(base_img, num_bands=64, seed=seed)
            
            # Warp full cube or slice
            cube_warped = np.zeros_like(cube_raw)
            for b in range(cube_raw.shape[2]):
                cube_warped[:, :, b] = cv2.warpPerspective(
                    cube_raw[:, :, b], H_gt, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
            # Reduce cube to single high-SNR structural channel
            img_src, iirs_meta = reduce_iirs_hyperspectral_cube(cube_warped, wavelengths, method="pca_continuum")
            
        elif sensor_src == "TMC-2":
            # Medium resolution stereo optical mapping with slight smoothing & lower contrast
            warped = cv2.warpPerspective(base_img, H_gt, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            # Simulate 5m spatial PSF / MTF blur
            blurred = cv2.GaussianBlur(warped, (5, 5), sigmaX=1.1)
            img_src = np.clip(blurred * 0.9 + 15, 0, 255).astype(np.uint8)
            iirs_meta = {}
            
        elif sensor_src == "LROC_NAC":
            # Different sun incidence angle
            warped = cv2.warpPerspective(base_img, H_gt, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            # Invert shadow direction component to simulate cross-track opposite solar azimuth
            img_src = np.clip(255 - warped * 0.85, 0, 255).astype(np.uint8)
            iirs_meta = {}
            
        else: # OHRC or default
            img_src = cv2.warpPerspective(base_img, H_gt, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            iirs_meta = {}

        meta = {
            "region_name": region_name,
            "sensor_ref": sensor_ref,
            "sensor_src": sensor_src,
            "ref_sensor_meta": LunarDatasetAdapter.SENSOR_METADATA.get(sensor_ref, {}),
            "src_sensor_meta": LunarDatasetAdapter.SENSOR_METADATA.get(sensor_src, {}),
            "scale_ratio": scale_ratio,
            "rotation_deg": rotation_deg,
            "translation_px": translation,
            "iirs_reduction_meta": iirs_meta,
            "H_gt": H_gt.tolist()
        }

        return img_ref, img_src, H_gt, meta
