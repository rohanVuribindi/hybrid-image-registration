"""
ADAPT module for Zenith pipeline.
Provides synthetic benchmark pair generation, real lunar sensor adapters (OHRC, TMC-2, IIRS, LROC NAC),
and hyperspectral dimensionality reduction.
"""
from .synthetic import generate_synthetic_lunar_pair, load_and_preprocess_image, generate_procedural_lunar_surface
from .iirs_reduction import reduce_iirs_hyperspectral_cube, simulate_synthetic_iirs_cube
from .lunar_data import LunarDatasetAdapter

__all__ = [
    "generate_synthetic_lunar_pair",
    "load_and_preprocess_image",
    "generate_procedural_lunar_surface",
    "reduce_iirs_hyperspectral_cube",
    "simulate_synthetic_iirs_cube",
    "LunarDatasetAdapter"
]
