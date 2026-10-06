# Hybrid Image Registration

> **Physics-Informed Multi-Stage Geometric Coregistration for Lunar & Planetary Cross-Modal Remote Sensing**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)

---

## 1. Project Overview

**Hybrid Image Registration (ZENITH Engine)** is a high-precision, physics-informed computer vision pipeline engineered to solve the challenging problem of **cross-modal, multi-illumination, and cross-resolution planetary image registration**.

Designed for orbital observation platforms—such as **ISRO Chandrayaan-2/3 (TMC-2, OHRC, IIRS)** and **NASA LRO (LROC NAC/WAC)**—the system aligns optical, terrain-mapping, and hyperspectral datasets captured under extreme differences in solar elevation, shadow orientations, optical zoom, and sensor geometries.

Unlike naive deep learning approaches that hallucinate textures or fail under unfamiliar extraterrestrial topography, this system combines **frequency-domain physical invariants (Log-Gabor Phase Congruency)** with **high-throughput classical descriptors (RootSIFT)**, backed by a **deep transformer fallback (LoFTR)**, and bounded by **marginalized robust geometric estimation (USAC_MAGSAC++)**.

---

## 2. Problem Statement & Key Challenges

Planetary optical and spectral imagery presents extreme radiometric and geometric challenges that cause standard feature detectors (SIFT, ORB, Harris) and standard RANSAC to fail:

1. **Terminator Shadow Inversion**: The Moon's lack of atmosphere produces zero-diffusion, high-contrast shadows. When solar azimuth flips by $90^\circ$ to $180^\circ$, crater rims cast shadows in opposite directions, causing gradient magnitudes and directions to invert.
2. **Cross-Resolution & Multi-Scale Gaps**: High-resolution narrow-angle cameras (e.g., OHRC at $0.25\text{ m/px}$) must be registered against wide-angle context cameras (e.g., TMC-2 at $5.0\text{ m/px}$), representing a $20\times$ scale divergence.
3. **Hyperspectral Radiometric Mismatch**: Infrared imaging spectrometers (e.g., IIRS 256 contiguous spectral bands from $0.8\text{ to }5.0\ \mu\text{m}$) capture chemical absorption properties rather than purely visual albedo.
4. **Feature Clustering on Craters**: Standard keypoint detectors oversample high-contrast rims of a single prominent crater, causing degenerate, ill-conditioned geometric homographies that distort the rest of the terrain.

---

## 3. Key Objectives

- **Sub-Pixel Geodetic Accuracy**: Achieve $< 0.5\text{ pixel}$ Root Mean Square Error (RMSE) on multi-modal lunar surfaces.
- **Illumination Invariance**: Maintain stable correspondence matching across extreme sun elevation changes ($> 45^\circ$) and solar azimuth reversals.
- **Cascaded Compute Efficiency**: Resolve $> 80\%$ of nominal image pairs using ultra-fast classical methods ($< 350\text{ ms}$ on CPU) while automatically routing pathological pairs to deep neural transformers.
- **Strict Scientific Non-Generative Constraint**: Zero artificial texture synthesis, diffusion models, or GANs. Every output pixel is an authentic resampled physical observation.

---

## 4. System Architecture & Pipeline

The core registration engine executes a 7-stage cascaded pipeline:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               HYBRID REGISTRATION PIPELINE                              │
└────────────────────────────────────────────────────────────────────────────────────────┘

  [ STAGE 1: ADAPT ]
  Ingestion (GeoTIFF/PDS4/NumPy) ➔ 1-99% Quantile Normalization ➔ Hyperspectral PCA
         │
         ▼
  [ STAGE 2: REPRESENT ]
  Kovesi 2D Log-Gabor Phase Congruency (3 scales, 6 orientations) ➔ Rayleigh Denoising
         │
         ▼
  [ STAGE 3: MATCH (ENSEMBLE ROUTER) ]
  Primary: RootSIFT (Hellinger Kernel) + FLANN Matcher (Lowe's Ratio τ = 0.75)
  Fallback: LoFTR Deep Transformer (Linear Cross-Attention, activated when inliers < 15)
  Pruning: Spatial KD-Tree Correspondence Deduplication (δ = 2.5 px)
         │
         ▼
  [ STAGE 4: ESTIMATE & REJECT ]
  USAC_MAGSAC++ (Marginalized Sample Consensus, Epanechnikov Kernel, σ = 3.0 px)
  Degenerate Quad Pruning (Collinear, Inverted, & Non-Convex Rejection)
         │
         ▼
  [ STAGE 5: REFINE ]
  Förstner Gradient Structure Tensor Sub-Pixel Optimization (S Δp = b)
  4×4 Spatial Grid Inlier Balancing (Cap 15 pts/cell, require ≥ 4 active quadrants)
  Levenberg-Marquardt Non-Linear Homography Optimization
         │
         ▼
  [ STAGE 6: VERIFY & DECIDE ]
  Multi-Factor Confidence Scoring: C = 0.35*R_inlier + 0.25*S_cov + 0.25*E_score + 0.15*N
  Decision Output: [ REGISTERED (C ≥ 0.70) | LOW_CONFIDENCE | FAILED ]
         │
         ▼
  [ STAGE 7: WARP & EXPORT ]
  High-Fidelity Perspective Warping (cv2.INTER_LANCZOS4 8×8 Sinc Interpolation)
  Artifact Generation: Registered Warped Image, Alignment Overlays, Feature Maps, CSV, JSON
```

---

## 5. Algorithmic Deep Dive

### 5.1 Kovesi 2D Log-Gabor Phase Congruency
Phase Congruency detects structural features (crater boundaries, ridge lines) at points of maximum phase harmony across frequency scales, independent of local brightness or contrast:

$$PC(x, y) = \frac{\sum_o \sum_n W_o(x, y) \lfloor A_{no}(x, y) \Delta \Phi_{no}(x, y) - T_o \rfloor_+}{\sum_o \sum_n A_{no}(x, y) + \epsilon}$$

where Log-Gabor transfer filters in the frequency domain are formulated as:
$$G(\omega) = \exp\left( -\frac{\left(\ln(\omega / \omega_0)\right)^2}{2 \left(\ln(k / \omega_0)\right)^2} \right)$$

### 5.2 RootSIFT with Hellinger Kernel
Standard SIFT computes Euclidean distance on $L_2$-normalized gradient histograms, which overweights large gradient bin noise. RootSIFT maps descriptors to the probability simplex via $L_1$ normalization and element-wise square rooting:

$$\mathbf{x}_{\text{root}} = \sqrt{\frac{\mathbf{x}}{\|\mathbf{x}\|_1}} \implies d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}_{\text{root}} - \mathbf{y}_{\text{root}}\|_2 = \sqrt{2 - 2 K_{\text{Hellinger}}(\mathbf{x}, \mathbf{y})}$$

### 5.3 LoFTR Deep Transformer Fallback
When severe $180^\circ$ shadow inversions cause keypoint detectors to find fewer than 15 inliers, the pipeline triggers **LoFTR (Local Feature TRansformer)**:
- **ResNet-FPN Backbone**: Extracts dense coarse ($1/8$) and fine ($1/2$) feature maps.
- **Linear Transformer**: Applies interleaved self-attention and cross-attention blocks ($O(N)$ linear complexity).
- **Optimal Transport / Dual-Softmax**: Evaluates dense mutual match probabilities $P(i, j) = \text{softmax}(\mathbf{S}(i, :))_j \cdot \text{softmax}(\mathbf{S}(:, j))_i$.

### 5.4 USAC_MAGSAC++ Robust Geometric Estimation
Unlike traditional RANSAC which requires a fixed manual threshold $\sigma$, MAGSAC++ marginalizes over the continuous noise distribution using an Epanechnikov kernel:

$$L(\mathbf{H}) = \sum_{i=1}^N \int_0^{\sigma_{\max}} \rho\left(\frac{r_i(\mathbf{H})^2}{2\sigma^2}\right) P(\sigma) d\sigma$$

### 5.5 Förstner Structure Tensor Sub-Pixel Refinement
For every inlier keypoint $\mathbf{p}_0$, the true sub-pixel corner location minimizes local gradient perpendicular errors:

$$\mathbf{S} \Delta \mathbf{p} = \mathbf{b} \implies \Delta \mathbf{p} = \mathbf{S}^{-1} \mathbf{b}, \qquad \mathbf{S} = \sum_{\Omega} w(\mathbf{p}) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

---

## 6. Project Structure

```
hybrid-image-registration/
├── LICENSE                               # MIT License
├── README.md                             # Comprehensive technical documentation
├── requirements.txt                      # Python dependencies
├── docs/                                 # Detailed engineering manuals & presentations
│   ├── SIH26-A0H-T363-SIH26166_Presentation.pdf
│   ├── SIH26-A0H-T363-SIH26166_Presentation.pptx
│   ├── ZENITH_ALGORITHMS_AND_ML_MODEL_REFERENCE.docx
│   ├── ZENITH_ALGORITHMS_AND_ML_MODEL_REFERENCE.md
│   ├── ZENITH_ALGORITHMS_AND_ML_MODEL_REFERENCE.pdf
│   ├── ZENITH_COMPLETE_TECHNICAL_DOCUMENTATION.docx
│   ├── ZENITH_COMPLETE_TECHNICAL_DOCUMENTATION.md
│   └── ZENITH_COMPLETE_TECHNICAL_DOCUMENTATION.pdf
└── zenith/                               # Core Python Package
    ├── __init__.py
    ├── app.py                            # Streamlit Mission Control Web Dashboard
    ├── pipeline.py                       # ZenithPipeline 7-stage orchestrator
    ├── run_mvp1.py ... run_mvp8.py       # Benchmark validation suite (MVP1 to MVP8)
    ├── adapt/                            # Data ingestion, normalization, PCA, synthetic pairs
    │   ├── iirs_reduction.py
    │   ├── lunar_data.py
    │   └── synthetic.py
    ├── enhance/                          # Non-generative post-processing filters
    │   └── image_quality.py
    ├── evaluate/                         # Decision engine, confidence scoring, RMSE metrics
    │   ├── confidence.py
    │   └── metrics.py
    ├── match/                            # Multi-matcher ensemble (RootSIFT, LoFTR, LightGlue, RoMa)
    │   ├── common.py
    │   ├── ensemble.py
    │   ├── hybrid.py
    │   ├── lightglue_matcher.py
    │   ├── loftr_matcher.py
    │   ├── roma_matcher.py
    │   ├── rootsift.py
    │   ├── router.py
    │   └── xoftr_matcher.py
    ├── output/                           # Lanczos-4 perspective warping & exporter
    │   └── warp.py
    ├── refine/                           # Förstner structure tensor & spatial grid balancing
    │   ├── spatial_balancer.py
    │   └── subpixel.py
    ├── represent/                        # Kovesi Log-Gabor Phase Congruency & scale-space
    │   ├── phase_congruency.py
    │   ├── pyramid.py
    │   └── structural.py
    ├── ui/                               # Dark-theme Space Mission UI components & styling
    │   ├── artifact_loader.py
    │   ├── components.py
    │   └── styles.py
    └── verify/                           # USAC_MAGSAC++ robust estimation & homography fitting
        └── estimator.py
```

---

## 7. Installation Requirements

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: NVIDIA GPU with CUDA 11.8+ for deep transformer acceleration (CPU fallback fully supported)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/rohanVuribindi/hybrid-image-registration.git
cd hybrid-image-registration

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. How to Run

### 1. Launch the Interactive Mission Control Dashboard
Run the Streamlit web application:

```bash
streamlit run zenith/app.py
```

Navigate to `http://localhost:8501` to access:
- **Dashboard**: Real-time telemetry, capability cards, and benchmark specifications.
- **Register Images**: Load Chandrayaan-2 / NASA LRO presets or upload custom raster pairs, adjust matcher strategies, and trigger live registration.
- **Results & Visualization**: View verified alignment status, interactive vertical before/after slider, spatially distributed feature correspondence vectors, and false-color overlays.

### 2. Run Benchmark Validation Milestones
Execute the automated scientific test scripts:

```bash
# MVP 1: Baseline SIFT vs Planetary Illumination Stress
python zenith/run_mvp1.py

# MVP 2: Kovesi Log-Gabor Phase Congruency Evaluation
python zenith/run_mvp2.py

# MVP 3: RootSIFT Hellinger Kernel vs Standard SIFT
python zenith/run_mvp3.py

# MVP 4: USAC_MAGSAC++ Robust Consensus Optimization
python zenith/run_mvp4.py

# MVP 5: Sub-Pixel Refinement & 4x4 Spatial Grid Balancing
python zenith/run_mvp5.py

# MVP 6: Multi-Sensor & Hyperspectral PCA Evaluation
python zenith/run_mvp6.py

# MVP 7: Deep Transformer Fallback Evaluation
python zenith/run_mvp7.py

# MVP 8: End-to-End Multi-Matcher Ensemble Benchmark
python zenith/run_mvp8.py
```

---

## 9. Python API Usage Example

```python
import cv2
from zenith.pipeline import ZenithPipeline

# Initialize the 7-stage registration engine
pipeline = ZenithPipeline(
    matcher_mode="ensemble",          # RootSIFT with automatic LoFTR fallback
    representation_mode="phase_congruency",
    reproj_threshold=3.0,             # USAC_MAGSAC++ pixel threshold
    output_dir="zenith_outputs/demo_run"
)

# Load reference and source lunar rasters
img_ref = cv2.imread("data/reference_lunar.png", cv2.IMREAD_GRAYSCALE)
img_src = cv2.imread("data/source_lunar.png", cv2.IMREAD_GRAYSCALE)

# Execute registration
results = pipeline.run_pair(
    img_ref=img_ref,
    img_src=img_src,
    experiment_id="lunar_coregistration_01"
)

# Inspect results
print(f"Status: {results['status']}")                   # 'REGISTERED' / 'LOW_CONFIDENCE' / 'FAILED'
print(f"Confidence: {results['confidence_score']:.2f}") # e.g. 0.94
print(f"Inliers: {results['inlier_count']}")            # e.g. 295
print(f"Reprojection RMSE: {results['mean_inlier_residual_px']:.3f} px")
print(f"Homography Matrix H:\n{results['H_estimated']}")
```

---

## 10. Quantitative Benchmark Metrics

| Metric | Nominal Synthetic | Chandrayaan-2 OHRC vs TMC-2 | Severe Solar Terminator ($> 45^\circ$) | Hyperspectral IIRS PCA |
| :--- | :---: | :---: | :---: | :---: |
| **Registration Status** | `REGISTERED` | `REGISTERED` | `REGISTERED` | `REGISTERED` |
| **Reprojection RMSE** | **$0.103\text{ px}$** | **$0.408\text{ px}$** | **$0.472\text{ px}$** | **$0.518\text{ px}$** |
| **Inlier Ratio** | $94.5\%$ | $78.2\%$ | $64.8\%$ | $61.2\%$ |
| **Spatial Grid Coverage** | $100\%$ ($16/16$ cells) | $87.5\%$ ($14/16$ cells) | $81.2\%$ ($13/16$ cells) | $75.0\%$ ($12/16$ cells) |
| **Composite Confidence $C$** | **$0.96$** | **$0.92$** | **$0.86$** | **$0.84$** |
| **Execution Latency (CPU)** | $0.27\text{ s}$ | $0.34\text{ s}$ | $1.42\text{ s}$ *(LoFTR Triggered)* | $0.41\text{ s}$ |

---

## 11. Technology Stack

- **Core Engine**: Python 3.10+, NumPy, SciPy
- **Computer Vision**: OpenCV (`cv2.USAC_MAGSAC`, `cv2.INTER_LANCZOS4`), Scikit-Image
- **Deep Transformers**: PyTorch, Kornia, LoFTR (`loftr_outdoor.ckpt`), LightGlue
- **Interactive UI**: Streamlit, Pandas, Matplotlib, Pillow
- **Documentation**: Markdown, KaTeX, Python-Docx, PyMuPDF, ReportLab

---

## 12. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
