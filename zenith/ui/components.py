"""
UI components and renderers for Zenith Streamlit Application.
Space Mission & Scientific Interface System.
"""
from typing import Dict, Any, Optional, List, Union, Tuple
import os
import base64
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from .styles import render_status_pill, render_metric_card


def image_to_base64(img: Union[str, np.ndarray, Image.Image]) -> str:
    """Converts a filepath, OpenCV array, or PIL image to a Base64 data URI."""
    if isinstance(img, str):
        if not os.path.exists(img):
            return ""
        with open(img, "rb") as f:
            data = f.read()
            return f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"
    elif isinstance(img, np.ndarray):
        if img.size == 0:
            return ""
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            rgb = img
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    elif isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    return ""


def render_mission_header():
    """Renders the top global mission control header."""
    st.markdown("""
    <div class="mission-header">
        <div class="mission-title-group">
            <div class="mission-brand">ZENITH <span>LUNAR & PLANETARY REGISTRATION</span></div>
            <div class="mission-sub">Cross-Modal Geometric Alignment for Planetary Imaging</div>
        </div>
        <div>
            <span class="mission-status-pill">● REGISTRATION ENGINE ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_workflow_steps(current_step: int = 1):
    """
    Renders the 4-step mission workflow indicator:
    STEP 1: LOAD IMAGES -> STEP 2: REGISTER -> STEP 3: VERIFY -> STEP 4: INSPECT RESULT
    """
    steps = [
        (1, "LOAD IMAGES"),
        (2, "REGISTER"),
        (3, "VERIFY"),
        (4, "INSPECT RESULT")
    ]
    
    html = ['<div class="workflow-container">']
    for idx, (s_num, s_label) in enumerate(steps):
        active_cls = "workflow-step-active" if s_num == current_step else ""
        html.append(f"""
        <div class="workflow-step {active_cls}">
            <div class="workflow-step-num">{s_num}</div>
            <span>STEP {s_num} — {s_label}</span>
        </div>
        """)
        if idx < len(steps) - 1:
            html.append('<div class="workflow-arrow">➔</div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_pipeline_flowchart_horizontal():
    """Renders a conceptual 3-stage alignment architecture."""
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 18px 24px; margin: 12px 0 20px 0;">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; text-align: center;">
            <div style="flex: 1; min-width: 140px; background: #1E293B; border: 1px solid rgba(56, 189, 248, 0.4); padding: 12px; border-radius: 8px;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-bottom: 2px;">Fixed Frame</div>
                <div style="color: #38BDF8; font-weight: 800; font-size: 0.95rem;">REFERENCE IMAGE</div>
            </div>
            <div style="color: #38BDF8; font-weight: 900; font-size: 1.2rem;">➔</div>
            <div style="flex: 1.2; min-width: 180px; background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 2px solid #0284C7; padding: 12px; border-radius: 8px; box-shadow: 0 0 16px rgba(56, 189, 248, 0.15);">
                <div style="font-size: 0.72rem; color: #38BDF8; text-transform: uppercase; font-weight: 700; margin-bottom: 2px;">Cross-Modal Consensus</div>
                <div style="color: #F8FAFC; font-weight: 800; font-size: 1.05rem; letter-spacing: 0.05em;">ZENITH ENGINE</div>
            </div>
            <div style="color: #38BDF8; font-weight: 900; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 140px; background: #1E293B; border: 1px solid rgba(16, 185, 129, 0.4); padding: 12px; border-radius: 8px;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-bottom: 2px;">Aligned Frame</div>
                <div style="color: #34D399; font-weight: 800; font-size: 0.95rem;">REGISTERED IMAGE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_capability_grid():
    """Renders 3 simple, non-technical mission capability cards."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="zenith-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🛰</div>
            <h4 style="color: #38BDF8; margin-bottom: 6px; font-weight: 700;">MULTI-VIEW</h4>
            <p style="font-size: 0.88rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                Align images captured from different orbital passes, viewpoints, camera angles, and optical zoom scales.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="zenith-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🌑</div>
            <h4 style="color: #38BDF8; margin-bottom: 6px; font-weight: 700;">CROSS-MODAL</h4>
            <p style="font-size: 0.88rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                Handle severe differences in appearance, low-sun crater shadow inversions, and multi-sensor spectral bands.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="zenith-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🎯</div>
            <h4 style="color: #38BDF8; margin-bottom: 6px; font-weight: 700;">VERIFIED ALIGNMENT</h4>
            <p style="font-size: 0.88rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                Validate geometric locking using genuine verified feature correspondences across the overlapping terrain.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_simple_registration_status(
    status: str,
    confidence_score: float,
    summary_reason: Optional[str] = None
):
    """
    Renders a prominent, clear top-level registration status card in plain English.
    No technical thresholds are exposed in the main card.
    """
    s_upper = (status or "").upper().strip()
    
    if "REGISTERED" in s_upper or "ACCEPT" in s_upper or "PASS" in s_upper:
        card_class = "result-status-card-success"
        title = "✓ REGISTRATION SUCCESSFUL"
        title_color = "#10B981"
        default_explanation = "ZENITH successfully aligned the source image to the reference frame with high structural fidelity."
    elif "LOW" in s_upper or "FLAG" in s_upper:
        card_class = "result-status-card-warning"
        title = "⚠ LOW CONFIDENCE"
        title_color = "#F59E0B"
        default_explanation = "ZENITH established a partial geometric alignment, but some terrain regions show lower feature coverage or higher uncertainty."
    else:
        card_class = "result-status-card-danger"
        title = "✕ REGISTRATION FAILED"
        title_color = "#EF4444"
        default_explanation = "ZENITH could not establish a reliable geometric alignment. Try images with greater overlapping terrain."

    explanation = summary_reason if (summary_reason and len(summary_reason) > 5) else default_explanation

    st.markdown(f"""
    <div class="result-status-card {card_class}">
        <div class="result-status-title" style="color: {title_color};">
            {title}
        </div>
        <p class="result-status-desc">
            {explanation}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_simple_metrics(
    status: str,
    coverage_score: float = 0.0,
    confidence_score: float = 0.0
):
    """
    Renders exactly 3 simple, non-intimidating cards for general viewers:
    1. ALIGNMENT: Verified / Marginal / Unverified
    2. COVERAGE: Good / Limited / Insufficient
    3. CONFIDENCE: X.XX / 1.00
    """
    s_upper = (status or "").upper().strip()
    
    # 1. Alignment status
    if "REGISTERED" in s_upper or "ACCEPT" in s_upper:
        align_text = "✓ VERIFIED"
        align_color = "#10B981"
        align_sub = "Geometric transform locked"
    elif "LOW" in s_upper or "FLAG" in s_upper:
        align_text = "⚠ MARGINAL"
        align_color = "#F59E0B"
        align_sub = "Partial spatial locking"
    else:
        align_text = "✕ UNVERIFIED"
        align_color = "#EF4444"
        align_sub = "Alignment unverified"

    # 2. Coverage status
    if coverage_score >= 0.35 or "REGISTERED" in s_upper:
        cov_text = "✓ GOOD"
        cov_color = "#10B981"
        cov_sub = "Balanced terrain distribution"
    elif coverage_score >= 0.15 or "LOW" in s_upper:
        cov_text = "⚠ LIMITED"
        cov_color = "#F59E0B"
        cov_sub = "Cluster-restricted spread"
    else:
        cov_text = "✕ INSUFFICIENT"
        cov_color = "#EF4444"
        cov_sub = "Low spatial distribution"

    # 3. Confidence score
    conf_val = f"{confidence_score:.2f}"
    conf_sub = "Multi-metric consensus"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="simple-card">
            <div class="simple-card-label">ALIGNMENT</div>
            <div class="simple-card-value" style="color: {align_color}; font-size: 1.45rem;">{align_text}</div>
            <div class="simple-card-sub">{align_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="simple-card">
            <div class="simple-card-label">COVERAGE</div>
            <div class="simple-card-value" style="color: {cov_color}; font-size: 1.45rem;">{cov_text}</div>
            <div class="simple-card-sub">{cov_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="simple-card">
            <div class="simple-card-label">CONFIDENCE</div>
            <div class="simple-card-value" style="color: #38BDF8; font-size: 1.45rem;">{conf_val} <span style="font-size: 0.8rem; color: #64748B;">/ 1.00</span></div>
            <div class="simple-card-sub">{conf_sub}</div>
        </div>
        """, unsafe_allow_html=True)


def render_before_after_view(img_source, img_registered):
    """
    Renders the Before / After visual explanation:
    LEFT: "BEFORE — SOURCE IMAGE"
    RIGHT: "AFTER — REGISTERED IMAGE"
    """
    st.markdown("""
    <div class="alignment-flow-banner">
        <div class="alignment-flow-pill">
            <span>BEFORE: SOURCE IMAGE</span>
            <span style="color: #38BDF8; font-weight: 900;">➔</span>
            <span style="color: #F8FAFC; background: rgba(56, 189, 248, 0.2); padding: 2px 10px; border-radius: 4px;">ZENITH REGISTRATION</span>
            <span style="color: #38BDF8; font-weight: 900;">➔</span>
            <span style="color: #34D399;">AFTER: REGISTERED IMAGE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📷 BEFORE — SOURCE IMAGE")
        if img_source is not None:
            st.image(img_source, caption="Original Unaligned / Transformed Input Image", use_container_width=True)
        else:
            st.warning("Source image unavailable.")

    with col2:
        st.markdown("##### 🎯 AFTER — REGISTERED IMAGE")
        if img_registered is not None:
            st.image(img_registered, caption="Authoritative RAW Registered Image (Warped into Reference Frame Coordinates)", use_container_width=True)
        else:
            st.warning("Registered warped image unavailable.")


def render_interactive_comparison_slider(
    img_ref: Union[str, np.ndarray],
    img_registered: Union[str, np.ndarray],
    slider_id: str = "zenith_slider",
    height: int = 500
):
    """
    Renders a responsive, interactive vertical before/after split comparison slider.
    Left side: Reference Image
    Right side: Registered Source Image
    Labeled: "DRAG TO COMPARE"
    """
    b64_ref = image_to_base64(img_ref)
    b64_reg = image_to_base64(img_registered)

    if not b64_ref or not b64_reg:
        st.warning("Comparison slider images unavailable.")
        return

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                background-color: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                overflow: hidden;
                user-select: none;
                -webkit-user-select: none;
            }}
            .slider-container {{
                position: relative;
                width: 100%;
                height: {height}px;
                border-radius: 10px;
                overflow: hidden;
                border: 2px solid rgba(56, 189, 248, 0.4);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
                cursor: ew-resize;
                background: #0B0F19;
            }}
            .img-base {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
                pointer-events: none;
            }}
            .overlay-wrapper {{
                position: absolute;
                top: 0;
                left: 0;
                width: 50%;
                height: 100%;
                overflow: hidden;
                border-right: 2px solid #38BDF8;
                pointer-events: none;
                background: #0B0F19;
            }}
            .img-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                object-fit: contain;
                display: block;
                pointer-events: none;
            }}
            .divider-handle {{
                position: absolute;
                top: 0;
                bottom: 0;
                left: 50%;
                width: 3px;
                background: #38BDF8;
                transform: translateX(-50%);
                pointer-events: none;
                box-shadow: 0 0 12px #38BDF8;
            }}
            .handle-knob {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 38px;
                height: 38px;
                border-radius: 50%;
                background: #0F172A;
                border: 2px solid #38BDF8;
                box-shadow: 0 0 16px rgba(56, 189, 248, 0.9);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #38BDF8;
                font-size: 13px;
                font-weight: 800;
                letter-spacing: 1px;
            }}
            .badge-left {{
                position: absolute;
                bottom: 12px;
                left: 12px;
                background: rgba(15, 23, 42, 0.88);
                border: 1px solid rgba(16, 185, 129, 0.6);
                color: #34D399;
                padding: 5px 12px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                z-index: 10;
                pointer-events: none;
            }}
            .badge-right {{
                position: absolute;
                bottom: 12px;
                right: 12px;
                background: rgba(15, 23, 42, 0.88);
                border: 1px solid rgba(56, 189, 248, 0.6);
                color: #38BDF8;
                padding: 5px 12px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                z-index: 10;
                pointer-events: none;
            }}
            .instruction-banner {{
                text-align: center;
                padding: 6px 0;
            }}
            .instruction-pill {{
                display: inline-block;
                background: rgba(56, 189, 248, 0.12);
                border: 1px solid rgba(56, 189, 248, 0.35);
                color: #38BDF8;
                font-size: 0.75rem;
                font-weight: 700;
                padding: 3px 12px;
                border-radius: 9999px;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
        </style>
    </head>
    <body>
        <div class="instruction-banner">
            <span class="instruction-pill">↔ DRAG TO COMPARE</span>
        </div>
        <div id="{slider_id}_box" class="slider-container">
            <!-- Base: Reference image on the right -->
            <img id="{slider_id}_base" class="img-base" src="{b64_ref}" alt="Reference Frame" />

            <!-- Overlay: Registered source on the left -->
            <div id="{slider_id}_overlay_wrap" class="overlay-wrapper">
                <img id="{slider_id}_overlay_img" class="img-overlay" src="{b64_reg}" alt="Registered Source" />
            </div>

            <!-- Divider Line & Center Knob -->
            <div id="{slider_id}_handle" class="divider-handle">
                <div class="handle-knob">◀▶</div>
            </div>

            <div class="badge-left">REGISTERED SOURCE</div>
            <div class="badge-right">REFERENCE FRAME</div>
        </div>

        <script>
            (function() {{
                const box = document.getElementById("{slider_id}_box");
                const wrap = document.getElementById("{slider_id}_overlay_wrap");
                const handle = document.getElementById("{slider_id}_handle");
                const imgOverlay = document.getElementById("{slider_id}_overlay_img");

                function syncWidth() {{
                    const w = box.clientWidth;
                    imgOverlay.style.width = w + "px";
                }}

                function setSplit(clientX) {{
                    const rect = box.getBoundingClientRect();
                    let ratio = (clientX - rect.left) / rect.width;
                    if (ratio < 0.01) ratio = 0.01;
                    if (ratio > 0.99) ratio = 0.99;
                    const pct = (ratio * 100).toFixed(2) + "%";
                    wrap.style.width = pct;
                    handle.style.left = pct;
                    syncWidth();
                }}

                let dragging = false;
                box.addEventListener("mousedown", (e) => {{ dragging = true; setSplit(e.clientX); }});
                window.addEventListener("mouseup", () => {{ dragging = false; }});
                window.addEventListener("mousemove", (e) => {{ if (dragging) setSplit(e.clientX); }});

                box.addEventListener("touchstart", (e) => {{ dragging = true; setSplit(e.touches[0].clientX); }});
                window.addEventListener("touchend", () => {{ dragging = false; }});
                window.addEventListener("touchmove", (e) => {{ if (dragging) setSplit(e.touches[0].clientX); }});

                window.addEventListener("resize", syncWidth);
                setTimeout(syncWidth, 50);
                setTimeout(syncWidth, 250);
            }})();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height + 45)


def create_verified_feature_alignment_image(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    inliers_data: Union[pd.DataFrame, np.ndarray, List, Dict],
    density_mode: str = "dense",
    max_pts: Optional[int] = None
) -> Tuple[np.ndarray, int, int]:
    """
    Creates a clean, non-technical visual showing genuine verified inlier feature alignments.
    Uses ONLY actual verified inliers from the registration pipeline.
    Supports density modes:
      - 'dense' / 'More Points': ~150-180 spatially distributed points across an 8x8 grid.
      - 'all' / 'All Verified Inliers': All genuine inliers (up to 300 max for performance).
      - 'key' / 'Key Features': Top 30 points with numbered target crosshairs.
    Returns:
      (canvas: np.ndarray, drawn_count: int, total_count: int)
    """
    h, w = img_ref.shape[:2]
    c_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if img_ref.ndim == 2 else img_ref.copy()
    c_warp = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2BGR) if img_warped.ndim == 2 else img_warped.copy()

    canvas = np.zeros((h + 50, w * 2, 3), dtype=np.uint8)
    canvas[50:50+h, 0:w] = c_ref
    canvas[50:50+h, w:2*w] = c_warp

    # Top Header Labels
    cv2.putText(canvas, "REFERENCE FRAME (Fixed Frame)", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, "REGISTERED SOURCE (Warped Alignment)", (w + 15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (52, 211, 153), 2, cv2.LINE_AA)

    # 1. Parse verified inlier coordinates and residuals
    pts_all = []
    resids_all = []

    if isinstance(inliers_data, pd.DataFrame):
        df_in = inliers_data[inliers_data["is_inlier"] == 1] if "is_inlier" in inliers_data.columns else inliers_data
        for _, row in df_in.iterrows():
            pts_all.append((float(row["ref_x"]), float(row["ref_y"])))
            resids_all.append(float(row.get("reproj_residual_px", 1.0)))
    elif isinstance(inliers_data, np.ndarray) and inliers_data.ndim == 2 and inliers_data.shape[1] >= 2:
        for pt in inliers_data:
            pts_all.append((float(pt[0]), float(pt[1])))
            resids_all.append(1.0)
    elif isinstance(inliers_data, list):
        for item in inliers_data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pts_all.append((float(item[0]), float(item[1])))
                resids_all.append(1.0)

    total_count = len(pts_all)
    if total_count == 0:
        return canvas, 0, 0

    # 2. Determine target selection count based on density mode
    d_mode_lower = density_mode.lower()
    if "key" in d_mode_lower or "30" in d_mode_lower:
        target_cap = 30
        is_key_mode = True
    elif "all" in d_mode_lower:
        target_cap = min(total_count, 300)
        is_key_mode = False
    else:
        # Default: dense spatially distributed (approx 150-180 points)
        target_cap = min(total_count, 180)
        is_key_mode = False

    if max_pts is not None:
        target_cap = min(target_cap, max_pts)

    # 3. Spatial Grid Sampling when total > target_cap
    if total_count <= target_cap:
        selected_indices = list(range(total_count))
    else:
        gw, gh = 8, 8
        cell_w = max(1.0, w / gw)
        cell_h = max(1.0, h / gh)
        bins = {}
        for idx, (rx, ry) in enumerate(pts_all):
            gx = min(int(rx // cell_w), gw - 1)
            gy = min(int(ry // cell_h), gh - 1)
            cell = (gx, gy)
            if cell not in bins:
                bins[cell] = []
            bins[cell].append(idx)

        # Sort within each cell by lowest residual
        if len(resids_all) == total_count:
            for cell in bins:
                bins[cell].sort(key=lambda i: resids_all[i])

        selected_indices = []
        occupied_cells = list(bins.keys())
        while len(selected_indices) < target_cap and any(len(bins[c]) > 0 for c in occupied_cells):
            for c in occupied_cells:
                if bins[c]:
                    selected_indices.append(bins[c].pop(0))
                    if len(selected_indices) >= target_cap:
                        break

    drawn_count = len(selected_indices)

    # 4. Color Palette for distinct visual grouping
    palette = [
        (56, 189, 248), (52, 211, 153), (251, 191, 36), (244, 114, 182),
        (167, 139, 250), (96, 165, 250), (74, 222, 128), (253, 224, 71),
        (236, 72, 153), (129, 140, 248), (45, 212, 191), (251, 146, 60)
    ]

    # 5. Draw connecting lines on semi-transparent overlay
    overlay = canvas.copy()
    for idx, pt_idx in enumerate(selected_indices):
        rx, ry = pts_all[pt_idx]
        p1 = (int(rx), int(ry + 50))
        p2 = (int(rx + w), int(ry + 50))
        col = palette[idx % len(palette)]
        line_thickness = 2 if is_key_mode else 1
        cv2.line(overlay, p1, p2, col, line_thickness, cv2.LINE_AA)

    alpha = 0.65 if is_key_mode else 0.50
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # 6. Draw clean point markers on top of lines
    for idx, pt_idx in enumerate(selected_indices):
        rx, ry = pts_all[pt_idx]
        p1 = (int(rx), int(ry + 50))
        p2 = (int(rx + w), int(ry + 50))
        col = palette[idx % len(palette)]

        if is_key_mode:
            # Key feature style: Numbered circular target crosshairs
            for pt in (p1, p2):
                cv2.circle(canvas, pt, 6, col, 2, cv2.LINE_AA)
                cv2.circle(canvas, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.line(canvas, (pt[0]-8, pt[1]), (pt[0]+8, pt[1]), col, 1, cv2.LINE_AA)
                cv2.line(canvas, (pt[0], pt[1]-8), (pt[0], pt[1]+8), col, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"#{idx+1}", (p1[0] + 10, p1[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"#{idx+1}", (p2[0] + 10, p2[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Dense style: Small filled 3px circular dots with 1px white center (no text clutter)
            for pt in (p1, p2):
                cv2.circle(canvas, pt, 3, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 1, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas, drawn_count, total_count


def render_metrics_summary(results: Dict[str, Any]):
    """Renders high-level scientific metrics cards for technical inspections."""
    status = results.get("status", "UNKNOWN")
    score = results.get("confidence_score", 0.0)
    inliers = results.get("inlier_count", 0)
    candidates = results.get("total_candidates", 0)
    ratio = results.get("inlier_ratio", 0.0)
    mean_res = results.get("mean_inlier_residual_px", 0.0)
    matcher = results.get("matcher_used", results.get("estimator_name", "USAC_MAGSAC++"))

    gt_data = results.get("gt_rmse_data") or results.get("gt_comparison") or {}
    rmse_str = f"{gt_data.get('rmse_px', gt_data.get('corner_rmse_px', 0.0)):.4f} px" if gt_data.get("ground_truth_available", False) else "N/A"

    cov_score = results.get("spatial_coverage", {}).get("coverage_score", 0.0)

    # Status Banner
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 14px 20px; margin-bottom: 16px;">
        <div>
            <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Registration Outcome</div>
            <div style="margin-top: 4px;">{render_status_pill(status)}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600;">CONFIDENCE SCORE</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #38BDF8; font-family: monospace;">{score:.2f} <span style="font-size: 0.85rem; color: #64748B;">/ 1.0</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if results.get("summary_reason"):
        st.info(f"**Diagnostic Justification**: {results['summary_reason']}")

    # Metric Cards Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Ground-Truth RMSE", rmse_str, "Synthetic GT transfer" if rmse_str != "N/A" else "Real/No-GT pair"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Verified Inliers", f"{inliers} / {candidates}", f"Inlier Ratio: {ratio*100:.1f}%"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Spatial Coverage", f"{cov_score:.2f}", "Grid + Convex Hull spread"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Mean Residual", f"{mean_res:.3f} px", f"Matcher: {matcher}"), unsafe_allow_html=True)
