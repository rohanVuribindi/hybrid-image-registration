"""
ZENITH SIH 2026 6-Slide Submission Presentation Generator
Complies strictly with the Official SIH 2026 Idea Submission Template.
Outputs:
- SIH26-A0H-T363-SIH26166_Presentation.pptx
- SIH26-A0H-T363-SIH26166_Presentation.pdf
"""

import os
import sys
import shutil
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE

# Color Constants
C_NAVY = RGBColor(14, 30, 64)        # #0E1E40
C_SPACE_BLUE = RGBColor(26, 54, 93)  # #1A365D
C_BLUE = RGBColor(43, 108, 176)      # #2B6CB0
C_SLATE = RGBColor(45, 55, 72)       # #2D3748
C_MUTED = RGBColor(113, 128, 150)    # #718096
C_CARD_BG = RGBColor(245, 248, 252)  # #F5F8FC
C_CARD_BORDER = RGBColor(203, 213, 224) # #CBD5E0
C_WHITE = RGBColor(255, 255, 255)
C_GOLD = RGBColor(192, 86, 33)       # #C05621 (Dark Amber/Gold)
C_GREEN_DARK = RGBColor(34, 84, 61)  # #22543D
C_GREEN_BG = RGBColor(230, 255, 245)
C_TABLE_HDR = RGBColor(26, 54, 93)   # #1A365D
C_ROW_ALT = RGBColor(244, 247, 251)  # #F4F7FB
C_BOX_BG = RGBColor(240, 244, 251)   # #F0F4FB

FONT_HEADING = "Arial"
FONT_BODY = "Arial"

def add_header(slide, title_text, subtitle_text=""):
    tb = slide.shapes.add_textbox(Inches(1.75), Inches(0.18), Inches(8.8), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = FONT_HEADING
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = FONT_BODY
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = C_BLUE

def add_footer(slide, slide_num):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.12), Inches(13.33), Inches(0.38))
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_SPACE_BLUE
    shape.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.6), Inches(7.15), Inches(10.0), Inches(0.32))
    tf = tb.text_frame
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "@SIH Idea submission- Template"
    p.font.name = FONT_BODY
    p.font.size = Pt(9.5)
    p.font.color.rgb = C_WHITE

    tb_num = slide.shapes.add_textbox(Inches(12.0), Inches(7.15), Inches(0.8), Inches(0.32))
    tf_num = tb_num.text_frame
    tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
    p_num = tf_num.paragraphs[0]
    p_num.text = str(slide_num)
    p_num.alignment = PP_ALIGN.RIGHT
    p_num.font.name = FONT_BODY
    p_num.font.size = Pt(9.5)
    p_num.font.bold = True
    p_num.font.color.rgb = C_WHITE

    oval = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.4), Inches(0.18), Inches(1.18), Inches(0.80))
    oval.fill.background()
    oval.line.color.rgb = C_BLUE
    oval.line.width = Pt(1.5)
    tf_oval = oval.text_frame
    tf_oval.word_wrap = True
    tf_oval.margin_top = Inches(0.08)
    p_o1 = tf_oval.paragraphs[0]
    p_o1.text = "Team"
    p_o1.alignment = PP_ALIGN.CENTER
    p_o1.font.name = FONT_BODY
    p_o1.font.size = Pt(9)
    p_o1.font.color.rgb = C_SLATE
    p_o2 = tf_oval.add_paragraph()
    p_o2.text = "Zenith"
    p_o2.alignment = PP_ALIGN.CENTER
    p_o2.font.name = FONT_BODY
    p_o2.font.size = Pt(9.5)
    p_o2.font.bold = True
    p_o2.font.color.rgb = C_SPACE_BLUE

def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_CARD_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def build_presentation(base_template_path, output_pptx_path):
    print(f"Loading template from {base_template_path}...")
    prs = Presentation(base_template_path)

    # ==========================================
    # SLIDE 1: TITLE PAGE
    # ==========================================
    s1 = prs.slides[0]
    shapes_to_remove = []
    for shape in s1.shapes:
        if shape.has_text_frame:
            txt = shape.text_frame.text
            if "LUNAR-MATCH" in txt or "Problem Statement" in txt or "SMART INDIA" in txt:
                shapes_to_remove.append(shape)
    for sp in shapes_to_remove:
        sp._element.getparent().remove(sp._element)

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10.5), Inches(0.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "SMART INDIA HACKATHON 2026"
    p.font.name = FONT_HEADING
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    tb_sub = s1.shapes.add_textbox(Inches(0.8), Inches(1.15), Inches(10.5), Inches(0.7))
    tf_sub = tb_sub.text_frame
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "TITLE PAGE: ZENITH"
    p_sub.font.name = FONT_HEADING
    p_sub.font.size = Pt(18)
    p_sub.font.bold = True
    p_sub.font.color.rgb = C_SPACE_BLUE
    
    p_sub2 = tf_sub.add_paragraph()
    p_sub2.text = "Autonomous Multi-Modal Lunar Image Registration & Sub-Pixel Photogrammetry"
    p_sub2.font.name = FONT_BODY
    p_sub2.font.size = Pt(12)
    p_sub2.font.color.rgb = C_BLUE

    add_card(s1, 0.8, 1.95, 6.6, 4.85, C_CARD_BG, C_CARD_BORDER)
    tb_fields = s1.shapes.add_textbox(Inches(1.0), Inches(2.05), Inches(6.2), Inches(4.65))
    tf_fields = tb_fields.text_frame
    tf_fields.word_wrap = True

    fields = [
        ("Problem Statement ID:", "SIH26166"),
        ("Problem Statement Title:", "Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)"),
        ("Theme:", "Space Technology"),
        ("PS Category:", "Software"),
        ("Team ID:", "SIH26-A0H-T363"),
        ("Team Name (Registered on portal):", "Team Zenith")
    ]

    for idx, (label, val) in enumerate(fields):
        p = tf_fields.paragraphs[0] if idx == 0 else tf_fields.add_paragraph()
        p.space_after = Pt(7)
        r_lbl = p.add_run()
        r_lbl.text = f"• {label} "
        r_lbl.font.name = FONT_HEADING
        r_lbl.font.size = Pt(10.5)
        r_lbl.font.bold = True
        r_lbl.font.color.rgb = C_SPACE_BLUE

        r_val = p.add_run()
        r_val.text = val
        r_val.font.name = FONT_BODY
        r_val.font.size = Pt(10.5)
        r_val.font.bold = (label in ["Team ID:", "Team Name (Registered on portal):", "Problem Statement ID:"])
        r_val.font.color.rgb = C_NAVY if r_val.font.bold else C_SLATE

    # ==========================================
    # SLIDE 2: IDEA & PROPOSED SOLUTION
    # ==========================================
    s2 = prs.slides[1]
    shapes_to_remove = [s for s in s2.shapes if s.has_text_frame or s.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE]
    for sp in shapes_to_remove:
        sp._element.getparent().remove(sp._element)

    add_header(s2, "IDEA TITLE: ZENITH", "Proposed Solution (Describe your Idea/Solution/Prototype)")
    add_footer(s2, 2)

    # Column 1: Problem & Why is this hard? (Left: 0.6, Width: 3.7)
    add_card(s2, 0.6, 1.15, 3.7, 5.75, C_CARD_BG, C_CARD_BORDER)
    tb_c1 = s2.shapes.add_textbox(Inches(0.75), Inches(1.25), Inches(3.4), Inches(5.55))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    
    p = tf_c1.paragraphs[0]
    p.text = "THE PROBLEM & WHY IS THIS HARD?"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_GOLD
    p.space_after = Pt(4)

    prob_points = [
        ("1. Sun-Angle / Illumination Shifts", "Extreme solar angle shifts cause complete crater shadow inversions (bright-to-dark gradient reversal across identical topography)."),
        ("2. Extreme Scale Disparities", "Resolutions span 0.25 m (OHRC) → 5.0 m (TMC-2) → 80 m (IIRS) per pixel (up to 320x scale gap exceeding standard matchers)."),
        ("3. Cross-Modal Sensor Physics", "Panchromatic optical vs. 256-band hyperspectral cubes with non-linear radiometric response functions."),
        ("Result of Conventional CV", "Standard SIFT/ORB algorithms fail, produce severe spatial clustering, or output high false-positive rates.")
    ]
    for title, desc in prob_points:
        p_t = tf_c1.add_paragraph()
        p_t.text = f"• {title}"
        p_t.font.name = FONT_HEADING
        p_t.font.size = Pt(9.0)
        p_t.font.bold = True
        p_t.font.color.rgb = C_NAVY
        p_t.space_before = Pt(3)

        p_d = tf_c1.add_paragraph()
        p_d.text = desc
        p_d.font.name = FONT_BODY
        p_d.font.size = Pt(8.0)
        p_d.font.color.rgb = C_SLATE
        p_d.space_after = Pt(2)

    # Column 2: ZENITH 6-Stage Formulation (Middle: 4.5, Width: 4.6)
    add_card(s2, 4.5, 1.15, 4.6, 5.75, C_WHITE, C_SPACE_BLUE)
    tb_c2 = s2.shapes.add_textbox(Inches(4.65), Inches(1.25), Inches(4.3), Inches(5.55))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True

    p = tf_c2.paragraphs[0]
    p.text = "OUR SOLUTION: 6-STAGE ZENITH ARCHITECTURE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_SPACE_BLUE
    p.space_after = Pt(4)

    stages = [
        ("ADAPT", "Make source & reference imagery comparable using sensor, metadata, illumination & scale information."),
        ("REPRESENT", "Extract stable structural features via 2D Log-Gabor Phase Congruency & multi-scale pyramids."),
        ("MATCH", "Generate candidate correspondences via hybrid ensemble (RootSIFT + LoFTR deep transformer fallback)."),
        ("VERIFY", "Use USAC_MAGSAC++ and geometric consistency to reject false matches and estimate 8-DOF Homography H."),
        ("REFINE", "Select spatially distributed reliable matches (4x4 grid) and refine locations to sub-pixel precision."),
        ("OUTPUT", "Return registered image (Lanczos-4), verified match points, RMSE, inlier ratio, coverage & confidence status.")
    ]
    for stg, expl in stages:
        p_s = tf_c2.add_paragraph()
        r1 = p_s.add_run()
        r1.text = f"▶ {stg}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(9.0)
        r1.font.bold = True
        r1.font.color.rgb = C_BLUE
        
        r2 = p_s.add_run()
        r2.text = expl
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.0)
        r2.font.color.rgb = C_SLATE
        p_s.space_before = Pt(2)
        p_s.space_after = Pt(2)

    # Column 3: WHY ZENITH? (Right: 9.3, Width: 3.4)
    add_card(s2, 9.3, 1.15, 3.4, 5.75, C_CARD_BG, C_CARD_BORDER)
    tb_c3 = s2.shapes.add_textbox(Inches(9.45), Inches(1.25), Inches(3.1), Inches(5.55))
    tf_c3 = tb_c3.text_frame
    tf_c3.word_wrap = True

    p = tf_c3.paragraphs[0]
    p.text = "WHY ZENITH? KEY INNOVATIONS"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(4)

    innovations = [
        ("Physics & Metadata Aware", "Integrates PDS4 sensor parameters, sun angles, and PCA spectral reduction."),
        ("Illumination Robust", "Phase congruency extracts Fourier harmonic alignment invariant to lighting."),
        ("Hybrid Matcher Consensus", "Combines high-speed RootSIFT with deep LoFTR transformer fallback."),
        ("Spatially Uniform Matches", "4x4 spatial grid balancing eliminates single-crater clustering."),
        ("Sub-Pixel Refinement", "Förstner gradient structure tensors achieve sub-pixel accuracy (< 0.5 px)."),
        ("Confidence-Aware Output", "Decision Engine certifies result: REGISTERED / LOW_CONFIDENCE / FAILED."),
        ("Scientific Safety", "Strictly non-generative; raw registered output preserved with zero hallucination.")
    ]
    for inno, inno_desc in innovations:
        p_i = tf_c3.add_paragraph()
        r1 = p_i.add_run()
        r1.text = f"✓ {inno}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(8.5)
        r1.font.bold = True
        r1.font.color.rgb = C_GREEN_DARK
        
        r2 = p_i.add_run()
        r2.text = inno_desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(7.5)
        r2.font.color.rgb = C_SLATE
        p_i.space_before = Pt(1)
        p_i.space_after = Pt(2)

    # ==========================================
    # SLIDE 3: TECHNICAL APPROACH (Visual 8-Stage Architecture)
    # ==========================================
    s3 = prs.slides[2]
    shapes_to_remove = [s for s in s3.shapes if s.has_text_frame or s.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE]
    for sp in shapes_to_remove:
        sp._element.getparent().remove(sp._element)

    add_header(s3, "TECHNICAL APPROACH", "Technologies to be used & Implementation Methodology")
    add_footer(s3, 3)

    # 8 Stage Blocks (2 columns x 4 rows)
    # Left Col: left = 0.6, width = 5.95
    # Right Col: left = 6.75, width = 5.95
    stage_blocks = [
        ("INPUT PAIR", "Source (OHRC / TMC-2 / IIRS) + Reference Base Map (LROC NAC / TMC DEM / Synthetic)", 0.6, 1.15),
        ("1. ADAPT", "Radiometric 1-99% quantile scaling, sensor bit-depth normalization, IIRS 256-band PCA", 0.6, 1.95),
        ("2. REPRESENT", "2D Log-Gabor Phase Congruency (Kovesi, 3 scales, 6 orientations) + Multi-Scale Pyramid", 0.6, 2.75),
        ("3. MATCH", "Multi-Matcher Ensemble: Primary RootSIFT (Hellinger kernel) + LoFTR Deep Transformer", 0.6, 3.55),
        ("4. VERIFY", "USAC_MAGSAC++ robust estimation of 8-DOF Homography H (adaptive noise σ = 3.0 px)", 6.75, 1.15),
        ("5. REFINE", "Spatial 4x4 Grid Inlier Balancing (prevents cluster collapse) + Förstner Structure Tensor", 6.75, 1.95),
        ("6. EVALUATE", "Decision Engine: Inlier Ratio (≥35%), Spatial Coverage (≥0.35), Reprojection Residuals", 6.75, 2.75),
        ("7. OUTPUT", "Lanczos-4 Sub-Pixel Warping → Raw Registered Output + JSON Telemetry + Match Table", 6.75, 3.55)
    ]

    for title, desc, b_left, b_top in stage_blocks:
        add_card(s3, b_left, b_top, 5.95, 0.72, C_BOX_BG, C_SPACE_BLUE)
        tb_b = s3.shapes.add_textbox(Inches(b_left + 0.12), Inches(b_top + 0.04), Inches(5.7), Inches(0.64))
        tf_b = tb_b.text_frame
        tf_b.word_wrap = True
        tf_b.margin_left = tf_b.margin_top = tf_b.margin_right = tf_b.margin_bottom = 0
        
        # Line 1: Bold Stage Header
        p1 = tf_b.paragraphs[0]
        p1.text = f"▶ {title}"
        p1.font.name = FONT_HEADING
        p1.font.size = Pt(9.5)
        p1.font.bold = True
        p1.font.color.rgb = C_NAVY if "INPUT" in title or "OUTPUT" in title else C_SPACE_BLUE
        p1.space_after = Pt(1)

        # Line 2: Explanation
        p2 = tf_b.add_paragraph()
        p2.text = desc
        p2.font.name = FONT_BODY
        p2.font.size = Pt(8.0)
        p2.font.color.rgb = C_SLATE

    # Core Design Principle Banner (Left: 0.6, Top: 4.40, Width: 12.1, Height: 0.75)
    add_card(s3, 0.6, 4.40, 12.1, 0.75, C_GREEN_BG, C_GREEN_DARK)
    tb_dp = s3.shapes.add_textbox(Inches(0.75), Inches(4.45), Inches(11.8), Inches(0.65))
    tf_dp = tb_dp.text_frame
    tf_dp.word_wrap = True
    p_dp = tf_dp.paragraphs[0]
    r_dp1 = p_dp.add_run()
    r_dp1.text = "CORE DESIGN PRINCIPLE: "
    r_dp1.font.name = FONT_HEADING
    r_dp1.font.size = Pt(9.5)
    r_dp1.font.bold = True
    r_dp1.font.color.rgb = C_GREEN_DARK

    r_dp2 = p_dp.add_run()
    r_dp2.text = "AI / feature matching proposes correspondences → Geometry verifies them → Spatial balancing improves coverage → Refinement improves precision → Quality metrics decide whether the result is trustworthy."
    r_dp2.font.name = FONT_BODY
    r_dp2.font.size = Pt(9.0)
    r_dp2.font.bold = True
    r_dp2.font.color.rgb = C_NAVY

    # Technology Stack Box (Left: 0.6, Top: 5.25, Width: 12.1, Height: 1.65)
    add_card(s3, 0.6, 5.25, 12.1, 1.65, C_CARD_BG, C_CARD_BORDER)
    tb_tech = s3.shapes.add_textbox(Inches(0.75), Inches(5.32), Inches(11.8), Inches(1.50))
    tf_tech = tb_tech.text_frame
    tf_tech.word_wrap = True

    p = tf_tech.paragraphs[0]
    p.text = "IMPLEMENTATION TECHNOLOGY STACK"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(2)

    techs = [
        ("Core & Computer Vision:", "Python 3.10+, OpenCV (USAC_MAGSAC++, Lanczos-4, CLAHE), NumPy, SciPy, scikit-image"),
        ("Deep Learning & Matchers:", "PyTorch, LoFTR Transformer (Local Feature TRansformer), Kornia, FLANN Matcher"),
        ("Geospatial & Metadata:", "GDAL, Rasterio, Planetary Data System (PDS4/LBL) parser, IAU Lunar Reference Models"),
        ("Interface & Demonstration:", "Streamlit Mission Control UI, Interactive Comparison Slider, Matplotlib, ReportLab")
    ]
    for cat, items in techs:
        p_t = tf_tech.add_paragraph()
        r1 = p_t.add_run()
        r1.text = f"• {cat} "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(8.5)
        r1.font.bold = True
        r1.font.color.rgb = C_SPACE_BLUE
        
        r2 = p_t.add_run()
        r2.text = items
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_SLATE
        p_t.space_before = Pt(1)
        p_t.space_after = Pt(1)

    # ==========================================
    # SLIDE 4: FEASIBILITY AND VIABILITY
    # ==========================================
    s4 = prs.slides[3]
    shapes_to_remove = [s for s in s4.shapes if s.has_text_frame or s.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE]
    for sp in shapes_to_remove:
        sp._element.getparent().remove(sp._element)

    add_header(s4, "FEASIBILITY AND VIABILITY", "Analysis of Feasibility, Potential Challenges & Risk Mitigation Strategies")
    add_footer(s4, 4)

    # Left: Challenges Table (Top: 1.15, Left: 0.6, Width: 7.6, Height: 5.75)
    add_card(s4, 0.6, 1.15, 7.6, 5.75, C_WHITE, C_CARD_BORDER)
    tb_tbl = s4.shapes.add_textbox(Inches(0.75), Inches(1.25), Inches(7.3), Inches(5.55))
    tf_tbl = tb_tbl.text_frame
    tf_tbl.word_wrap = True

    p = tf_tbl.paragraphs[0]
    p.text = "CHALLENGES & ZENITH TECHNICAL MITIGATION"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_SPACE_BLUE
    p.space_after = Pt(4)

    challenges = [
        ("Sun-Angle / Shadow Shifts", "2D Log-Gabor Phase Congruency", "Fourier harmonic phase alignment is completely contrast/brightness invariant."),
        ("Extreme Scale Gap (320x)", "Multi-Scale Pyramids & Hierarchy", "Multi-resolution octave decimation handles large scale differences progressively."),
        ("Cross-Modal Mismatch", "PCA Reduction + Structural Maps", "Matches shared surface topography without requiring raw intensity equivalence."),
        ("Severe Outlier Matches", "USAC_MAGSAC++ Estimator", "Marginalizes over continuous noise scale; tolerates up to 85% outlier rates."),
        ("Match Clustering Collapse", "4x4 Spatial Grid Balancing", "Enforces distributed inliers across 16 cells; prevents single-crater bias."),
        ("Low-Texture Maria Basins", "LoFTR Transformer Fallback", "Global contextual cross-attention finds matches where local gradients fail."),
        ("Large Swath Computation", "Tile-Based Processing & FP16", "Memory-capped processing (< 1.4 GB VRAM) with fast CPU fallback.")
    ]
    for chal, resp, why in challenges:
        p_c = tf_tbl.add_paragraph()
        r1 = p_c.add_run()
        r1.text = f"• {chal}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(9.0)
        r1.font.bold = True
        r1.font.color.rgb = C_NAVY
        
        r2 = p_c.add_run()
        r2.text = f"{resp} → {why}"
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_SLATE
        p_c.space_before = Pt(2)
        p_c.space_after = Pt(2)

    # Right Column: Current Prototype Status & Future Roadmap (Left: 8.4, Width: 4.3)
    add_card(s4, 8.4, 1.15, 4.3, 3.25, C_GREEN_BG, C_GREEN_DARK)
    tb_proto = s4.shapes.add_textbox(Inches(8.55), Inches(1.25), Inches(4.0), Inches(3.05))
    tf_proto = tb_proto.text_frame
    tf_proto.word_wrap = True

    p = tf_proto.paragraphs[0]
    p.text = "CURRENT PROTOTYPE STATUS (DEMONSTRATED)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_GREEN_DARK
    p.space_after = Pt(2)

    proto_items = [
        ("✓ End-to-End Pipeline:", "Verified across MVP1–MVP8 test runners with 100% test pass rate."),
        ("✓ Measured Sub-Pixel RMSE:", "0.154 px (Nominal Optical), 0.730 px (Difficult Terminator Ramp)."),
        ("✓ Streamlit Web Interface:", "Live Mission Control UI with interactive before/after split slider."),
        ("✓ Feature Alignment:", "Spatially distributed inlier correspondence vector visualization."),
        ("✓ Raw Output Preserved:", "Strictly non-generative; raw registered homography & artifacts exported.")
    ]
    for lbl, desc in proto_items:
        p_pr = tf_proto.add_paragraph()
        r1 = p_pr.add_run()
        r1.text = f"{lbl} "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(8.0)
        r1.font.bold = True
        r1.font.color.rgb = C_NAVY
        
        r2 = p_pr.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.0)
        r2.font.color.rgb = C_SLATE
        p_pr.space_before = Pt(1)
        p_pr.space_after = Pt(1)

    # Future Roadmap Box
    add_card(s4, 8.4, 4.50, 4.3, 2.40, C_CARD_BG, C_CARD_BORDER)
    tb_fut = s4.shapes.add_textbox(Inches(8.55), Inches(4.58), Inches(4.0), Inches(2.20))
    tf_fut = tb_fut.text_frame
    tf_fut.word_wrap = True

    p = tf_fut.paragraphs[0]
    p.text = "FUTURE VALIDATION & DEPLOYMENT ROADMAP"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_SPACE_BLUE
    p.space_after = Pt(2)

    future_items = [
        ("• Expanded Chandrayaan-2 Dataset:", "Multi-orbit validation across wider lunar longitudes/latitudes."),
        ("• DEM-Assisted Geometry:", "Integration with LOLA/TMC DEMs for extreme 3D relief correction."),
        ("• Space-Hardened FPGA Quantization:", "INT8 execution targeted for onboard Terrain Relative Navigation.")
    ]
    for lbl, desc in future_items:
        p_ft = tf_fut.add_paragraph()
        r1 = p_ft.add_run()
        r1.text = f"{lbl} "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(8.0)
        r1.font.bold = True
        r1.font.color.rgb = C_SPACE_BLUE
        
        r2 = p_ft.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.0)
        r2.font.color.rgb = C_SLATE
        p_ft.space_before = Pt(1)
        p_ft.space_after = Pt(1)

    # ==========================================
    # SLIDE 5: IMPACT AND BENEFITS
    # ==========================================
    s5 = prs.slides[4]
    shapes_to_remove = [s for s in s5.shapes if s.has_text_frame or s.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE]
    for sp in shapes_to_remove:
        sp._element.getparent().remove(sp._element)

    add_header(s5, "IMPACT AND BENEFITS", "Potential Impact on Space Operations & Multi-Dimensional Benefits")
    add_footer(s5, 5)

    # Left Column: Projected Outcomes (Width: 5.9, Left: 0.6)
    add_card(s5, 0.6, 1.15, 5.9, 5.75, C_WHITE, C_SPACE_BLUE)
    tb_out = s5.shapes.add_textbox(Inches(0.75), Inches(1.25), Inches(5.6), Inches(5.55))
    tf_out = tb_out.text_frame
    tf_out.word_wrap = True

    p = tf_out.paragraphs[0]
    p.text = "PROJECTED DELIVERABLES & CAPABILITIES"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_SPACE_BLUE
    p.space_after = Pt(4)

    outcomes = [
        ("1. Automated Lunar Registration", "Aligns difficult planetary images without relying entirely on manual ground control tie points."),
        ("2. Reliable Correspondences", "Automatically identifies geometrically consistent, outlier-free match points across disparate sensors."),
        ("3. Spatially Distributed Matches", "4x4 grid balancing ensures correspondences span the full terrain footprint rather than clustering."),
        ("4. Sub-Pixel Refinement", "Gradient structure tensor optimization achieves sub-pixel localization accuracy (< 0.5 px)."),
        ("5. Quality-Aware Output", "Delivers verified aligned image + full telemetry (RMSE, Inlier Ratio, Spatial Coverage, Confidence Status)."),
        ("6. Reusable Planetary Framework", "Modular open architecture adaptable to future ISRO and international deep space exploration missions.")
    ]
    for title, desc in outcomes:
        p_o = tf_out.add_paragraph()
        r1 = p_o.add_run()
        r1.text = f"{title}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(9.0)
        r1.font.bold = True
        r1.font.color.rgb = C_NAVY
        
        r2 = p_o.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_SLATE
        p_o.space_before = Pt(2)
        p_o.space_after = Pt(2)

    # Right Column: Multi-Dimensional Benefits (Width: 5.9, Left: 6.8)
    add_card(s5, 6.8, 1.15, 5.9, 5.75, C_CARD_BG, C_CARD_BORDER)
    tb_ben = s5.shapes.add_textbox(Inches(6.95), Inches(1.25), Inches(5.6), Inches(5.55))
    tf_ben = tb_ben.text_frame
    tf_ben.word_wrap = True

    p = tf_ben.paragraphs[0]
    p.text = "MULTI-SECTOR VALUE & BENEFITS"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(4)

    benefits = [
        ("Scientific Value", "Enables precise co-registration of morphology (OHRC), topography (TMC-2), and mineralogy (IIRS) for landing site characterization (Chandrayaan-3/4, Artemis, Lunar South Pole water-ice mapping)."),
        ("Operational Value", "Can significantly reduce repetitive manual image-alignment effort for orbital cartographers, accelerating baseline mosaic generation at planetary data centers."),
        ("Technical Value", "Provides measurable, explainable, and certified registration quality with autonomous confidence gates (eliminates undetected silent alignment errors)."),
        ("Strategic & Educational", "Strengthens Indian space-technology capabilities in autonomous computer vision and remote sensing, demonstrating indigenous student innovation in planetary science.")
    ]
    for b_title, b_desc in benefits:
        p_b = tf_ben.add_paragraph()
        r1 = p_b.add_run()
        r1.text = f"★ {b_title}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = C_BLUE
        
        r2 = p_b.add_run()
        r2.text = b_desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_SLATE
        p_b.space_before = Pt(3)
        p_b.space_after = Pt(3)

    # ==========================================
    # SLIDE 6: RESEARCH AND REFERENCES (Perfect Inset Spacing)
    # ==========================================
    s6 = prs.slides[5]
    shapes_to_remove = [s for s in s6.shapes if s.has_text_frame or s.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE]
    for sp in shapes_to_remove:
        sp._element.getparent().remove(sp._element)

    add_header(s6, "RESEARCH AND REFERENCES", "Details / Links of Reference Literature and Scientific Grounding")
    add_footer(s6, 6)

    # Left Column: Research Literature & Mission Sources (Left: 0.6, Width: 5.6)
    add_card(s6, 0.6, 1.12, 5.6, 5.80, C_CARD_BG, C_CARD_BORDER)
    tb_ref = s6.shapes.add_textbox(Inches(0.75), Inches(1.38), Inches(5.3), Inches(5.40))
    tf_ref = tb_ref.text_frame
    tf_ref.word_wrap = True

    p = tf_ref.paragraphs[0]
    p.text = "PRIMARY RESEARCH & DATA SOURCES"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_SPACE_BLUE
    p.space_after = Pt(4)

    citations = [
        ("[1] ISRO / ISSDC", "Chandrayaan-2 Data Explorer & Payload Documentation (OHRC 0.25m, TMC-2 5m, IIRS 80m), PRADAN Archive."),
        ("[2] NASA PDS / LROC", "Lunar Reconnaissance Orbiter Camera Data Archive & USGS Astrogeology ISIS Documentation."),
        ("[3] Kovesi, P. (1999/2003)", "\"Image Features From Phase Congruency,\" Videre / DICTA (Illumination-invariant harmonic model)."),
        ("[4] Lowe, D. G. (2004)", "\"Distinctive Image Features from Scale-Invariant Keypoints (SIFT),\" IJCV."),
        ("[5] Arandjelović & Zisserman (2012)", "\"Three things everyone should know to improve object retrieval (RootSIFT),\" CVPR."),
        ("[6] Sun, J. et al. (2021)", "\"LoFTR: Detector-Free Local Feature Matching with Transformers,\" IEEE/CVF CVPR."),
        ("[7] Barath, D. et al. (2020)", "\"MAGSAC++, a Fast, Reliable and Accurate Robust Estimator,\" IEEE/CVF CVPR."),
        ("[8] Li, J. et al. (2020)", "\"RIFT: Multi-Modal Image Matching Based on Radiation-Variation Insensitive Feature Transform,\" IEEE TIP.")
    ]
    for cit_num, cit_text in citations:
        p_c = tf_ref.add_paragraph()
        r1 = p_c.add_run()
        r1.text = f"{cit_num} "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(8.0)
        r1.font.bold = True
        r1.font.color.rgb = C_NAVY
        
        r2 = p_c.add_run()
        r2.text = cit_text
        r2.font.name = FONT_BODY
        r2.font.size = Pt(7.5)
        r2.font.color.rgb = C_SLATE
        p_c.space_before = Pt(1)
        p_c.space_after = Pt(1)

    # Right Column: Comparative Evaluation Table (Left: 6.4, Width: 6.3)
    add_card(s6, 6.4, 1.12, 6.3, 5.80, C_WHITE, C_SPACE_BLUE)
    
    # Title placed comfortably inside card, centered
    tb_t = s6.shapes.add_textbox(Inches(6.55), Inches(1.28), Inches(6.0), Inches(0.35))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = "CAPABILITY & BENCHMARK COMPARISON MATRIX"
    p_t.alignment = PP_ALIGN.CENTER
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(10.5)
    p_t.font.bold = True
    p_t.font.color.rgb = C_SPACE_BLUE

    # Add PowerPoint Table for structured display (placed at top=1.75)
    rows_data = [
        ["Capability", "SIFT", "LoFTR", "RIFT", "ZENITH"],
        ["Illumination / Shadow Invariance", "✗ Fails", "△ Partial", "✓ High", "✓ Superior (Phase Congruency)"],
        ["Scale Disparities (> 2x)", "△ Moderate", "✗ Weak", "△ Moderate", "✓ High (Pyramid+Ensemble)"],
        ["Cross-Modal / Hyperspectral", "✗ None", "△ Limited", "✓ Good", "✓ Robust (PCA+Structural)"],
        ["Sub-Pixel Refinement", "✗ Coarse", "△ Approx.", "✗ None", "✓ Structure Tensor (<0.5px)"],
        ["Robust Outlier Rejection", "✗ RANSAC", "✗ RANSAC", "✗ RANSAC", "✓ USAC_MAGSAC++"],
        ["Spatial Match Distribution", "✗ Clustered", "✗ Clustered", "✗ Clustered", "✓ 4x4 Grid Balancing"],
        ["Autonomous Quality Gate", "✗ None", "✗ None", "✗ None", "✓ Decision Engine (3-State)"]
    ]

    tbl_shape = s6.shapes.add_table(len(rows_data), 5, Inches(6.55), Inches(1.72), Inches(6.0), Inches(4.55))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(2.2)
    tbl.columns[1].width = Inches(0.8)
    tbl.columns[2].width = Inches(0.8)
    tbl.columns[3].width = Inches(0.8)
    tbl.columns[4].width = Inches(1.4)

    for r_idx, row in enumerate(rows_data):
        for c_idx, cell_value in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = cell_value
            cell.margin_left = cell.margin_right = Inches(0.04)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_BODY
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 or c_idx == 4 else PP_ALIGN.CENTER
            
            if r_idx == 0:
                p.font.bold = True
                p.font.size = Pt(8.5)
                p.font.color.rgb = C_WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_TABLE_HDR
            else:
                p.font.size = Pt(7.5)
                if c_idx == 4:
                    p.font.bold = True
                    p.font.color.rgb = C_GREEN_DARK
                elif "✓" in cell_value:
                    p.font.color.rgb = C_GREEN_DARK
                elif "✗" in cell_value:
                    p.font.color.rgb = RGBColor(197, 48, 48)
                else:
                    p.font.color.rgb = C_SLATE

                cell.fill.solid()
                if r_idx % 2 == 1:
                    cell.fill.fore_color.rgb = C_ROW_ALT
                else:
                    cell.fill.fore_color.rgb = C_WHITE

    # Legend underneath table
    tb_leg = s6.shapes.add_textbox(Inches(6.55), Inches(6.40), Inches(6.0), Inches(0.35))
    tf_leg = tb_leg.text_frame
    tf_leg.margin_left = tf_leg.margin_top = tf_leg.margin_right = tf_leg.margin_bottom = 0
    p_l = tf_leg.paragraphs[0]
    p_l.text = "Legend: ✓ = Supported / High  ·  △ = Partial  ·  ✗ = Unsupported / Fails"
    p_l.font.name = FONT_BODY
    p_l.font.size = Pt(7.5)
    p_l.font.italic = True
    p_l.font.color.rgb = C_MUTED

    # Save presentation
    prs.save(output_pptx_path)
    print(f"Successfully created polished presentation: {output_pptx_path}")

def convert_pptx_to_pdf_com(pptx_path, pdf_path):
    print(f"Converting PPTX to PDF using PowerPoint COM: {pptx_path} -> {pdf_path}...")
    import win32com.client
    import pythoncom
    
    pythoncom.CoInitialize()
    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    abs_pptx = os.path.abspath(pptx_path)
    abs_pdf = os.path.abspath(pdf_path)
    
    deck = powerpoint.Presentations.Open(abs_pptx, False, False, False)
    deck.SaveAs(abs_pdf, 32)
    deck.Close()
    powerpoint.Quit()
    pythoncom.CoUninitialize()
    print(f"Successfully converted to PDF: {abs_pdf}")

if __name__ == "__main__":
    base_ppt = r"c:\Users\rohan\Downloads\LUNAR-MATCH_SIH2026_v5.pptx"
    out_ppt1 = r"c:\Users\rohan\Downloads\SIH26-A0H-T363-SIH26166_Presentation.pptx"
    out_ppt2 = r"c:\Users\rohan\Downloads\prototype\docs\SIH26-A0H-T363-SIH26166_Presentation.pptx"
    out_pdf1 = r"c:\Users\rohan\Downloads\SIH26-A0H-T363-SIH26166_Presentation.pdf"
    out_pdf2 = r"c:\Users\rohan\Downloads\prototype\docs\SIH26-A0H-T363-SIH26166_Presentation.pdf"

    build_presentation(base_ppt, out_ppt1)
    
    # Save copy to docs/
    shutil.copyfile(out_ppt1, out_ppt2)
    shutil.copyfile(out_ppt1, base_ppt) # Also update base ppt
    
    convert_pptx_to_pdf_com(out_ppt1, out_pdf1)
    shutil.copyfile(out_pdf1, out_pdf2)
    print("All PPTX and PDF deliverables generated successfully!")
