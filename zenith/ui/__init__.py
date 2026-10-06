"""
ZENITH UI Module for Streamlit Interface.
"""
from .styles import DARK_SPACE_CSS, render_status_pill, render_metric_card
from .components import (
    image_to_base64,
    render_mission_header,
    render_workflow_steps,
    render_pipeline_flowchart_horizontal,
    render_capability_grid,
    render_simple_registration_status,
    render_simple_metrics,
    render_before_after_view,
    render_interactive_comparison_slider,
    create_verified_feature_alignment_image,
    render_metrics_summary
)
from .artifact_loader import (
    get_zenith_outputs_dir,
    list_available_mvps,
    load_mvp_summary,
    find_mvp_artifacts,
    load_correspondence_dataframe,
    load_mvp7_evaluation_matrix
)

__all__ = [
    "DARK_SPACE_CSS",
    "render_status_pill",
    "render_metric_card",
    "image_to_base64",
    "render_mission_header",
    "render_workflow_steps",
    "render_pipeline_flowchart_horizontal",
    "render_capability_grid",
    "render_simple_registration_status",
    "render_simple_metrics",
    "render_before_after_view",
    "render_interactive_comparison_slider",
    "create_verified_feature_alignment_image",
    "render_metrics_summary",
    "get_zenith_outputs_dir",
    "list_available_mvps",
    "load_mvp_summary",
    "find_mvp_artifacts",
    "load_correspondence_dataframe",
    "load_mvp7_evaluation_matrix"
]
