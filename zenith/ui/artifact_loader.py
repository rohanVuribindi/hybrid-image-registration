"""
Safe and robust artifact discovery and loader for Zenith outputs.
"""
import os
import glob
import json
import csv
from typing import Dict, Any, List, Optional
import pandas as pd


def get_zenith_outputs_dir() -> str:
    """Returns absolute path to zenith_outputs directory."""
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, "zenith_outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


def list_available_mvps() -> List[str]:
    """Lists all available MVP output directories."""
    base = get_zenith_outputs_dir()
    mvp_dirs = []
    for i in range(1, 9):
        dir_name = f"mvp{i}"
        if os.path.exists(os.path.join(base, dir_name)):
            mvp_dirs.append(dir_name)
    return mvp_dirs


def load_mvp_summary(mvp_name: str) -> Optional[Dict[str, Any]]:
    """Loads run summary or metrics JSON for an MVP safely."""
    mvp_dir = os.path.join(get_zenith_outputs_dir(), mvp_name)
    if not os.path.exists(mvp_dir):
        return None

    # Search for summary JSON files in order of priority
    patterns = [
        f"{mvp_name}_run_summary.json",
        "mvp8_ensemble_benchmark.json",
        "zenith_mvp7_final_report.json",
        "mvp7_evaluation_matrix.json",
        "mvp3_evaluation_matrix.json",
        "*_summary.json",
        "*_report.json",
        "*_metrics.json"
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(mvp_dir, pat))
        if matches:
            try:
                with open(matches[0], "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def find_mvp_artifacts(mvp_name: str, case_filter: Optional[str] = None) -> Dict[str, str]:
    """Finds all PNG and CSV artifacts in an MVP folder, optionally filtered by case prefix."""
    mvp_dir = os.path.join(get_zenith_outputs_dir(), mvp_name)
    if not os.path.exists(mvp_dir):
        return {}

    artifacts = {}
    for f in sorted(os.listdir(mvp_dir)):
        full_path = os.path.join(mvp_dir, f)
        if not os.path.isfile(full_path):
            continue

        f_lower = f.lower()
        if case_filter:
            cf_clean = case_filter.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            # If case filter specified and not in filename, skip unless it's general
            if cf_clean not in f_lower and not f_lower.startswith(cf_clean[:8]):
                continue
        
        # Categorize artifact
        if f_lower.endswith(".png"):
            if "enhanced_warped" in f_lower or "enhanced_image" in f_lower:
                artifacts["enhanced_image"] = full_path
            elif "registered_warped" in f_lower or "registered" in f_lower:
                artifacts["registered_image"] = full_path
            elif "enhanced_alignment_overlay" in f_lower or "enhanced_overlay" in f_lower:
                artifacts["enhanced_alignment_overlay"] = full_path
            elif "alignment_overlay" in f_lower or "overlay" in f_lower:
                artifacts["alignment_overlay"] = full_path
            elif "checkerboard" in f_lower:
                artifacts["checkerboard_blend"] = full_path
            elif "matches" in f_lower:
                artifacts["matches_vis"] = full_path
            elif "input_pair" in f_lower or "pair" in f_lower:
                artifacts["input_pair"] = full_path
            elif "pc_representation_ref" in f_lower:
                artifacts["pc_ref"] = full_path
            elif "pc_representation_src" in f_lower:
                artifacts["pc_src"] = full_path
            else:
                artifacts[f"image_{f}"] = full_path

        elif f_lower.endswith(".csv"):
            if "correspondences" in f_lower or "corr" in f_lower:
                artifacts["correspondences_csv"] = full_path
            else:
                artifacts[f"csv_{f}"] = full_path

        elif f_lower.endswith(".json"):
            if "summary" in f_lower or "report" in f_lower:
                artifacts["summary_json"] = full_path
            elif "metrics" in f_lower:
                artifacts["metrics_json"] = full_path

    return artifacts


def load_correspondence_dataframe(csv_path: str, max_rows: int = 2000) -> Optional[pd.DataFrame]:
    """Safely loads a correspondence CSV into a pandas DataFrame."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, nrows=max_rows)
        return df
    except Exception:
        return None


def load_mvp7_evaluation_matrix() -> List[Dict[str, Any]]:
    """Loads MVP7 evaluation matrix records from zenith_outputs/mvp7."""
    mvp7_dir = os.path.join(get_zenith_outputs_dir(), "mvp7")
    report_file = os.path.join(mvp7_dir, "zenith_mvp7_final_report.json")
    
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "evaluation_matrix" in data:
                    return data["evaluation_matrix"]
        except Exception:
            pass
            
    # Fallback to loading directly from run summary
    summary = load_mvp_summary("mvp7")
    if summary and isinstance(summary, dict):
        if "evaluation_matrix" in summary:
            return summary["evaluation_matrix"]
            
    return []
