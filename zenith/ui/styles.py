"""
Scientific space mission styling and CSS for Zenith Streamlit UI.
Inspired by NASA/JPL/ISRO space mission interfaces and satellite imaging systems.
"""

DARK_SPACE_CSS = """
<style>
/* Space Dark Theme & Typography */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #F8FAFC;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at 50% 0%, #0B1120 0%, #070B14 100%) !important;
}

/* Base Card Styles */
.zenith-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(10, 15, 29, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
}

.zenith-card:hover {
    border-color: rgba(56, 189, 248, 0.45);
    box-shadow: 0 6px 28px rgba(56, 189, 248, 0.08);
}

/* Mission Control Header */
.mission-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    padding: 6px 0 16px 0;
    margin-bottom: 24px;
}

.mission-title-group {
    display: flex;
    flex-direction: column;
}

.mission-brand {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}

.mission-brand span {
    color: #38BDF8;
}

.mission-sub {
    font-size: 0.85rem;
    color: #94A3B8;
    letter-spacing: 0.04em;
    font-weight: 500;
}

.mission-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34D399;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Workflow Step Indicator */
.workflow-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 10px;
    padding: 12px 18px;
    margin: 8px 0 20px 0;
    overflow-x: auto;
}

.workflow-step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #94A3B8;
    letter-spacing: 0.04em;
}

.workflow-step-active {
    color: #38BDF8;
    font-weight: 700;
}

.workflow-step-num {
    background: #1E293B;
    border: 1px solid #475569;
    color: #F8FAFC;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

.workflow-step-active .workflow-step-num {
    background: #0284C7;
    border-color: #38BDF8;
    color: #FFFFFF;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
}

.workflow-arrow {
    color: #475569;
    font-size: 0.9rem;
    font-weight: bold;
}

/* Large Mission Result Status Cards */
.result-status-card {
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
}

.result-status-card-success {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 78, 59, 0.35) 100%);
    border: 2px solid #10B981;
}

.result-status-card-warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.14) 0%, rgba(120, 53, 15, 0.35) 100%);
    border: 2px solid #F59E0B;
}

.result-status-card-danger {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.14) 0%, rgba(127, 29, 29, 0.35) 100%);
    border: 2px solid #EF4444;
}

.result-status-title {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.result-status-desc {
    font-size: 1.05rem;
    color: #E2E8F0;
    line-height: 1.55;
    margin: 0;
}

/* Simple 3-Card Metrics */
.simple-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.simple-card-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #94A3B8;
    letter-spacing: 0.09em;
    margin-bottom: 6px;
}

.simple-card-value {
    font-size: 1.65rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}

.simple-card-sub {
    font-size: 0.78rem;
    color: #64748B;
}

/* Alignment Flow Banner */
.alignment-flow-banner {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 6px 0 16px 0;
}

.alignment-flow-pill {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 9999px;
    padding: 8px 22px;
    color: #38BDF8;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Metric Display Box */
.metric-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 4px;
    letter-spacing: 0.05em;
}

.metric-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #38BDF8;
    font-family: 'JetBrains Mono', monospace;
}

.metric-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 4px;
}

/* Stage Step Progress */
.stage-step {
    padding: 8px 12px;
    border-radius: 6px;
    margin: 4px 0;
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.stage-done {
    background: rgba(16, 185, 129, 0.1);
    color: #34D399;
    border-left: 3px solid #10B981;
}

.stage-running {
    background: rgba(56, 189, 248, 0.1);
    color: #38BDF8;
    border-left: 3px solid #38BDF8;
}

.stage-idle {
    background: rgba(30, 41, 59, 0.4);
    color: #64748B;
    border-left: 3px solid #475569;
}

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.status-registered {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.status-low-confidence {
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.status-failed {
    background-color: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
</style>
"""


def render_status_pill(status: str) -> str:
    """Returns HTML for a color-coded status badge."""
    s_upper = (status or "").upper().strip()
    if "REGISTERED" in s_upper or "ACCEPT" in s_upper:
        cls = "status-registered"
        icon = "✓"
    elif "LOW" in s_upper or "FLAG" in s_upper:
        cls = "status-low-confidence"
        icon = "⚠"
    else:
        cls = "status-failed"
        icon = "✕"
    return f'<span class="status-badge {cls}">{icon} {s_upper}</span>'


def render_metric_card(label: str, value: str, subtext: str = "") -> str:
    """Returns HTML for a metric display box."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""
    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{value}</div>
        {sub_html}
    </div>
    """
