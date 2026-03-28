"""
CodeSense - Reusable UI Components
Streamlit components for scores, issues, DSA cards, feedback panels, etc.
"""

import streamlit as st
from typing import Any, Dict, List, Optional

from constants import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    COLOR_INFO, COLOR_NEUTRAL,
)


# ─── CSS ──────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  code, pre, .stCodeBlock {
    font-family: 'Fira Code', monospace !important;
  }

  /* Card */
  .cs-card {
    background: #1E1E2E;
    border: 1px solid #2D2D3F;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .cs-card-header {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #E2E8F0;
  }

  /* Score ring */
  .score-ring {
    display: flex;
    align-items: center;
    gap: 24px;
  }
  .score-number {
    font-size: 56px;
    font-weight: 700;
    line-height: 1;
  }
  .grade-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
  }

  /* Issue severity */
  .sev-error   { border-left: 4px solid #E53935; padding-left: 12px; }
  .sev-warning { border-left: 4px solid #FDD835; padding-left: 12px; }
  .sev-info    { border-left: 4px solid #00ACC1; padding-left: 12px; }
  .sev-positive{ border-left: 4px solid #43A047; padding-left: 12px; }

  /* Metric pill */
  .metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #252535;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
    color: #A0AEC0;
    margin: 4px;
  }
  .metric-pill b { color: #E2E8F0; }

  /* Badge */
  .badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .badge-critical { background:#E53935; color:#fff; }
  .badge-high     { background:#FF7043; color:#fff; }
  .badge-medium   { background:#FDD835; color:#000; }
  .badge-low      { background:#43A047; color:#fff; }
  .badge-info     { background:#00ACC1; color:#fff; }

  /* Step indicator */
  .step-circle {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #1E88E5;
    color: #fff;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    flex-shrink: 0;
  }

  /* Progress bar */
  .cs-progress-bar {
    background: #252535;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 8px 0;
  }
  .cs-progress-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #1E1E2E;
    border-radius: 8px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 18px;
    color: #A0AEC0;
  }
  .stTabs [aria-selected="true"] {
    background: #1E88E5 !important;
    color: #fff !important;
  }

  /* Button */
  .stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    transform: scale(1.03);
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #1E1E2E;
    border-right: 1px solid #2D2D3F;
  }
</style>
"""


def inject_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─── Score Display ────────────────────────────────────────────────────────────

def score_card(score: float, grade: str, confidence: float,
               label: str = "", improvement: Optional[float] = None) -> None:
    """
    Score card using st.columns layout — avoids nested HTML divs which
    Streamlit's sanitizer strips, causing raw </div> text to appear.
    """
    color = _score_color(score)
    lbl   = label or _score_label(score)
    pct   = int(score)

    # Outer card shell — single div, no nesting, safe
    st.markdown(
        f'<div style="background:#1E1E2E;border:1px solid #2D2D3F;border-radius:12px;padding:16px 20px;margin-bottom:8px">',
        unsafe_allow_html=True,
    )
    col_ring, col_info = st.columns([1, 1.6])

    with col_ring:
        # Flat ring using span elements only (no nested divs)
        st.markdown(f"""
<span style="display:block;width:90px;height:90px;border-radius:50%;
             background:conic-gradient({color} {pct}%,#252535 {pct}%);
             display:flex;align-items:center;justify-content:center;margin:auto">
  <span style="background:#1E1E2E;width:68px;height:68px;border-radius:50%;
               display:flex;flex-direction:column;align-items:center;justify-content:center">
    <b style="font-size:22px;color:{color};line-height:1.1">{score:.0f}</b>
    <small style="font-size:10px;color:#757575">/100</small>
  </span>
</span>""", unsafe_allow_html=True)

    with col_info:
        st.markdown(f'<b style="font-size:36px;color:{color}">{grade}</b>',
                    unsafe_allow_html=True)
        st.markdown(f'<span style="color:#A0AEC0;font-size:13px;display:block">{lbl}</span>',
                    unsafe_allow_html=True)
        st.markdown(f'<span style="color:#757575;font-size:11px">±{confidence:.1f} confidence</span>',
                    unsafe_allow_html=True)
        if improvement is not None:
            sign   = "+" if improvement >= 0 else ""
            color2 = COLOR_SUCCESS if improvement >= 0 else COLOR_ERROR
            st.markdown(
                f'<span style="color:{color2};font-size:12px;display:block;margin-top:4px">' +
                f'{sign}{improvement:.1f} pts vs last</span>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)


def progress_bar(value: float, max_val: float = 100,
                 color: str = COLOR_PRIMARY, label: str = "") -> None:
    pct = min(100, value / max_val * 100) if max_val else 0
    st.markdown(f"""
    <div style="margin:4px 0">
      {"<small style='color:#A0AEC0'>" + label + "</small>" if label else ""}
      <div class="cs-progress-bar">
        <div class="cs-progress-fill" style="width:{pct}%;background:{color}"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Issue Cards ─────────────────────────────────────────────────────────────

def issue_card(item: Dict) -> None:
    """
    Render an issue card. Uses st.code() for code snippets to avoid
    Streamlit's HTML sanitizer stripping nested tags inside st.markdown.
    """
    sev   = item.get("severity", "info")
    title = item.get("title", "")
    msg   = item.get("message", "")
    line  = item.get("line")
    bef   = item.get("code_before", "")
    aft   = item.get("code_after", "")
    res   = item.get("resource")

    border = {
        "error":    "#E53935",
        "warning":  "#FDD835",
        "info":     "#00ACC1",
        "positive": "#43A047",
    }.get(sev, "#757575")

    line_badge = (
        f'<span style="color:#757575;font-size:12px;margin-left:8px">Line {line}</span>'
        if line else ""
    )

    resource_html = ""
    if res and res.get("url"):
        resource_html = (
            f'<a href="{res["url"]}" target="_blank" '
            f'style="color:#1E88E5;font-size:12px;text-decoration:none">'
            f'📖 {res.get("title","Learn more")}</a>'
        )

    # Header rendered via st.markdown (no nested code — safe from sanitizer)
    st.markdown(f"""
    <div style="border-left:4px solid {border};background:#1E1E2E;border-radius:0 8px 8px 0;
                padding:12px 16px;margin-bottom:4px;border:1px solid #2D2D3F;
                border-left:4px solid {border}">
      <div style="font-weight:600;color:#E2E8F0;font-size:14px">{_esc(title)}{line_badge}</div>
      <div style="color:#A0AEC0;font-size:13px;margin-top:4px">{_esc(msg)}</div>
      <div style="margin-top:6px">{resource_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Code snippets via st.code — completely bypasses HTML sanitizer
    if bef:
        st.caption("📄 Your code:")
        st.code(bef, language="python")
    if aft:
        st.caption("✅ Suggestion:")
        st.code(aft, language="python")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ─── DSA Cards ───────────────────────────────────────────────────────────────

def algo_card(algo: Dict) -> None:
    cx    = algo.get("complexity", {})
    conf  = int(algo.get("confidence", 0) * 100)
    color = COLOR_SUCCESS if cx.get("avg", "").startswith("O(log") or \
            cx.get("avg", "").startswith("O(n log") else COLOR_WARNING

    st.markdown(f"""
    <div class="cs-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-weight:600;color:#E2E8F0">🧠 {algo['display_name']}</span>
        <span class="badge badge-info">{algo.get('category','')}</span>
      </div>
      <div style="color:#A0AEC0;font-size:13px;margin:6px 0">
        Lines {algo.get('start_line','?')}–{algo.get('end_line','?')} &nbsp;|&nbsp;
        Confidence: {conf}%
      </div>
      <div style="margin:8px 0">
        <span class="metric-pill">⏱ Best <b>{cx.get('best','?')}</b></span>
        <span class="metric-pill">⏱ Avg <b>{cx.get('avg','?')}</b></span>
        <span class="metric-pill">⏱ Worst <b>{cx.get('worst','?')}</b></span>
        <span class="metric-pill">💾 Space <b>{cx.get('space','?')}</b></span>
      </div>
      <div style="color:#A0AEC0;font-size:13px;font-style:italic">{algo.get('reason','')}</div>
      {"<div style='margin-top:8px;color:#FDD835;font-size:13px'>💡 " + algo.get('suggestion','') + "</div>" if algo.get('suggestion') else ""}
    </div>
    """, unsafe_allow_html=True)


def ds_chip(ds: Dict) -> None:
    st.markdown(
        f'<span class="metric-pill">📦 <b>{ds["display_name"]}</b> (line {ds["line"]})</span>',
        unsafe_allow_html=True,
    )


# ─── Stats Metrics Row ───────────────────────────────────────────────────────

def stat_row(items: List[Dict]) -> None:
    """
    Render a horizontal row of stat cards.
    Each item: {label, value, delta, color, icon}
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            delta_html = ""
            if "delta" in item and item["delta"] is not None:
                d = item["delta"]
                c = COLOR_SUCCESS if d >= 0 else COLOR_ERROR
                delta_html = f'<div style="color:{c};font-size:12px">{"▲" if d >= 0 else "▼"} {abs(d):.1f}</div>'
            col.markdown(f"""
            <div class="cs-card" style="text-align:center;padding:16px">
              <div style="font-size:24px">{item.get('icon','')}</div>
              <div style="font-size:28px;font-weight:700;color:{item.get('color', COLOR_PRIMARY)}">{item['value']}</div>
              <div style="color:#A0AEC0;font-size:13px">{item['label']}</div>
              {delta_html}
            </div>
            """, unsafe_allow_html=True)


# ─── Loading States ───────────────────────────────────────────────────────────

def loading_spinner(message: str = "Analyzing your code...") -> None:
    st.markdown(f"""
    <div style="text-align:center;padding:40px">
      <div style="font-size:32px;animation:spin 1s linear infinite">⚙️</div>
      <div style="color:#A0AEC0;margin-top:12px">{message}</div>
    </div>
    <style>@keyframes spin {{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}</style>
    """, unsafe_allow_html=True)


# ─── Achievement Badge ───────────────────────────────────────────────────────

def achievement_badge(title: str, icon: str, description: str) -> None:
    st.markdown(f"""
    <div class="cs-card" style="display:flex;align-items:center;gap:16px;padding:12px 20px">
      <div style="font-size:36px">{icon}</div>
      <div>
        <div style="font-weight:600;color:#E2E8F0">{title}</div>
        <div style="color:#A0AEC0;font-size:13px">{description}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 85: return COLOR_SUCCESS
    if score >= 70: return COLOR_PRIMARY
    if score >= 50: return COLOR_WARNING
    return COLOR_ERROR


def _score_label(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 75: return "Good"
    if score >= 60: return "Average"
    if score >= 40: return "Below Average"
    return "Poor"


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))