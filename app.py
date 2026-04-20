"""
CodeSense - Main Application
Professional AI-Powered Code Quality Analyzer with modern UI/UX.
Run with: streamlit run app.py
"""

import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

# ── Page config must be FIRST Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="CodeSense – AI Code Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Third-party imports ───────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Internal imports after page config ───────────────────────────────────────
from analyzer import StaticAnalyzer
from dsa_detector import DSADetector
from features import FeatureExtractor
from train_model import QualityPredictor, calculate_contextual_adjustments, score_to_grade, score_to_label
from student_feedback import FeedbackEngine
from syntax_analyzer import SyntaxAnalyzer
from semantic_analyzer import SemanticAnalyzer
from context_engine import ContextEngine
from code_fixer import CodeFixer
from db import Database
from auth import Auth
from cache import get_cache, AnalysisCache
from utils import (
    detect_language_from_code, results_to_markdown,
    results_to_json, Timer, fetch_github_file,
)
from validators import validate_code, validate_language, sanitize_code
from ui_components import (
    inject_css, score_card, progress_bar, issue_card,
    algo_card, ds_chip, stat_row, achievement_badge,
    loading_spinner, _score_color,
)
from constants import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    SUPPORTED_LANGUAGES,
)
from logger import get_logger

logger = get_logger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return Database()

@st.cache_resource
def get_auth():
    return Auth(get_db())

@st.cache_resource
def get_predictor():
    p = QualityPredictor()
    p.ensure_model()
    return p

@st.cache_resource
def get_engines():
    return {
        "static":   StaticAnalyzer(),
        "dsa":      DSADetector(),
        "features": FeatureExtractor(),
        "syntax":   SyntaxAnalyzer(),
        "semantic": SemanticAnalyzer(),
        "context":  ContextEngine(),
        "fixer":    CodeFixer(),
        "feedback": FeedbackEngine(),
    }

cache: AnalysisCache = get_cache()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "user":           None,
        "token":          None,
        "page":           "dashboard",
        "last_result":    None,
        "last_code":      "",
        "analysis_level": "intermediate",
        "dark_mode":      True,
        "otp_email":      None,
        "otp_sent":       False,
        "auth_tab":       "login",
        # Input persistence across reruns
        "input_code":     "",
        "input_language": "python",
        "input_filename": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _is_logged_in() -> bool:
    return st.session_state.get("user") is not None


def _current_user() -> Optional[Dict]:
    return st.session_state.get("user")


def _go(page: str):
    st.session_state.page = page
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(code: str, language: str, filename: str = "",
                 level: str = "intermediate") -> Dict[str, Any]:
    """
    Full analysis pipeline. Returns a structured result dict.
    The ML model score IS the primary quality score.
    """
    engines   = get_engines()
    predictor = get_predictor()
    db        = get_db()
    user      = _current_user()

    # ── Cache check ──────────────────────────────────────────────────────────
    cache_key = cache.make_key(code, language, level)
    cached    = cache.get(cache_key)
    if cached:
        logger.info("Cache hit for analysis.")
        return cached

    with Timer() as t:
        # 1. Syntax
        syntax = engines["syntax"].analyze(code, language)

        # 2. Semantic
        semantic = engines["semantic"].analyze(code, language)

        # 3. Static analysis
        analysis = engines["static"].analyze(code, language)

        # 4. DSA
        dsa = engines["dsa"].detect(code, language)

        # 5. Context
        context = engines["context"].analyze(code, language, filename, analysis)

        # 6. Features
        feat_dict  = engines["features"].extract(code, language, analysis, dsa)
        feat_array = engines["features"].to_array(feat_dict)

        # 7. ML prediction (PRIMARY score)
        ctx_adj = calculate_contextual_adjustments(analysis, dsa, language)
        ml_score, confidence = predictor.predict(feat_array, ctx_adj)

        # 8. Grade
        grade = score_to_grade(ml_score)
        label = score_to_label(ml_score)

        # 9. Auto-fixes
        fixes = engines["fixer"].suggest_fixes(
            code, language,
            semantic.get("issues", []),
            analysis.get("security", {}).get("findings", []),
        )

        # 10. Feedback
        prev_score = None
        if user:
            stats = db.get_analysis_stats(user["id"])
            prev_score = stats.get("avg_score")

        feedback = engines["feedback"].generate(
            code=code, language=language, score=ml_score, grade=grade,
            analysis=analysis, dsa=dsa, syntax=syntax, semantic=semantic,
            fixes=fixes, context=context, level=level,
            previous_score=prev_score,
        )

    result = {
        "score":       ml_score,
        "ml_score":    ml_score,
        "confidence":  confidence,
        "grade":       grade,
        "label":       label,
        "language":    language,
        "filename":    filename,
        "features":    feat_dict,
        "syntax":      syntax,
        "semantic":    semantic,
        "analysis":    analysis,
        "dsa":         dsa,
        "context":     context,
        "fixes":       fixes,
        "feedback":    feedback,
        "processing_ms": t.elapsed_ms,
        "analyzed_at": datetime.utcnow().isoformat(),
    }

    # ── Persist to DB ────────────────────────────────────────────────────────
    if user:
        aid = db.save_analysis(
            user_id=user["id"], language=language, filename=filename,
            code=code, score=ml_score, grade=grade, confidence=confidence,
            ml_score=ml_score, features=feat_dict,
            results={k: v for k, v in result.items() if k != "features"},
            analysis_level=level, processing_ms=t.elapsed_ms,
        )
        _check_achievements(user["id"], ml_score, db)
        result["analysis_id"] = aid

    # ── Cache store ──────────────────────────────────────────────────────────
    cache.set(cache_key, result)
    return result


def _check_achievements(user_id: int, score: float, db: Database):
    db_inst = db
    stats   = db_inst.get_analysis_stats(user_id)
    total   = stats.get("total", 0)
    avg     = stats.get("avg_score", 0) or 0

    achv = [
        (total >= 1,   "first_analysis",  "First Analysis",    "Completed your first code analysis!",    "🎯"),
        (total >= 10,  "ten_analyses",    "Dedicated Coder",   "Completed 10 code analyses!",            "📊"),
        (total >= 50,  "fifty_analyses",  "Analysis Master",   "Completed 50 code analyses!",            "🏆"),
        (score >= 90,  "high_score",      "Code Excellence",   "Scored 90+ on a single analysis!",       "⭐"),
        (avg >= 80,    "consistent",      "Consistent Quality","Maintained 80+ average score!",           "💎"),
        (score == 100, "perfect",         "Perfection",        "Achieved a perfect score of 100!",        "🌟"),
    ]
    for condition, key, title, desc, icon in achv:
        if condition and not db_inst.has_achievement(user_id, key):
            db_inst.award_achievement(user_id, key, title, desc, icon)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────────────────────────────────────

def page_auth():
    inject_css()
    st.markdown("""
    <div style="max-width:480px;margin:40px auto">
      <div style="text-align:center;margin-bottom:32px">
        <div style="font-size:48px">🧠</div>
        <h1 style="margin:8px 0;font-size:32px">CodeSense</h1>
        <p style="color:#757575">AI-Powered Code Quality Analyzer</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    auth_obj = get_auth()
    tab_login, tab_register = st.tabs(["🔑 Sign In", "✨ Create Account"])

    with tab_login:
        _login_form(auth_obj)

    with tab_register:
        _register_form(auth_obj)


def _login_form(auth_obj: Auth):
    with st.form("login_form"):
        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submit   = st.form_submit_button("🔑 Sign In", use_container_width=True)

    if submit:
        ok, msg, user = auth_obj.login(email, password)
        if ok:
            st.session_state.user  = user
            st.session_state.token = auth_obj.create_session(user["id"])
            st.success("✅ " + msg)
            time.sleep(0.5)
            _go("dashboard")
        else:
            st.error("❌ " + msg)

    st.markdown("---")
    st.markdown("**Demo Account** (no email needed):")
    if st.button("🚀 Try Demo", use_container_width=True):
        _create_demo_user(auth_obj)


def _create_demo_user(auth_obj: Auth):
    db         = get_db()
    demo_email = "demo@codesense.ai"
    demo_pass  = "Demo@1234!"

    user = db.get_user_by_email(demo_email)
    if not user:
        # Register creates + auto-verifies the user
        ok, msg, uid = auth_obj.register("demo_user", demo_email, demo_pass, "Demo User")
        if not ok and "already" not in msg.lower():
            st.error(f"Could not create demo account: {msg}")
            return
        user = db.get_user_by_email(demo_email)

    if not user:
        st.error("Demo account setup failed. Please register manually.")
        return

    # Make sure it is verified (safe to call even if already verified)
    db.verify_user(user["id"])

    ok, msg, logged_in = auth_obj.login(demo_email, demo_pass)
    if ok and logged_in:
        st.session_state.user  = logged_in
        st.session_state.token = auth_obj.create_session(logged_in["id"])
        _go("dashboard")
    else:
        st.error(f"Demo login failed: {msg}")


def _register_form(auth_obj: Auth):
    with st.form("register_form"):
        full_name = st.text_input("Full Name")
        username  = st.text_input("Username")
        email     = st.text_input("Email")
        password  = st.text_input("Password", type="password",
                                   help="Min 8 chars, uppercase, lowercase, digit, special char")
        submit    = st.form_submit_button("✨ Create Account", use_container_width=True)

    if submit:
        ok, msg, uid = auth_obj.register(username, email, password, full_name)
        if ok:
            st.success("✅ Account created! You can now sign in.")
        else:
            st.error("❌ " + msg)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    user = _current_user()
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0;border-bottom:1px solid #2D2D3F;margin-bottom:16px">
          <div style="font-size:22px;font-weight:700;color:#1E88E5">🧠 CodeSense</div>
          <div style="font-size:11px;color:#757575">AI Code Quality Analyzer v2.0</div>
        </div>
        """, unsafe_allow_html=True)

        if user:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:12px;
                        background:#252535;border-radius:8px;margin-bottom:20px">
              <div style="font-size:28px">👤</div>
              <div>
                <div style="font-weight:600;color:#E2E8F0">{user.get('full_name') or user['username']}</div>
                <div style="font-size:12px;color:#757575">{user['email']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        pages = [
            ("🏠", "Dashboard",     "dashboard"),
            ("⚡", "Analyze",       "analyze"),
            ("📈", "Progress",      "progress"),
            ("🏆", "Achievements",  "achievements"),
            ("⚙️",  "Settings",     "settings"),
        ]
        for icon, label, page_key in pages:
            active = st.session_state.page == page_key
            bg     = "#1E88E5" if active else "transparent"
            color  = "#fff"    if active else "#A0AEC0"
            if st.sidebar.button(
                f"{icon}  {label}",
                key=f"nav_{page_key}",
                use_container_width=True,
            ):
                _go(page_key)

        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Sign Out", use_container_width=True):
            if st.session_state.get("token"):
                get_auth().logout(st.session_state.token)
            st.session_state.user  = None
            st.session_state.token = None
            _go("auth")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard():
    user  = _current_user()
    db    = get_db()
    name  = user.get("full_name") or user["username"]
    hour  = datetime.now().hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    st.markdown(f"## {greeting}, {name}! 👋")
    st.markdown("Here's your coding activity at a glance.")

    stats    = db.get_analysis_stats(user["id"])
    analyses = db.get_user_analyses(user["id"], limit=100)

    total   = stats.get("total", 0) or 0
    avg_sc  = round(stats.get("avg_score") or 0, 1)
    max_sc  = round(stats.get("max_score") or 0, 1)
    impr    = stats.get("recent_improvement", 0) or 0

    stat_row([
        {"label": "Total Analyses", "value": total,   "icon": "📊", "color": COLOR_PRIMARY},
        {"label": "Average Score",  "value": avg_sc,  "icon": "⭐", "color": COLOR_SUCCESS,
         "delta": impr},
        {"label": "Best Score",     "value": max_sc,  "icon": "🏆", "color": COLOR_WARNING},
        {"label": "Improvement",    "value": f"{'+' if impr>=0 else ''}{impr:.1f}",
         "icon": "📈", "color": COLOR_SUCCESS if impr >= 0 else COLOR_ERROR},
    ])

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📊 Score History")
        if analyses:
            df = pd.DataFrame([
                {"Date": a["created_at"][:10], "Score": a["score"],
                 "Language": a["language"].upper(), "Grade": a["grade"]}
                for a in analyses[:30]
            ])
            fig = px.line(
                df, x="Date", y="Score", color="Language",
                markers=True,
                color_discrete_map={"PYTHON": "#1E88E5", "JAVA": "#E53935", "CPP": "#43A047"},
            )
            fig.update_layout(
                paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
                font_color="#A0AEC0", margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(bgcolor="#252535"),
                yaxis=dict(range=[0, 100], gridcolor="#2D2D3F"),
                xaxis=dict(gridcolor="#2D2D3F"),
            )
            fig.add_hrect(y0=90, y1=100, fillcolor=COLOR_SUCCESS, opacity=0.07, line_width=0)
            fig.add_hrect(y0=75, y1=90, fillcolor=COLOR_PRIMARY,  opacity=0.05, line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analyses yet. Analyze some code to see your history!")

    with col2:
        st.markdown("### 🌐 Languages Used")
        if analyses:
            lang_counts = {}
            for a in analyses:
                l = a["language"].upper()
                lang_counts[l] = lang_counts.get(l, 0) + 1
            fig2 = go.Figure(go.Pie(
                labels=list(lang_counts.keys()),
                values=list(lang_counts.values()),
                hole=0.5,
                marker_colors=["#1E88E5", "#E53935", "#43A047"],
            ))
            fig2.update_layout(
                paper_bgcolor="#1E1E2E", font_color="#A0AEC0",
                margin=dict(l=0, r=0, t=20, b=0),
                showlegend=True,
                legend=dict(bgcolor="#252535"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📋 Recent Analyses")
    if analyses:
        for a in analyses[:5]:
            sc    = a["score"]
            color = _score_color(sc)
            st.markdown(f"""
            <div class="cs-card" style="display:flex;align-items:center;
                 justify-content:space-between;padding:12px 20px">
              <div>
                <span style="color:#E2E8F0;font-weight:500">
                  {a['filename'] or 'Pasted Code'}
                </span>
                <span style="color:#757575;font-size:12px;margin-left:8px">
                  {a['language'].upper()} · {a['created_at'][:16].replace('T',' ')}
                </span>
              </div>
              <div style="display:flex;align-items:center;gap:12px">
                <span style="font-size:20px;font-weight:700;color:{color}">{sc:.0f}</span>
                <span style="color:{color};font-weight:600">{a['grade']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent analyses yet.")

    st.markdown("---")
    if st.button("⚡ Analyze New Code", use_container_width=True, type="primary"):
        _go("analyze")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_analyze():
    st.markdown("## ⚡ Analyze Your Code")
    st.markdown("Paste your code, upload a file, or fetch from GitHub.")

    # ── Persist code/language/filename across reruns via session_state ────────
    # Streamlit reruns the whole script on every interaction, so local variables
    # assigned inside tab blocks are lost. We use session_state as the single
    # source of truth for the current code being analyzed.
    if "input_code"     not in st.session_state: st.session_state.input_code     = ""
    if "input_language" not in st.session_state: st.session_state.input_language = "python"
    if "input_filename" not in st.session_state: st.session_state.input_filename = ""

    tab_paste, tab_upload, tab_github = st.tabs(["📝 Paste Code", "📁 Upload File", "🐙 GitHub"])

    with tab_paste:
        lang_col, level_col = st.columns([2, 2])
        with lang_col:
            paste_lang = st.selectbox("Language", SUPPORTED_LANGUAGES,
                                      format_func=str.upper, key="paste_lang_sel")
        with level_col:
            st.session_state.analysis_level = st.selectbox(
                "Feedback Level",
                ["beginner", "intermediate", "advanced"],
                index=1,
            )
        pasted = st.text_area(
            "Your Code",
            height=320,
            placeholder="# Paste your Python, Java, or C++ code here...\n\ndef hello():\n    print('Hello, World!')",
            label_visibility="collapsed",
            key="paste_textarea",
        )
        if pasted:
            st.session_state.input_code     = pasted
            st.session_state.input_language = paste_lang
            st.session_state.input_filename = "pasted_code"

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop your file here",
            type=["py", "java", "cpp", "cc", "cxx", "c"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw = uploaded.read().decode("utf-8", errors="replace")
            fname = uploaded.name
            ext   = "." + fname.rsplit(".", 1)[-1].lower()
            from constants import LANGUAGE_EXTENSIONS as _LE
            lang  = _LE.get(ext, detect_language_from_code(raw))
            st.session_state.input_code     = raw
            st.session_state.input_language = lang
            st.session_state.input_filename = fname
            st.success(f"✅ Loaded **{fname}** ({len(raw.splitlines())} lines, {lang.upper()})")
            with st.expander("Preview"):
                st.code(raw[:2000] + ("..." if len(raw) > 2000 else ""), language=lang)

    with tab_github:
        gh_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/user/repo/blob/main/file.py",
            key="gh_url_input",
        )
        if st.button("📥 Fetch from GitHub", key="gh_fetch_btn"):
            if not gh_url.strip():
                st.error("❌ Please enter a GitHub URL first.")
            else:
                with st.spinner("Fetching from GitHub..."):
                    try:
                        fetched_code, fetched_lang = fetch_github_file(gh_url.strip())
                        fetched_name = gh_url.strip().split("/")[-1]
                        # Persist to session state immediately
                        st.session_state.input_code     = fetched_code
                        st.session_state.input_language = fetched_lang
                        st.session_state.input_filename = fetched_name
                        st.success(f"✅ Fetched **{fetched_name}** ({len(fetched_code.splitlines())} lines, {fetched_lang.upper()})")
                        with st.expander("Preview", expanded=True):
                            st.code(fetched_code[:2000], language=fetched_lang)
                    except Exception as exc:
                        st.error(f"❌ {exc}")

    # ── Read from session state (survives reruns) ─────────────────────────────
    code     = st.session_state.input_code
    language = st.session_state.input_language
    filename = st.session_state.input_filename

    # ── Options ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Advanced Options"):
        col_a, col_b = st.columns(2)
        with col_a:
            auto_detect = st.checkbox("Auto-detect language", value=True)
        with col_b:
            show_fixes = st.checkbox("Show auto-fix suggestions", value=True)

    if auto_detect and code:
        detected = detect_language_from_code(code)
        if detected != language:
            st.info(f"🔍 Auto-detected: **{detected.upper()}**")
            st.session_state.input_language = detected
            language = detected

    # Show what's ready to analyze
    if code:
        st.caption(f"📄 Ready: **{filename or 'pasted code'}** · {language.upper()} · {len(code.splitlines())} lines")

    # ── Analyze Button ────────────────────────────────────────────────────────
    analyze_clicked = st.button(
        "🚀 Analyze Code",
        use_container_width=True,
        type="primary",
        disabled=not bool(code.strip()),
    )

    if analyze_clicked:
        code = sanitize_code(code)
        valid, err = validate_code(code)
        if not valid:
            st.error(f"❌ {err}")
            return

        with st.spinner("🔍 Running full analysis pipeline..."):
            try:
                result = run_analysis(
                    code=code, language=language, filename=filename,
                    level=st.session_state.analysis_level,
                )
                st.session_state.last_result = result
                st.session_state.last_code   = code
            except Exception as exc:
                st.error(f"❌ Analysis failed: {exc}")
                logger.error("Analysis error: %s", traceback.format_exc())
                return

        _render_results(result, code, show_fixes)

    elif st.session_state.get("last_result"):
        with st.expander("📊 Last Analysis Results", expanded=True):
            _render_results(
                st.session_state.last_result,
                st.session_state.get("last_code", ""),
                True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(result: Dict, code: str, show_fixes: bool = True):
    score    = result["score"]
    grade    = result["grade"]
    label    = result["label"]
    feedback = result.get("feedback", {})
    dsa      = result.get("dsa", {})
    syntax   = result.get("syntax", {})
    semantic = result.get("semantic", {})
    analysis = result.get("analysis", {})
    context  = result.get("context", {})
    fixes    = result.get("fixes", [])
    ms       = result.get("processing_ms", 0)

    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    st.caption(f"Analyzed in {ms}ms · Language: {result['language'].upper()}")

    # ── Score + grade ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        score_card(
            score=score, grade=grade, confidence=result.get("confidence", 3),
            label=label,
        )
    with col2:
        st.markdown(f"### {feedback.get('opening', '')}")

        summ = feedback.get("summary", {})
        col_e, col_w, col_i, col_p = st.columns(4)
        col_e.metric("❌ Errors",    summ.get("errors",   0))
        col_w.metric("⚠️ Warnings", summ.get("warnings", 0))
        col_i.metric("ℹ️ Info",     summ.get("infos",    0))
        col_p.metric("✅ Strengths", summ.get("positives",0))

        # Encouragement
        enc = feedback.get("encouragement", "")
        if enc:
            st.info(enc)

    # ── Context notes ─────────────────────────────────────────────────────────
    ctx_notes = context.get("intent_notes", [])
    if ctx_notes:
        with st.expander("🔍 Context Notes", expanded=False):
            for note in ctx_notes:
                st.info(note)

    # ── Tabs for detail ───────────────────────────────────────────────────────
    tabs = st.tabs([
        "💪 Strengths",
        "🔴 Issues",
        "🧠 DSA",
        "🔒 Security",
        "📊 Metrics",
        "🔧 Fixes",
        "📚 Learning Path",
        "📤 Export",
    ])

    # Tab 0: Strengths
    with tabs[0]:
        strengths = feedback.get("strengths", [])
        if strengths:
            for s in strengths:
                st.markdown(f"""
                <div class="cs-card sev-positive" style="margin-bottom:8px">
                  <span style="color:#43A047">✅</span>
                  <span style="color:#E2E8F0;margin-left:8px">{s}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific strengths identified. Keep working on your code quality!")

    # Tab 1: Issues
    with tabs[1]:
        items = feedback.get("items", [])
        non_positive = [i for i in items if i["severity"] != "positive"]
        if non_positive:
            for item in non_positive:
                issue_card(item)
        else:
            st.success("🎉 No issues found! Your code is clean.")

    # Tab 2: DSA
    with tabs[2]:
        algos = dsa.get("algorithms", [])
        ds_list = dsa.get("data_structures", [])
        dsa_summary = dsa.get("summary", {})

        if algos or ds_list:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("### 🧮 Algorithms Detected")
                if algos:
                    for algo in algos:
                        algo_card(algo)
                else:
                    st.info("No specific algorithms detected.")
            with col_b:
                st.markdown("### 📦 Data Structures")
                if ds_list:
                    for ds in ds_list:
                        ds_chip(ds)
                    st.markdown("")
                    # Complexity Score
                    cx_score = dsa_summary.get("complexity_score", 0)
                    st.markdown(f"**DSA Complexity Score:** {cx_score:.1f}/100")
                    progress_bar(cx_score, color=COLOR_PRIMARY)
                else:
                    st.info("No data structures identified.")
        else:
            st.info("No DSA patterns detected in this code.")

    # Tab 3: Security
    with tabs[3]:
        findings = analysis.get("security", {}).get("findings", [])
        if findings:
            from constants import SEVERITY_CRITICAL, SEVERITY_HIGH
            critical = [f for f in findings if f["severity"] == SEVERITY_CRITICAL]
            high     = [f for f in findings if f["severity"] == SEVERITY_HIGH]
            rest     = [f for f in findings if f["severity"] not in (SEVERITY_CRITICAL, SEVERITY_HIGH)]

            for grp, label in [(critical, "🔴 Critical"), (high, "🟠 High"), (rest, "🟡 Medium/Low")]:
                if grp:
                    st.markdown(f"#### {label}")
                    for f in grp:
                        issue_card({
                            "severity": "error" if f["severity"] in (SEVERITY_CRITICAL, SEVERITY_HIGH) else "warning",
                            "title":    f"[{f['severity']}] {f['type'].replace('_',' ').title()} — Line {f['line']}",
                            "message":  f["description"],
                            "line":     f["line"],
                            "code_before": f.get("code_snippet",""),
                        })
        else:
            st.success("🔒 No security vulnerabilities found!")

    # Tab 4: Metrics
    with tabs[4]:
        metrics  = analysis.get("metrics", {})
        cx       = analysis.get("complexity", {})
        style    = analysis.get("style", {})
        doc      = analysis.get("documentation", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📏 Code Metrics")
            for label, val in [
                ("Lines of Code",      metrics.get("code_lines", 0)),
                ("Blank Lines",        metrics.get("blank_lines", 0)),
                ("Comment Lines",      metrics.get("comment_lines", 0)),
                ("Comment Ratio",      f"{metrics.get('comment_ratio',0)*100:.1f}%"),
                ("Avg Line Length",    metrics.get("avg_line_len", 0)),
                ("Max Line Length",    metrics.get("max_line_len", 0)),
            ]:
                st.markdown(f"<div class='metric-pill'>{label}: <b>{val}</b></div>",
                            unsafe_allow_html=True)

            st.markdown("#### 🔄 Complexity")
            avg_cx = cx.get("avg_complexity", 0)
            max_cx = cx.get("max_complexity", 0)
            st.markdown(f"<div class='metric-pill'>Avg Cyclomatic: <b>{avg_cx:.1f}</b></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='metric-pill'>Max Cyclomatic: <b>{max_cx:.1f}</b></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='metric-pill'>Max Nesting: <b>{cx.get('max_nesting',0)}</b></div>",
                        unsafe_allow_html=True)
            progress_bar(min(100, avg_cx * 10), label="Complexity Score (lower=better)",
                         color=COLOR_ERROR if avg_cx > 10 else COLOR_SUCCESS)

        with col2:
            st.markdown("#### 🎨 Style")
            for label, val in [
                ("Magic Numbers",      style.get("magic_number_count", 0)),
                ("Long Lines",         style.get("long_line_count", 0)),
                ("Naming Score",       f"{style.get('naming_score',100):.0f}/100"),
                ("Quote Consistency",  style.get("quote_consistency","n/a")),
            ]:
                st.markdown(f"<div class='metric-pill'>{label}: <b>{val}</b></div>",
                            unsafe_allow_html=True)

            st.markdown("#### 📖 Documentation")
            ratio = doc.get("docstring_ratio", doc.get("comment_ratio", 0))
            st.markdown(f"<div class='metric-pill'>Coverage: <b>{ratio*100:.1f}%</b></div>",
                        unsafe_allow_html=True)
            progress_bar(ratio * 100, label="Documentation Coverage",
                         color=COLOR_SUCCESS if ratio > 0.3 else COLOR_WARNING)

        # Feature importance mini chart
        st.markdown("#### 🎯 ML Feature Contributions")
        feat = result.get("features", {})
        if feat:
            top_keys = [
                "avg_cyclomatic_complexity", "security_issue_count",
                "docstring_ratio", "code_duplication_score",
                "naming_consistency_score", "code_smell_count",
            ]
            labels = [k.replace("_", " ").title() for k in top_keys]
            vals   = [feat.get(k, 0) for k in top_keys]
            fig = go.Figure(go.Bar(
                x=labels, y=vals,
                marker_color=[COLOR_PRIMARY]*len(vals),
            ))
            fig.update_layout(
                paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
                font_color="#A0AEC0", margin=dict(l=0, r=0, t=10, b=0),
                height=220, xaxis_tickangle=-30,
                yaxis=dict(gridcolor="#2D2D3F"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 5: Fixes
    with tabs[5]:
        if not show_fixes:
            st.info("Enable 'Show auto-fix suggestions' in Advanced Options.")
        elif fixes:
            safe   = [f for f in fixes if f.get("is_safe")]
            unsafe = [f for f in fixes if not f.get("is_safe")]

            if safe:
                st.markdown("#### ✅ Safe to Apply")
                for fix in safe:
                    with st.expander(f"Line {fix['line']} — {fix['description']}"):
                        col_b, col_a = st.columns(2)
                        with col_b:
                            st.markdown("**Before:**")
                            st.code(fix["before"], language=result["language"])
                        with col_a:
                            st.markdown("**After:**")
                            st.code(fix["after"] or "(remove line)", language=result["language"])
                        st.info(fix["explanation"])
                        st.caption(f"Confidence: {fix['confidence']*100:.0f}%")

            if unsafe:
                st.markdown("#### 🔍 Requires Review")
                for fix in unsafe:
                    with st.expander(f"Line {fix['line']} — {fix['description']}"):
                        col_b, col_a = st.columns(2)
                        with col_b:
                            st.markdown("**Before:**")
                            st.code(fix["before"], language=result["language"])
                        with col_a:
                            st.markdown("**After:**")
                            st.code(fix["after"] or "(see explanation)", language=result["language"])
                        st.warning(fix["explanation"])
        else:
            st.success("🎉 No auto-fixes needed — code is clean!")

    # Tab 6: Learning Path
    with tabs[6]:
        path       = feedback.get("learning_path", [])
        next_steps = feedback.get("next_steps", [])

        if next_steps:
            st.markdown("#### 📋 Immediate Next Steps")
            for step in next_steps:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:12px">
                  <div class="step-circle">→</div>
                  <div style="color:#E2E8F0;padding-top:4px">{step}</div>
                </div>
                """, unsafe_allow_html=True)

        if path:
            st.markdown("#### 📚 Learning Path")
            for item in path:
                r = item.get("resource") or {}
                url = r.get("url", "#")
                st.markdown(f"""
                <div class="cs-card" style="display:flex;align-items:flex-start;gap:16px">
                  <div class="step-circle">{item['step']}</div>
                  <div>
                    <div style="font-weight:600;color:#E2E8F0">{item['title']}</div>
                    <div style="color:#A0AEC0;font-size:13px;margin-top:4px">{item['description']}</div>
                    {"<a href='" + url + "' target='_blank' style='color:#1E88E5;font-size:12px'>📖 " + r.get('title','') + "</a>" if r else ""}
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Tab 7: Export
    with tabs[7]:
        st.markdown("#### 📤 Export Your Results")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            md_report = results_to_markdown(result)
            st.download_button(
                "📄 Download Markdown",
                data=md_report,
                file_name=f"codesense_report_{result['language']}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_b:
            json_report = results_to_json({
                k: v for k, v in result.items()
                if k not in ("features",)
            })
            st.download_button(
                "📊 Download JSON",
                data=json_report,
                file_name=f"codesense_report_{result['language']}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_c:
            st.download_button(
                "💾 Download Code",
                data=code,
                file_name=result.get("filename") or f"analyzed.{result['language']}",
                mime="text/plain",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_progress():
    user     = _current_user()
    db       = get_db()
    analyses = db.get_user_analyses(user["id"], limit=100)
    stats    = db.get_analysis_stats(user["id"])

    st.markdown("## 📈 Your Progress")

    if not analyses:
        st.info("No analyses yet. Start by analyzing some code!")
        if st.button("⚡ Analyze Code"):
            _go("analyze")
        return

    total = stats.get("total", 0)
    avg   = round(stats.get("avg_score") or 0, 1)
    best  = round(stats.get("max_score") or 0, 1)
    impr  = round(stats.get("recent_improvement") or 0, 1)

    stat_row([
        {"label": "Total",       "value": total, "icon": "📊", "color": COLOR_PRIMARY},
        {"label": "Average",     "value": avg,   "icon": "⭐", "color": COLOR_SUCCESS},
        {"label": "Best",        "value": best,  "icon": "🏆", "color": COLOR_WARNING},
        {"label": "Improvement", "value": f"{'+' if impr>=0 else ''}{impr}", "icon": "📈",
         "color": COLOR_SUCCESS if impr >= 0 else COLOR_ERROR},
    ])

    # ── Filters ──────────────────────────────────────────────────────────────
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        lang_filter = st.selectbox("Filter by Language",
                                   ["All"] + [l.upper() for l in SUPPORTED_LANGUAGES])
    with f_col2:
        date_filter = st.selectbox("Time Range",
                                   ["All time", "Last 30 days", "Last 7 days"])

    import pandas as pd
    from datetime import timedelta

    df = pd.DataFrame([
        {
            "Date":     a["created_at"][:10],
            "Score":    a["score"],
            "Grade":    a["grade"],
            "Language": a["language"].upper(),
            "File":     a["filename"] or "Pasted",
            "Time (ms)": a.get("processing_ms", 0),
        }
        for a in analyses
    ])

    df["Date"] = pd.to_datetime(df["Date"])
    if lang_filter != "All":
        df = df[df["Language"] == lang_filter]
    if date_filter == "Last 30 days":
        df = df[df["Date"] >= datetime.now() - timedelta(days=30)]
    elif date_filter == "Last 7 days":
        df = df[df["Date"] >= datetime.now() - timedelta(days=7)]

    # Score trend
    st.markdown("### 📊 Score Trend")
    if not df.empty:
        fig = px.scatter(df, x="Date", y="Score", color="Language",
                         trendline="lowess", size_max=10,
                         color_discrete_map={"PYTHON": "#1E88E5", "JAVA": "#E53935", "CPP": "#43A047"})
        fig.add_hline(y=75, line_dash="dash", line_color=COLOR_WARNING,
                      annotation_text="Good (75)")
        fig.add_hline(y=90, line_dash="dash", line_color=COLOR_SUCCESS,
                      annotation_text="Excellent (90)")
        fig.update_layout(
            paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
            font_color="#A0AEC0", margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(range=[0, 100], gridcolor="#2D2D3F"),
            xaxis=dict(gridcolor="#2D2D3F"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Grade distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎓 Grade Distribution")
        if not df.empty:
            grade_counts = df["Grade"].value_counts()
            fig2 = go.Figure(go.Bar(
                x=grade_counts.index.tolist(),
                y=grade_counts.values.tolist(),
                marker_color=COLOR_PRIMARY,
            ))
            fig2.update_layout(
                paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
                font_color="#A0AEC0", margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(gridcolor="#2D2D3F"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### 📋 Recent History")
        if not df.empty:
            show_df = df[["Date", "Language", "Score", "Grade", "File"]].head(10)
            show_df["Date"] = show_df["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(show_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ACHIEVEMENTS PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_achievements():
    user = _current_user()
    db   = get_db()

    st.markdown("## 🏆 Achievements")

    all_achievements = [
        ("first_analysis",  "🎯", "First Analysis",    "Completed your first code analysis"),
        ("ten_analyses",    "📊", "Dedicated Coder",   "Completed 10 analyses"),
        ("fifty_analyses",  "🏆", "Analysis Master",   "Completed 50 analyses"),
        ("high_score",      "⭐", "Code Excellence",   "Scored 90+ on a single analysis"),
        ("consistent",      "💎", "Consistent Quality","Maintained 80+ average score"),
        ("perfect",         "🌟", "Perfection",        "Achieved a perfect score of 100"),
    ]

    earned = {a["achievement_key"] for a in db.get_user_achievements(user["id"])}
    earned_count = len(earned)
    total_count  = len(all_achievements)

    st.markdown(f"**{earned_count}/{total_count} achievements earned**")
    progress_bar(earned_count, total_count, label="")

    cols = st.columns(2)
    for idx, (key, icon, title, desc) in enumerate(all_achievements):
        with cols[idx % 2]:
            if key in earned:
                achievement_badge(title, icon, desc)
            else:
                st.markdown(f"""
                <div class="cs-card" style="opacity:0.4;display:flex;align-items:center;gap:16px;padding:12px 20px">
                  <div style="font-size:36px;filter:grayscale(1)">🔒</div>
                  <div>
                    <div style="font-weight:600;color:#A0AEC0">{title}</div>
                    <div style="color:#757575;font-size:13px">{desc}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_settings():
    user = _current_user()
    db   = get_db()

    st.markdown("## ⚙️ Settings")

    tab_profile, tab_analysis, tab_account = st.tabs(
        ["👤 Profile", "🔬 Analysis", "🔒 Account"]
    )

    with tab_profile:
        st.markdown("### Update Profile")
        with st.form("profile_form"):
            full_name = st.text_input("Full Name", value=user.get("full_name", ""))
            submitted = st.form_submit_button("💾 Save Changes")
        if submitted:
            db.update_user(user["id"], full_name=full_name)
            st.session_state.user["full_name"] = full_name
            st.success("✅ Profile updated!")

    with tab_analysis:
        st.markdown("### Analysis Preferences")
        level = st.select_slider(
            "Default Feedback Level",
            options=["beginner", "intermediate", "advanced"],
            value=st.session_state.analysis_level,
        )
        st.session_state.analysis_level = level
        st.markdown("#### Cache")
        stats = cache.stats()
        st.markdown(f"Memory: **{stats['memory']['size']}** items cached")
        st.markdown(f"Hit rate: **{stats['memory']['hit_rate']}**")
        if st.button("🗑️ Clear Cache"):
            cache.clear_all()
            st.success("Cache cleared!")

    with tab_account:
        st.markdown("### Account Information")
        st.markdown(f"**Username:** `{user['username']}`")
        st.markdown(f"**Email:** `{user['email']}`")
        st.markdown(f"**Member since:** {user.get('created_at','')[:10]}")
        st.markdown("---")

        if st.button("📤 Export My Data"):
            all_data = db.get_user_analyses(user["id"], limit=1000)
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(all_data, indent=2, default=str),
                file_name="codesense_my_data.json",
                mime="application/json",
            )

        st.markdown("---")
        st.markdown("#### Model Information")
        try:
            predictor = get_predictor()
            meta = predictor.get_model_meta()
            if meta:
                st.markdown(f"- **Trained:** {meta.get('trained_at','')[:10]}")
                st.markdown(f"- **R²:** {meta.get('r2','?')}")
                st.markdown(f"- **MAE:** {meta.get('mae','?')}")
                st.markdown(f"- **Samples:** {meta.get('n_samples','?')}")
        except Exception:
            st.info("Model metadata unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _init_session()
    inject_css()

    if not _is_logged_in():
        page_auth()
        return

    render_sidebar()

    page = st.session_state.page
    if   page == "dashboard":    page_dashboard()
    elif page == "analyze":      page_analyze()
    elif page == "progress":     page_progress()
    elif page == "achievements": page_achievements()
    elif page == "settings":     page_settings()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()