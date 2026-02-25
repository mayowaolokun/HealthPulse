import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Optional click-to-filter support
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

# Sparse support
try:
    import scipy.sparse as sp
except Exception:
    sp = None


# -----------------------------
# Page config + styling
# -----------------------------
st.set_page_config(page_title="HealthPulse Dashboard", layout="wide")

st.markdown(
    """
<style>
div[data-baseweb="tab-list"] { gap: 10px; }
button[data-baseweb="tab"] { font-size: 16px; padding: 10px 18px; }

.hp-card {
    border: 1px solid rgba(0,0,0,0.10);
    border-radius: 14px;
    padding: 16px;
    background: rgba(0,0,0,0.02);
}
.small-muted { opacity: 0.75; font-size: 12px; }
.kpi-wrap { margin-top: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# Consistent visuals everywhere
PLOTLY_TEMPLATE = "plotly_white"
PRIMARY_BAR_COLOR = "#66c2a5"  # consistent bar color everywhere
RISK_COLORS = {
    "Low": "#4caf50",
    "Medium": "#ff9800",
    "High": "#f44336",
}

DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# -----------------------------
# Paths (repo-safe)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts"

MODEL_CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
FRIENDLY_SCHEMA_PATH = ARTIFACT_DIR / "friendly_schema.json"
GLOBAL_IMP_PATH = ARTIFACT_DIR / "global_feature_importance.csv"
SHAP_IMP_PATH = ARTIFACT_DIR / "shap_feature_importance.csv"


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_local_model_pipeline():
    """Optional local pipeline to compute segment drivers.
    If missing, segment drivers will be disabled gracefully."""
    import joblib
    model_path = ARTIFACT_DIR / "model_pipeline.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


# -----------------------------
# Helpers
# -----------------------------
def ensure_friendly_columns(df: pd.DataFrame, friendly_features: List[str]) -> pd.DataFrame:
    """Keep only friendly features; add missing ones as np.nan."""
    out = df.copy()
    for c in friendly_features:
        if c not in out.columns:
            out[c] = np.nan
    out = out[friendly_features]
    out = out.replace({pd.NA: np.nan})
    return out

def safe_percent_table(s: pd.Series) -> pd.DataFrame:
    s = s.dropna().astype(str)
    counts = s.value_counts(dropna=False)
    pct = (counts / counts.sum() * 100).astype(float).round(2)
    tab = pd.DataFrame(
        {"category": counts.index.astype(str), "count": counts.values, "percent": pct.values}
    )
    tab["percent"] = pd.to_numeric(tab["percent"], errors="coerce").fillna(0.0)
    return tab

def percent_histogram(df: pd.DataFrame, col: str, bins: int = 20):
    fig = px.histogram(df, x=col, nbins=bins, histnorm="percent", template=PLOTLY_TEMPLATE)
    fig.update_traces(marker_color=PRIMARY_BAR_COLOR)
    fig.update_layout(
        xaxis_title=col.replace("_", " ").title(),
        yaxis_title="Percent (%)",
        margin=dict(l=10, r=10, t=10, b=10),
        bargap=0.05,
    )
    return fig

def categorical_percent_bar(df: pd.DataFrame, col: str, top_n: int = 15):
    """
    FIXED:
    - percent labels are now real strings like '12.50%' (no %{y:.2f}% literal)
    - better x-axis labeling + ordering
    - consistent colors
    """
    tab = safe_percent_table(df[col]).head(top_n)

    # Order day_of_week nicely if this is that column
    if col == "day_of_week":
        tab["category"] = pd.Categorical(tab["category"], categories=DOW_ORDER, ordered=True)
        tab = tab.sort_values("category")

    fig = px.bar(
        tab,
        x="category",
        y="percent",
        template=PLOTLY_TEMPLATE,
    )

    fig.update_traces(
        marker_color=PRIMARY_BAR_COLOR,
        text=[f"{v:.2f}%" for v in tab["percent"].astype(float).tolist()],
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        xaxis_title=col.replace("_", " ").title(),
        yaxis_title="Percent (%)",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    fig.update_xaxes(
        tickangle=-20,
        automargin=True,
        categoryorder="total descending" if col != "day_of_week" else "array",
    )

    maxv = float(tab["percent"].max()) if len(tab) else 0.0
    fig.update_yaxes(range=[0, max(5, maxv + 8)])  # room for labels above bars

    return fig

# NEW: normalize API base URL so users can paste base, /predict, or /health
def _normalize_api_base(api_url: str) -> str:
    u = (api_url or "").strip()
    if not u:
        return u
    u = u.rstrip("/")
    if u.endswith("/predict"):
        u = u[: -len("/predict")]
    if u.endswith("/health"):
        u = u[: -len("/health")]
    return u.rstrip("/")

def call_predict_api(api_url: str, records: List[Dict[str, Any]], timeout: int = 60) -> Dict[str, Any]:
    base = _normalize_api_base(api_url)
    r = requests.post(f"{base}/predict", json={"records": records}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def attach_predictions(df: pd.DataFrame, api_result: Dict[str, Any]) -> pd.DataFrame:
    preds = api_result.get("predictions", [])
    out = df.copy()
    out["no_show_probability"] = [p.get("no_show_probability", np.nan) for p in preds]
    out["predicted_label"] = [p.get("predicted_label", np.nan) for p in preds]
    if preds and isinstance(preds[0], dict) and "appointment_id" in preds[0]:
        out["appointment_id"] = [p.get("appointment_id", None) for p in preds]
    return out

def kpi_cards(total: int, high_risk: int, avg_risk: float,
              baseline_high_risk_rate: float, baseline_avg_risk: float,
              thr: float):
    high_rate = (high_risk / total) if total else 0.0
    d_high = (high_rate - baseline_high_risk_rate) * 100
    d_avg = (avg_risk - baseline_avg_risk) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{total:,}")
    c2.metric("High-Risk Rate", f"{high_rate*100:.2f}%", delta=f"{d_high:+.2f}%")
    c3.metric("Avg Risk", f"{avg_risk*100:.2f}%", delta=f"{d_avg:+.2f}%")
    c4.metric("Threshold", f"{thr:.2f}")

def make_heatmap(df: pd.DataFrame, row: str, col: str, value: str = "no_show_probability"):
    pivot = df.pivot_table(index=row, columns=col, values=value, aggfunc="mean").fillna(0)

    # order day_of_week columns
    if col == "day_of_week":
        pivot = pivot.reindex(columns=[c for c in DOW_ORDER if c in pivot.columns])

    fig = px.imshow(pivot, aspect="auto", origin="lower", template=PLOTLY_TEMPLATE)
    fig.update_layout(
        xaxis_title=col.replace("_", " ").title(),
        yaxis_title=row.replace("_", " ").title(),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig

def _get_transformer_cols(preprocess, name: str) -> List[str]:
    try:
        for tname, _trans, cols in preprocess.transformers_:
            if tname == name:
                return list(cols)
    except Exception:
        pass
    return []

def clean_for_local_pipeline(df_model: pd.DataFrame,
                             numeric_cols: List[str],
                             cat_cols: List[str]) -> pd.DataFrame:
    """Prevent pandas NAType from hitting sklearn; enforce safe types."""
    out = df_model.copy()
    out = out.replace({pd.NA: np.nan})

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64").fillna(0.0)

    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna("Unknown").replace("nan", "Unknown")

    out = out.replace({pd.NA: np.nan})
    return out

def compute_segment_drivers_local(pipeline,
                                 df_friendly: pd.DataFrame,
                                 model_features: List[str],
                                 friendly_cat: List[str]) -> Optional[pd.DataFrame]:
    """
    Segment-level drivers for Logistic Regression:
    mean(|coef * x|) over the filtered segment, aggregated back to parent columns.
    """
    if pipeline is None:
        return None
    if "preprocess" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        return None

    preprocess = pipeline.named_steps["preprocess"]
    lr = pipeline.named_steps["model"]
    if not hasattr(lr, "coef_"):
        return None

    df_model = df_friendly.copy().replace({pd.NA: np.nan})

    for c in model_features:
        if c not in df_model.columns:
            df_model[c] = np.nan
    df_model = df_model[model_features]

    num_cols = _get_transformer_cols(preprocess, "num")
    cat_cols = _get_transformer_cols(preprocess, "cat")
    df_model = clean_for_local_pipeline(df_model, numeric_cols=num_cols, cat_cols=cat_cols)

    Xt = preprocess.transform(df_model)
    coef = lr.coef_.ravel()

    if sp is not None and sp.issparse(Xt):
        contrib = Xt.multiply(coef)
        mean_abs = np.asarray(np.abs(contrib).mean(axis=0)).ravel()
    else:
        contrib = Xt * coef
        mean_abs = np.abs(contrib).mean(axis=0)

    transformed_names: List[str] = []
    transformed_names.extend(list(num_cols))
    try:
        ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
        cat_ohe_names = ohe.get_feature_names_out(cat_cols)
        transformed_names.extend(list(cat_ohe_names))
    except Exception:
        transformed_names.extend([f"cat_{i}" for i in range(max(0, len(mean_abs) - len(num_cols)))])

    imp = pd.DataFrame({"feature": transformed_names, "importance": mean_abs})
    imp = imp.sort_values("importance", ascending=False)

    def parent(f: str) -> str:
        for c in friendly_cat:
            if isinstance(f, str) and f.startswith(c + "_"):
                return c
        return str(f)

    imp["parent_feature"] = imp["feature"].map(parent)
    grouped = imp.groupby("parent_feature", as_index=False)["importance"].sum()
    grouped = grouped.sort_values("importance", ascending=False)
    return grouped

def risk_band(p: float, thr: float) -> str:
    if p >= max(thr, 0.60):
        return "High"
    if p >= thr:
        return "Medium"
    return "Low"


# -----------------------------
# Load configs + schema
# -----------------------------
st.title("HealthPulse — AI-Powered No-Show Intelligence")

if not MODEL_CONFIG_PATH.exists() or not FRIENDLY_SCHEMA_PATH.exists():
    st.error("Missing required files in artifacts/: model_config.json and friendly_schema.json")
    st.stop()

model_config = load_json(MODEL_CONFIG_PATH)
friendly_schema = load_json(FRIENDLY_SCHEMA_PATH)

MODEL_FEATURES = model_config["all_features"]
THRESHOLD_DEFAULT = float(model_config.get("chosen_threshold", 0.45))

FRIENDLY_NUM = friendly_schema["friendly_num_features"]
FRIENDLY_CAT = friendly_schema["friendly_cat_features"]
FRIENDLY_ALL = friendly_schema["friendly_all_features"]

pipeline_local = load_local_model_pipeline()

MODEL_TYPE = model_config.get("model_type", "logistic_regression")
TARGET = model_config.get("target", "is_no_show")


# -----------------------------
# Tabs
# -----------------------------
tab_home, tab_dash, tab_predict, tab_drivers, tab_reco = st.tabs(
    ["Home", "Dashboard", "Prediction", "Explainability", "Recommendations"]
)

# =========================================================
# HOME
# =========================================================
with tab_home:
    st.markdown(
        f"""
<div class="hp-card">
<h3 style="margin-top:0;">Project Overview</h3>

<b>Objective:</b> Reduce missed appointments by predicting <b>No-Show risk</b> early enough for targeted interventions
(reminders, rescheduling, transport support, and capacity planning).<br><br>

<b>Model in Production:</b> <code>{MODEL_TYPE}</code><br>
<b>Target:</b> <code>{TARGET}</code><br>
<b>Decision Threshold:</b> <code>{THRESHOLD_DEFAULT:.2f}</code><br><br>

<b>How HealthPulse is meant to be used (operations-first)</b>
<ul>
<li><b>Find</b> where risk concentrates (clinic, specialty, weekday, insurance).</li>
<li><b>Prioritize</b> outreach to the highest-risk appointments (limited staff time).</li>
<li><b>Intervene</b> with the lowest-friction option (SMS → call → reschedule/transport help).</li>
<li><b>Measure</b> impact using KPI deltas (segment vs baseline).</li>
</ul>

<span class="small-muted">
If you’re presenting this, the story is: “We can predict no-shows ahead of time and allocate interventions where they return the most value.”
</span>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.info(
            "Click-to-filter works if `streamlit-plotly-events` is installed. "
            f"Status: {'✅ Enabled' if HAS_PLOTLY_EVENTS else '⚠️ Not installed (manual filters only)'}"
        )
    with c2:
        st.markdown(
            f"""
<div class="hp-card">
<b>Friendly Input Features</b><br>
{", ".join(FRIENDLY_ALL)}
</div>
""",
            unsafe_allow_html=True,
        )


# =========================================================
# DASHBOARD
# =========================================================
with tab_dash:
    st.subheader("Interactive Dashboard")

    DEFAULT_API = os.getenv("HEALTHPULSE_API_URL", "http://127.0.0.1:8000")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API, key="api_url_main")
    uploaded = st.file_uploader("Upload CSV for dashboard exploration", type=["csv"], key="dash_upload")

    if uploaded is None:
        st.info("Upload a CSV containing the friendly features to unlock the full dashboard.")
        st.stop()

    raw = pd.read_csv(uploaded)

    df = ensure_friendly_columns(raw, FRIENDLY_ALL)
    if "appointment_id" in raw.columns:
        df.insert(0, "appointment_id", raw["appointment_id"].astype(str))

    st.caption(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

    for k in ["flt_specialty", "flt_clinic", "flt_dow", "flt_ins"]:
        if k not in st.session_state:
            st.session_state[k] = []

    st.divider()

    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 2])

    with f1:
        specialty_opts = sorted(df["specialty"].dropna().astype(str).unique().tolist())
        st.session_state.flt_specialty = st.multiselect("Specialty", specialty_opts, default=st.session_state.flt_specialty)
    with f2:
        clinic_opts = sorted(df["clinic_name"].dropna().astype(str).unique().tolist())
        st.session_state.flt_clinic = st.multiselect("Clinic", clinic_opts, default=st.session_state.flt_clinic)
    with f3:
        dow_opts = sorted(df["day_of_week"].dropna().astype(str).unique().tolist())
        # prefer DOW order if possible
        dow_opts = [d for d in DOW_ORDER if d in dow_opts] + [d for d in dow_opts if d not in DOW_ORDER]
        st.session_state.flt_dow = st.multiselect("Day of week", dow_opts, default=st.session_state.flt_dow)
    with f4:
        ins_opts = sorted(df["insurance_type"].dropna().astype(str).unique().tolist())
        st.session_state.flt_ins = st.multiselect("Insurance", ins_opts, default=st.session_state.flt_ins)
    with f5:
        thr = st.slider("High-risk threshold", 0.05, 0.95, THRESHOLD_DEFAULT, 0.01)

    df_f = df.copy()
    if st.session_state.flt_specialty:
        df_f = df_f[df_f["specialty"].astype(str).isin(st.session_state.flt_specialty)]
    if st.session_state.flt_clinic:
        df_f = df_f[df_f["clinic_name"].astype(str).isin(st.session_state.flt_clinic)]
    if st.session_state.flt_dow:
        df_f = df_f[df_f["day_of_week"].astype(str).isin(st.session_state.flt_dow)]
    if st.session_state.flt_ins:
        df_f = df_f[df_f["insurance_type"].astype(str).isin(st.session_state.flt_ins)]

    st.caption(f"Filtered rows: {df_f.shape[0]:,}")

    st.markdown("## KPI Cards (with deltas)")
    st.markdown("<div class='small-muted'>Deltas compare filtered segment vs overall dataset.</div>", unsafe_allow_html=True)

    score_now = st.button("Score filtered segment (API) → unlock risk charts & tables", key="score_segment_btn")

    if score_now:
        try:
            overall_res = call_predict_api(api_url, df.to_dict("records"))
            df_overall_scored = attach_predictions(df, overall_res)

            filt_res = call_predict_api(api_url, df_f.to_dict("records"))
            df_scored = attach_predictions(df_f, filt_res)

            st.session_state.df_overall_scored = df_overall_scored
            st.session_state.df_scored = df_scored
        except Exception as e:
            # show API body if present
            detail = ""
            try:
                if hasattr(e, "response") and e.response is not None:
                    detail = f" Response: {e.response.text}"
            except Exception:
                pass
            st.error(f"Scoring failed: {e}{detail}")

    if "df_scored" in st.session_state and "df_overall_scored" in st.session_state:
        df_overall_scored = st.session_state.df_overall_scored
        df_scored = st.session_state.df_scored

        baseline_high_rate = float((df_overall_scored["no_show_probability"] >= thr).mean())
        baseline_avg_risk = float(df_overall_scored["no_show_probability"].mean())

        total = len(df_scored)
        high_risk = int((df_scored["no_show_probability"] >= thr).sum())
        avg_risk = float(df_scored["no_show_probability"].mean())

        kpi_cards(total, high_risk, avg_risk, baseline_high_rate, baseline_avg_risk, thr)
        st.divider()

        st.markdown("## Drill-down Charts (click-to-filter)")
        st.markdown("<div class='small-muted'>Click a bar to filter (if enabled). Otherwise use manual filters above.</div>", unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_spec = categorical_percent_bar(df_f, "specialty", top_n=15)
            st.plotly_chart(fig_spec, use_container_width=True)
            if HAS_PLOTLY_EVENTS:
                clicked = plotly_events(fig_spec, click_event=True, select_event=False, hover_event=False)
                if clicked:
                    chosen = clicked[0].get("x")
                    if chosen:
                        st.session_state.flt_specialty = [chosen]
                        st.rerun()

        with cc2:
            fig_dow = categorical_percent_bar(df_f, "day_of_week", top_n=15)
            st.plotly_chart(fig_dow, use_container_width=True)
            if HAS_PLOTLY_EVENTS:
                clicked = plotly_events(fig_dow, click_event=True, select_event=False, hover_event=False)
                if clicked:
                    chosen = clicked[0].get("x")
                    if chosen:
                        st.session_state.flt_dow = [chosen]
                        st.rerun()

        st.divider()

        st.markdown("## Feature Distributions (Percent-based)")
        st.markdown("### Numeric")
        n1, n2 = st.columns(2)
        with n1:
            st.plotly_chart(percent_histogram(df_f, "lead_time_days", bins=25), use_container_width=True)
        with n2:
            st.plotly_chart(percent_histogram(df_f, "age", bins=25), use_container_width=True)

        n3, n4, n5 = st.columns(3)
        with n3:
            st.plotly_chart(percent_histogram(df_f, "appt_hour", bins=24), use_container_width=True)
        with n4:
            st.plotly_chart(percent_histogram(df_f, "prev_no_shows", bins=20), use_container_width=True)
        with n5:
            st.plotly_chart(percent_histogram(df_f, "hist_no_show_rate", bins=20), use_container_width=True)

        st.markdown("### Categorical")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(categorical_percent_bar(df_f, "clinic_name", top_n=15), use_container_width=True)
        with c2:
            st.plotly_chart(categorical_percent_bar(df_f, "insurance_type", top_n=15), use_container_width=True)

        st.divider()

        st.markdown("## Appointment Risk Heatmaps")
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(make_heatmap(df_scored, row="specialty", col="day_of_week"), use_container_width=True)
        with h2:
            st.plotly_chart(make_heatmap(df_scored, row="clinic_name", col="day_of_week"), use_container_width=True)

        st.divider()

        st.markdown("## High-Risk Appointments")
        high = df_scored[df_scored["no_show_probability"] >= thr].sort_values("no_show_probability", ascending=False)
        st.dataframe(high.head(300), use_container_width=True)

        st.markdown("### Risk Score Distribution (%)")
        fig_risk = px.histogram(df_scored, x="no_show_probability", nbins=25, histnorm="percent", template=PLOTLY_TEMPLATE)
        fig_risk.update_traces(marker_color=PRIMARY_BAR_COLOR)
        fig_risk.update_layout(
            xaxis_title="No-show probability",
            yaxis_title="Percent (%)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.divider()

        st.markdown("## Segment-level Top Drivers")
        st.markdown("<div class='small-muted'>Computed locally (if model_pipeline.joblib exists). This will never crash the app.</div>", unsafe_allow_html=True)

        if pipeline_local is None:
            st.warning("Segment drivers disabled: artifacts/model_pipeline.joblib not found.")
        else:
            try:
                seg = compute_segment_drivers_local(
                    pipeline=pipeline_local,
                    df_friendly=df_f[FRIENDLY_ALL].copy(),
                    model_features=MODEL_FEATURES,
                    friendly_cat=FRIENDLY_CAT,
                )
                if seg is None or seg.empty:
                    st.warning("Segment drivers unavailable for this segment.")
                else:
                    topn = st.slider("Top N segment drivers", 5, 30, 10, 1)
                    seg_top = seg.head(topn).copy()

                    fig_seg = px.bar(
                        seg_top[::-1],
                        x="importance",
                        y="parent_feature",
                        orientation="h",
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_seg.update_traces(marker_color=PRIMARY_BAR_COLOR)
                    fig_seg.update_layout(
                        xaxis_title="Contribution Importance",
                        yaxis_title="Feature Group",
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_seg, use_container_width=True)
                    st.dataframe(seg_top, use_container_width=True)
            except Exception as e:
                st.warning(f"Segment drivers failed (won’t break app). Reason: {type(e).__name__}: {e}")
    else:
        st.info("Click **Score filtered segment (API)** to compute KPIs, heatmaps, high-risk table, and segment drivers.")


# =========================================================
# PREDICTION
# =========================================================
with tab_predict:
    st.subheader("Prediction (Single + Batch)")
    DEFAULT_API = os.getenv("HEALTHPULSE_API_URL", "http://127.0.0.1:8000")

    # FIX: unique widget key (prevents DuplicateWidgetID)
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API, key="api_url_pred")

    st.markdown("## Single Prediction")
    with st.form("single_form"):
        a1, a2, a3 = st.columns(3)
        with a1:
            appointment_id = st.text_input("appointment_id", "A000001")
            lead_time_days = st.number_input("lead_time_days", 0, 365, 10)
            age = st.number_input("age", 0, 120, 30)
        with a2:
            appt_hour = st.number_input("appt_hour", 0, 23, 10)
            prev_no_shows = st.number_input("prev_no_shows", 0, 999, 0)
        with a3:
            hist_no_show_rate = st.number_input("hist_no_show_rate", 0.0, 1.0, 0.18, step=0.01)

        b1, b2 = st.columns(2)
        with b1:
            specialty = st.text_input("specialty", "Cardiology")
            clinic_name = st.text_input("clinic_name", "Cardiology Clinic")
        with b2:
            day_of_week = st.selectbox("day_of_week", DOW_ORDER, index=0)
            insurance_type = st.text_input("insurance_type", "Private")

        submit = st.form_submit_button("Predict")

    if submit:
        record = {
            "appointment_id": appointment_id,
            "lead_time_days": lead_time_days,
            "age": age,
            "appt_hour": appt_hour,
            "prev_no_shows": prev_no_shows,
            "hist_no_show_rate": hist_no_show_rate,
            "specialty": specialty,
            "clinic_name": clinic_name,
            "day_of_week": day_of_week,
            "insurance_type": insurance_type,
        }
        try:
            res = call_predict_api(api_url, [record])

            pred = res["predictions"][0]
            p = float(pred["no_show_probability"])
            y = int(pred["predicted_label"])
            band = risk_band(p, THRESHOLD_DEFAULT)

            st.markdown("### Result (easy to read)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("No-Show Risk", f"{p*100:.2f}%")
            c2.metric("Prediction", "NO-SHOW" if y == 1 else "SHOW")
            c3.metric("Risk Band", band)
            c4.metric("Threshold", f"{THRESHOLD_DEFAULT:.2f}")

            st.markdown(
                f"""
<div class="hp-card">
<b>Interpretation:</b><br>
This appointment is classified as <b>{"High Concern" if band=="High" else "Moderate Concern" if band=="Medium" else "Lower Concern"}</b>.
Recommended action:
<ul>
<li><b>Low:</b> standard reminder</li>
<li><b>Medium:</b> reminder + offer easy reschedule link</li>
<li><b>High:</b> call/SMS + reschedule option + consider transport support</li>
</ul>
</div>
""",
                unsafe_allow_html=True,
            )

            with st.expander("Raw API response"):
                st.json(res)

        except Exception as e:
            detail = ""
            try:
                if hasattr(e, "response") and e.response is not None:
                    detail = f" Response: {e.response.text}"
            except Exception:
                pass
            st.error(f"Prediction failed: {e}{detail}")

    st.divider()

    st.markdown("## Batch Prediction (CSV Upload)")
    batch = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    thr = st.slider("High-risk threshold (batch)", 0.05, 0.95, THRESHOLD_DEFAULT, 0.01, key="thr_batch")

    if batch is not None:
        dfb_raw = pd.read_csv(batch)
        dfb = ensure_friendly_columns(dfb_raw, FRIENDLY_ALL)
        if "appointment_id" in dfb_raw.columns:
            dfb.insert(0, "appointment_id", dfb_raw["appointment_id"].astype(str))

        st.dataframe(dfb.head(10), use_container_width=True)

        if st.button("Run batch prediction (API)"):
            try:
                res = call_predict_api(api_url, dfb.to_dict("records"))
                scored = attach_predictions(dfb, res)

                total = len(scored)
                high_risk = int((scored["no_show_probability"] >= thr).sum())
                avg_risk = float(scored["no_show_probability"].mean())

                c1, c2, c3 = st.columns(3)
                c1.metric("Records", f"{total:,}")
                c2.metric("High-Risk", f"{high_risk:,}")
                c3.metric("Avg Risk", f"{avg_risk*100:.2f}%")

                st.markdown("### High-risk table")
                high = scored[scored["no_show_probability"] >= thr].sort_values("no_show_probability", ascending=False)
                st.dataframe(high.head(300), use_container_width=True)

                st.markdown("### Risk distribution (%)")
                fig_r = px.histogram(scored, x="no_show_probability", nbins=25, histnorm="percent", template=PLOTLY_TEMPLATE)
                fig_r.update_traces(marker_color=PRIMARY_BAR_COLOR)
                fig_r.update_layout(
                    xaxis_title="No-show probability",
                    yaxis_title="Percent (%)",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_r, use_container_width=True)

                st.markdown("### Download results")
                st.download_button(
                    "Download scored CSV",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="scored_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                detail = ""
                try:
                    if hasattr(e, "response") and e.response is not None:
                        detail = f" Response: {e.response.text}"
                except Exception:
                    pass
                st.error(f"Batch scoring failed: {e}{detail}")


# =========================================================
# EXPLAINABILITY
# =========================================================
with tab_drivers:
    st.subheader("Explainability (Global Drivers)")

    imp_df = None
    source = None

    if SHAP_IMP_PATH.exists():
        df_shap = load_csv(SHAP_IMP_PATH)
        if "mean_abs_shap" in df_shap.columns:
            imp_df = df_shap.rename(columns={"mean_abs_shap": "importance"})
            source = "SHAP mean(|value|)"

    if imp_df is None and GLOBAL_IMP_PATH.exists():
        df_imp = load_csv(GLOBAL_IMP_PATH)
        if "abs_importance" in df_imp.columns:
            imp_df = df_imp.rename(columns={"abs_importance": "importance"})
            source = "LogReg |coef|"

    if imp_df is None:
        st.warning("No importance file found in artifacts/. Generate it from your modeling notebook.")
    else:
        st.caption(f"Source: {source}")
        imp_df = imp_df.sort_values("importance", ascending=False)

        topn = st.slider("Top N global features", 10, 60, 20, 1)
        top = imp_df.head(topn).copy()

        fig = px.bar(
            top[::-1],
            x="importance",
            y="feature",
            orientation="h",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(marker_color=PRIMARY_BAR_COLOR)
        fig.update_layout(xaxis_title="Importance", yaxis_title="Feature", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top, use_container_width=True)


# =========================================================
# RECOMMENDATIONS
# =========================================================
with tab_reco:
    st.subheader("Recommendations for Reducing No-Shows")

    st.markdown(
        f"""
<div class="hp-card">
<h3 style="margin-top:0;">Project Summary</h3>
HealthPulse predicts appointment no-shows using a <b>{MODEL_TYPE}</b> model and a tuned decision threshold
(<b>{THRESHOLD_DEFAULT:.2f}</b>) to prioritize catching more likely no-shows.

<h3>How to turn predictions into impact</h3>
<ul>
<li><b>Focus on the high-risk list</b> to maximize ROI of outreach effort.</li>
<li><b>Use segment patterns</b> (heatmaps + filters) to tailor interventions per clinic/specialty/day.</li>
<li><b>Measure changes</b> using KPI deltas (segment vs overall baseline).</li>
</ul>

<h3>Operational Playbook</h3>
<ol>
<li><b>Tiered interventions:</b> SMS → SMS + reschedule link → call + reschedule + transport help.</li>
<li><b>Reduce friction:</b> make rescheduling easy (1-click) and add “confirm attendance” prompts.</li>
<li><b>Clinic workflow:</b> assign a daily high-risk call list by clinic team.</li>
<li><b>Capacity strategy:</b> consider risk-aware overbooking where clinically safe.</li>
<li><b>Monitor drift:</b> track weekly average risk + high-risk rate; retrain when distributions shift.</li>
</ol>

<span class="small-muted">
Use “Segment-level Top Drivers” to understand what drives risk in the filtered group, then choose an intervention that removes that friction.
</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Friendly inputs used by the app")
    st.code(", ".join(["appointment_id"] + FRIENDLY_ALL))