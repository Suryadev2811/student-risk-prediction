import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from sklearn.impute import SimpleImputer
if not hasattr(SimpleImputer, "_fill_dtype"):
    SimpleImputer._fill_dtype = property(lambda self: np.float64)

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="EduRisk · AI Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  GLOBAL CSS  (premium dark theme)
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:       #04080f;
  --surface:  #0a1628;
  --surface2: #0f1f38;
  --border:   rgba(99,179,255,0.12);
  --accent:   #3b82f6;
  --accent2:  #06b6d4;
  --good:     #10b981;
  --risk:     #f59e0b;
  --critical: #ef4444;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --glow:     rgba(59,130,246,0.15);
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'Space Grotesk', sans-serif;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── headers ── */
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif; color:var(--text); }

/* ── metric cards ── */
.kpi-card {
  background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px 24px;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content:'';
  position:absolute;
  top:-40px; right:-40px;
  width:120px; height:120px;
  border-radius:50%;
  background: var(--glow);
}
.kpi-label { font-size:12px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
.kpi-value { font-size:38px; font-weight:700; line-height:1; font-family:'JetBrains Mono',monospace; }
.kpi-sub   { font-size:12px; color:var(--muted); margin-top:6px; }

/* ── section card ── */
.sec {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  margin: 16px 0;
}
.sec-title {
  font-size:15px; font-weight:600; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted);
  margin-bottom:16px; padding-bottom:12px;
  border-bottom:1px solid var(--border);
}

/* ── risk badge ── */
.badge-good     { background:rgba(16,185,129,.15); color:#10b981; border:1px solid rgba(16,185,129,.3); padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-atrisk   { background:rgba(245,158,11,.15); color:#f59e0b; border:1px solid rgba(245,158,11,.3); padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-critical { background:rgba(239,68,68,.15);  color:#ef4444; border:1px solid rgba(239,68,68,.3);  padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }

/* ── hero banner ── */
.hero {
  background: linear-gradient(135deg,#0a1f40 0%,#071525 60%,#0a0f1a 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 36px 40px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}
.hero::after {
  content:'';
  position:absolute; top:0; right:0; bottom:0;
  width:40%;
  background: radial-gradient(ellipse at right center, rgba(59,130,246,.12), transparent 70%);
}
.hero h1 { font-size:28px; font-weight:700; margin:0 0 6px; }
.hero p  { color:var(--muted); font-size:14px; margin:0; }
.hero .tag { display:inline-block; background:rgba(59,130,246,.15); color:var(--accent); border:1px solid rgba(59,130,246,.25); border-radius:20px; padding:3px 12px; font-size:11px; font-weight:600; letter-spacing:.08em; margin-top:12px; }

/* ── table styling ── */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }

/* ── plotly dark override ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── tab styling ── */
[data-baseweb="tab-list"] { background:var(--surface) !important; border-radius:10px; padding:4px; border:1px solid var(--border); }
[data-baseweb="tab"] { color:var(--muted) !important; font-family:'Space Grotesk',sans-serif !important; font-weight:500 !important; }
[aria-selected="true"] { background:var(--surface2) !important; color:var(--text) !important; border-radius:8px !important; }

/* ── number input ── */
[data-testid="stNumberInput"] input { background:var(--surface2) !important; border:1px solid var(--border) !important; color:var(--text) !important; border-radius:8px !important; font-family:'JetBrains Mono',monospace !important; }

/* ── selectbox ── */
[data-testid="stSelectbox"] > div > div { background:var(--surface2) !important; border:1px solid var(--border) !important; border-radius:8px !important; }

/* ── button ── */
[data-testid="stDownloadButton"] button {
  background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
  border:none !important; border-radius:10px !important;
  color:white !important; font-weight:600 !important;
  font-family:'Space Grotesk',sans-serif !important;
  padding:10px 24px !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] { background:var(--surface2) !important; border-radius:12px !important; border:1px dashed var(--border) !important; }

/* ── slider ── */
[data-testid="stSlider"] .thumb { background:var(--accent) !important; }

/* scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

/* ── info box ── */
.info-box {
  background:rgba(59,130,246,.08);
  border:1px solid rgba(59,130,246,.2);
  border-radius:12px; padding:16px 20px;
  color:var(--text); font-size:14px; line-height:1.6;
}
.warn-box {
  background:rgba(245,158,11,.08);
  border:1px solid rgba(245,158,11,.2);
  border-radius:12px; padding:16px 20px;
  color:var(--text); font-size:14px; line-height:1.6;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8", size=12),
    margin=dict(t=30, b=30, l=10, r=10),
    xaxis=dict(gridcolor="rgba(99,179,255,0.06)", linecolor="rgba(99,179,255,0.1)"),
    yaxis=dict(gridcolor="rgba(99,179,255,0.06)", linecolor="rgba(99,179,255,0.1)"),
)
RISK_COLORS = {"Good": "#10b981", "AtRisk": "#f59e0b", "Critical": "#ef4444"}

# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import gdown 

MODEL_URL = "https://drive.google.com/uc?id=1lrxOFCTAuu8Im6aGwM8xawkBXLweWvz9"
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.joblib")
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇ Downloading ML model... please wait"):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.joblib"))

    return model, encoder
rf_model, label_encoder = load_artifacts()
FEATURES = [
    'attendance_pct',
    'quiz_1','quiz_2','quiz_3','quiz_4','quiz_5',
    'quiz_avg','quiz_std',
    'assignment_score',
    'sessional1',
    'cheating_count','teacher_feedback_score'
]
REQUIRED_COLS = [
    'student_id','attendance_pct',
    'quiz_1','quiz_2','quiz_3','quiz_4','quiz_5',
    'assignment_score','sessional1',
    'cheating_count','teacher_feedback_score'
]

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 12px;">
      <div style="font-size:36px;">🎓</div>
      <div style="font-size:18px;font-weight:700;letter-spacing:.04em;">EduRisk AI</div>
      <div style="font-size:11px;color:#64748b;margin-top:4px;">Academic Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📁 Upload Dataset**")
    uploaded_file = st.file_uploader("CSV or XLSX", type=["csv", "xlsx"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Required Columns**")
    cols_display = "\n".join(f"• {c}" for c in REQUIRED_COLS)
    st.markdown(f"<div style='font-size:12px;color:#64748b;line-height:1.8'>{cols_display}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#475569;line-height:1.7">
      <strong style="color:#64748b">Model:</strong> Random Forest + SHAP<br>
      <strong style="color:#64748b">Classes:</strong> Good · AtRisk · Critical<br>
      <strong style="color:#64748b">Explainability:</strong> Per-student SHAP<br>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <h1>🎓 Student Academic Risk Intelligence</h1>
      <p>A Modular ML Approach for Predicting Performance &amp; Supporting Early Academic Interventions — powered by Explainable AI (SHAP)</p>
      <span class="tag">✦ RANDOM FOREST · SHAP · REAL-TIME DASHBOARD</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  NO FILE STATE
# ─────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
      <strong>👈 Get started</strong> — Upload a CSV or XLSX file from the sidebar to generate predictions, 
      visualise risk distributions, and explore per-student SHAP explanations.
    </div>
    """, unsafe_allow_html=True)

    # Show sample schema
    st.markdown('<div class="sec"><div class="sec-title">Expected Data Schema</div>', unsafe_allow_html=True)
    sample = pd.DataFrame({
        "student_id":["S001","S002"],
        "attendance_pct":[85, 42],
        "quiz_1":[8,3],"quiz_2":[7,4],"quiz_3":[9,2],"quiz_4":[6,3],"quiz_5":[8,1],
        "assignment_score":[78,35],
        "sessional1":[70,28],
        "cheating_count":[0,2],"teacher_feedback_score":[4.2,1.8]
    })
    st.dataframe(sample, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────
#  LOAD & VALIDATE DATA
# ─────────────────────────────────────────
if uploaded_file.name.endswith(".xlsx"):
    df_raw = pd.read_excel(uploaded_file)
else:
    df_raw = pd.read_csv(uploaded_file)

missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_cols:
    st.error(f"❌ Missing columns: {missing_cols}")
    st.stop()

df = df_raw.copy()

# Feature engineering
quiz_cols = ['quiz_1','quiz_2','quiz_3','quiz_4','quiz_5']
df['quiz_avg'] = df[quiz_cols].mean(axis=1)
df['quiz_std'] = df[quiz_cols].std(axis=1)

# Align with model
model_features = rf_model.feature_names_in_ if hasattr(rf_model, "feature_names_in_") else FEATURES
for col in model_features:
    if col not in df.columns:
        df[col] = 0
X = df[model_features]

# Predict
preds = rf_model.predict(X)
probs = rf_model.predict_proba(X)

try:
    df['Risk'] = label_encoder.inverse_transform(preds)
except:
    df['Risk'] = preds

try:
    for i, cls in enumerate(label_encoder.classes_):
        df[f'Prob_{cls}'] = probs[:, i]
except:
    pass

# ─────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────
total    = len(df)
good_n   = (df['Risk'] == 'Good').sum()
risk_n   = (df['Risk'] == 'AtRisk').sum()
crit_n   = (df['Risk'] == 'Critical').sum()
avg_att  = df['attendance_pct'].mean()
avg_quiz = df['quiz_avg'].mean()

k1,k2,k3,k4,k5,k6 = st.columns(6)
kpis = [
    (k1, "Total Students",    str(total),     "white",            "Enrolled"),
    (k2, "Good Standing",     str(good_n),    "#10b981",          f"{good_n/total*100:.0f}% of class"),
    (k3, "At Risk",           str(risk_n),    "#f59e0b",          f"{risk_n/total*100:.0f}% of class"),
    (k4, "Critical",          str(crit_n),    "#ef4444",          f"{crit_n/total*100:.0f}% of class"),
    (k5, "Avg Attendance",    f"{avg_att:.1f}%", "#3b82f6",       "Class average"),
    (k6, "Avg Quiz Score",    f"{avg_quiz:.1f}", "#06b6d4",       "Out of 10"),
]
for col, label, value, color, sub in kpis:
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📋 Student Table", "📈 Analytics", "🧠 SHAP Explorer", "⚠️ Intervention List"
])

# ══════════════════════════════════
#  TAB 1 · OVERVIEW
# ══════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        # Donut
        risk_counts = df['Risk'].value_counts().reset_index()
        risk_counts.columns = ['Risk','Count']
        colors = [RISK_COLORS.get(r, "#64748b") for r in risk_counts['Risk']]
        fig_donut = go.Figure(go.Pie(
            labels=risk_counts['Risk'], values=risk_counts['Count'],
            hole=0.65,
            marker=dict(colors=colors, line=dict(color="#04080f", width=3)),
            textfont=dict(family="Space Grotesk", size=13),
        ))
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Risk Distribution", font=dict(size=14, color="#94a3b8")),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:10'>students</span>",
                               x=0.5, y=0.5, font_size=18, showarrow=False, font_color="white")]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Attendance vs Sessional1 scatter
        fig_scatter = px.scatter(
            df, x='attendance_pct', y='sessional1',
            color='Risk', size='quiz_avg',
            color_discrete_map=RISK_COLORS,
            hover_data=[c for c in ['student_id','student_name'] if c in df.columns] or None,
            labels={'attendance_pct':'Attendance %','sessional1':'Sessional 1 Score'},
        )
        fig_scatter.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Attendance vs Sessional Score", font=dict(size=14, color="#94a3b8")),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Quiz trend heatmap
    quiz_df = df[['Risk'] + quiz_cols].copy()
    quiz_avg_by_risk = quiz_df.groupby('Risk')[quiz_cols].mean()
    fig_heat = px.imshow(
        quiz_avg_by_risk,
        color_continuous_scale=[[0,"#1e1b4b"],[0.5,"#3b82f6"],[1,"#06b6d4"]],
        text_auto=".1f",
        labels=dict(x="Quiz", y="Risk Group", color="Avg Score"),
        aspect="auto",
    )
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Average Quiz Score by Risk Group", font=dict(size=14, color="#94a3b8")),
        coloraxis_showscale=True,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════
#  TAB 2 · STUDENT TABLE
# ══════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">Full Prediction Results</div>', unsafe_allow_html=True)

    # Filter
    f1, f2, f3 = st.columns([2,2,2])
    with f1:
        risk_filter = st.multiselect("Filter by Risk", options=df['Risk'].unique().tolist(), default=df['Risk'].unique().tolist())
    with f2:
        att_min = st.slider("Min Attendance %", 0, 100, 0)
    with f3:
        valid_sort_cols = [c for c in ['attendance_pct','quiz_avg','sessional1','assignment_score','cheating_count','teacher_feedback_score'] if c in df.columns]
        sort_col = st.selectbox("Sort by", options=valid_sort_cols)
        ascending = st.toggle("Ascending Order", value=False)

    df_filtered = df[
        df['Risk'].isin(risk_filter) &
        (df['attendance_pct'] >= att_min)
    ].sort_values(by=sort_col, ascending=ascending)
    
    # Display columns — name after id, prob cols renamed & rounded
    base_cols = ['student_id','student_name','Risk','attendance_pct','quiz_avg','quiz_std',
                 'assignment_score','sessional1','cheating_count','teacher_feedback_score']
    display_cols = [c for c in base_cols if c in df_filtered.columns]
    prob_cols = [c for c in df_filtered.columns if c.startswith('Prob_')]
    display_cols += prob_cols

    df_show = df_filtered[display_cols].reset_index(drop=True).copy()
    # Round all float cols for clean display
    for c in df_show.select_dtypes(include='float').columns:
        df_show[c] = df_show[c].round(3)
    # Rename prob cols to shorter names
    rename_map = {c: c.replace('Prob_','P(') + ')' for c in prob_cols}
    df_show.rename(columns=rename_map, inplace=True)

    st.dataframe(df_show, use_container_width=True, height=420)
    st.markdown(f"<div style='color:#64748b;font-size:12px'>Showing {len(df_filtered)} of {len(df)} students</div>", unsafe_allow_html=True)

    st.download_button(
        "⬇ Download Results CSV",
        df.to_csv(index=False).encode("utf-8"),
        "student_risk_predictions.csv",
        "text/csv"
    )

# ══════════════════════════════════
#  TAB 3 · ANALYTICS
# ══════════════════════════════════
with tab3:
    r1, r2 = st.columns(2)

    with r1:
        # Grouped bar: avg scores by risk
        feat_avg = df.groupby('Risk')[['attendance_pct','assignment_score','sessional1','quiz_avg']].mean().reset_index()
        feat_melt = feat_avg.melt(id_vars='Risk', var_name='Feature', value_name='Average')
        fig_bar = px.bar(
            feat_melt, x='Feature', y='Average', color='Risk', barmode='group',
            color_discrete_map=RISK_COLORS,
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Average Feature Values by Risk Group", font=dict(size=14, color="#94a3b8")),
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with r2:
        # Box plot attendance
        fig_box = px.box(
            df, x='Risk', y='attendance_pct', color='Risk',
            color_discrete_map=RISK_COLORS,
            points="all",
        )
        fig_box.update_traces(marker=dict(size=3, opacity=0.6))
        fig_box.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Attendance Distribution by Risk", font=dict(size=14, color="#94a3b8")),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    r3, r4 = st.columns(2)

    with r3:
        # Violin: quiz_avg
        fig_violin = px.violin(
            df, x='Risk', y='quiz_avg', color='Risk',
            color_discrete_map=RISK_COLORS, box=True,
        )
        fig_violin.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Quiz Average Distribution by Risk", font=dict(size=14, color="#94a3b8")),
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    with r4:
        # Cheating count bar
        cheat_df = df.groupby(['Risk','cheating_count']).size().reset_index(name='Count')
        fig_cheat = px.bar(cheat_df, x='cheating_count', y='Count', color='Risk',
                           color_discrete_map=RISK_COLORS, barmode='stack')
        fig_cheat.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Cheating Incidents by Risk Group", font=dict(size=14, color="#94a3b8")),
            xaxis_title="Cheating Count", yaxis_title="Students",
        )
        st.plotly_chart(fig_cheat, use_container_width=True)

    # Parallel coordinates (feature profile)
    parallel_cols = ['attendance_pct','quiz_avg','assignment_score','sessional1','teacher_feedback_score']
    parallel_cols = [c for c in parallel_cols if c in df.columns]
    risk_code = df['Risk'].map({'Good':0,'AtRisk':1,'Critical':2}).fillna(0).astype(int)
    fig_par = px.parallel_coordinates(
        df[parallel_cols + ['Risk']],
        color=risk_code,
        color_continuous_scale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]],
        dimensions=parallel_cols,
    )
    fig_par.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Feature Profiles Across Risk Levels", font=dict(size=14, color="#94a3b8")),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_par, use_container_width=True)
# ══════════════════════════════════
#  TAB 4 · SHAP EXPLORER
# ══════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">Explainable AI — Per-Student SHAP Analysis</div>', unsafe_allow_html=True)

    # Build dropdown
    def make_label(i, row):
        sid  = str(row['student_id']) if 'student_id' in df.columns else f"#{i}"
        name = str(row['student_name']) if 'student_name' in df.columns else ""
        risk = row['Risk']
        return f"{sid}  —  {name}  [{risk}]" if name else f"{sid}  [{risk}]"

    dropdown_options = [make_label(i, row) for i, row in df.iterrows()]
    selected_label = st.selectbox("🔍 Select Student", options=dropdown_options, index=0)
    student_idx = dropdown_options.index(selected_label)

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        sid = str(df.iloc[student_idx]['student_id'])
        sname = str(df.iloc[student_idx]['student_name'])
        risk_val = df.iloc[student_idx]['Risk']

        badge_cls = {"Good":"badge-good","AtRisk":"badge-atrisk","Critical":"badge-critical"}.get(risk_val,"badge-good")
        name_html = f'<div style="font-size:15px;color:#94a3b8;margin-top:2px;font-weight:500">{sname}</div>'

        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Student</div>
          <div style="font-size:26px;font-weight:700">{sid}</div>
          {name_html}
          <div style="margin-top:10px"><span class="{badge_cls}">{risk_val}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        snap_cols = ['attendance_pct','quiz_avg','quiz_std','assignment_score','sessional1','cheating_count','teacher_feedback_score']
        snap = df.iloc[student_idx][snap_cols]

        snap_df = pd.DataFrame({
            "Feature": snap.index,
            "Value": [round(float(v), 2) for v in snap.values]
        })

        fig_snap = px.bar(snap_df, x="Value", y="Feature", orientation="h")
        st.plotly_chart(fig_snap, use_container_width=True)

    # ───────────── SHAP FIXED BLOCK ─────────────
    st.markdown('<div class="sec-title">SHAP Feature Contributions</div>', unsafe_allow_html=True)

    try:
        rf_clf = rf_model.named_steps["clf"]
        imputer = rf_model.named_steps["imputer"]

        X_imp = imputer.transform(X)
        explainer = shap.TreeExplainer(rf_clf)

        shap_vals = explainer.shap_values(X_imp)

        pred_class_idx = int(preds[student_idx])

        if isinstance(shap_vals, list):
            sv = shap_vals[pred_class_idx][student_idx]
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[student_idx, :, pred_class_idx]
        else:
            sv = shap_vals[student_idx]

        # ✅ important fix
        sv = np.array(sv).reshape(-1)

        min_len = min(len(sv), len(model_features))

        shap_df = pd.DataFrame({
            "Feature": model_features[:min_len],
            "SHAP Value": sv[:min_len],
        }).sort_values("SHAP Value", key=abs, ascending=False).head(12)

        fig_shap = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="SHAP Value",
            color_continuous_scale="RdBu"
        )

        st.plotly_chart(fig_shap, use_container_width=True)

        # Global importance (FIXED)
        if isinstance(shap_vals, list):
            importance = np.mean(
                [np.abs(shap_vals[i]).mean(axis=0).flatten() for i in range(len(shap_vals))],
                axis=0
             )
        elif len(shap_vals.shape) == 3:
            importance = np.abs(shap_vals).mean(axis=0)[:, pred_class_idx]
        else:
            importance = np.abs(shap_vals).mean(axis=0)
        # ensure 1D
        importance = np.array(importance).reshape(-1)

        min_len = min(len(importance), len(model_features))

        global_df = pd.DataFrame({
            "Feature": model_features[:min_len],
            "Importance": importance[:min_len],
        }).sort_values("Importance", ascending=True)

        fig_global = px.bar(global_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig_global, use_container_width=True)
        # ✅ CLOSE TRY BLOCK HERE
    except Exception as e:
        st.markdown(f'<div class="warn-box">⚠️ SHAP error: {e}</div>', unsafe_allow_html=True)
# ══════════════════════════════════
#  TAB 5 · INTERVENTION LIST
# ══════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title">Students Requiring Academic Intervention</div>', unsafe_allow_html=True)

    at_risk_df = df[df['Risk'].isin(['AtRisk','Critical'])].copy()

    if at_risk_df.empty:
        st.markdown('<div class="info-box">🎉 No students require immediate intervention.</div>', unsafe_allow_html=True)
    else:
        # Priority score
        sessional_col = 'sessional1' if 'sessional1' in at_risk_df.columns else 'assignment_score'
        at_risk_df['Priority Score'] = (
            (100 - at_risk_df['attendance_pct']) * 0.3
            + (10 - at_risk_df['quiz_avg'].clip(0, 10)) * 3
            + at_risk_df['cheating_count'] * 5
            + (100 - at_risk_df[sessional_col].fillna(50)) * 0.2
        ).round(1)
        at_risk_df = at_risk_df.sort_values('Priority Score', ascending=False)

        # Intervention flags
        def flags(row):
            f = []
            if row['attendance_pct'] < 60: f.append("🔴 Low Attendance")
            if row['quiz_avg'] < 4:        f.append("🟠 Poor Quizzes")
            if row.get('cheating_count',0) > 0: f.append("⚫ Academic Integrity")
            if row.get('teacher_feedback_score',5) < 2: f.append("🔶 Low Teacher Feedback")
            return "  ".join(f) if f else "✅ Monitor"
        at_risk_df['Intervention Flags'] = at_risk_df.apply(flags, axis=1)

        show_cols = [c for c in ['student_id','student_name','Risk','Priority Score','attendance_pct','quiz_avg',
                                   'sessional1','cheating_count','teacher_feedback_score','Intervention Flags'] if c in at_risk_df.columns]
        st.dataframe(at_risk_df[show_cols].reset_index(drop=True), use_container_width=True, height=400)

        # Summary
        i1, i2, i3 = st.columns(3)
        i1.metric("Total At-Risk / Critical", len(at_risk_df))
        i2.metric("Avg Priority Score", f"{at_risk_df['Priority Score'].mean():.1f}")
        i3.metric("Avg Attendance (At-Risk)", f"{at_risk_df['attendance_pct'].mean():.1f}%")

        st.download_button(
            "⬇ Download Intervention List",
            at_risk_df.to_csv(index=False).encode("utf-8"),
            "intervention_list.csv",
            "text/csv"
        )
