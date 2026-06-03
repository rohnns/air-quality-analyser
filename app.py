import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import joblib
import hashlib
import datetime
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="AirPulse",
    page_icon="○",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:ital,wght@0,100..900;1,100..900&display=swap');

:root {
    --bg:       #121212;
    --surface:  #1a1a1a;
    --line:     #2a2a2a;
    --text:     #e8e8e8;
    --muted:    #666666;
    --dim:      #444444;
    --white:    #ffffff;
    --font:     'Archivo', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    --pad:      3rem;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
}

section[data-testid="stSidebar"] .stSelectbox {
    margin-top: 1rem;
    margin-bottom: 1rem;
}

section[data-testid="stSidebar"] [data-baseweb="select"] {
    min-height: 40px;
}

#MainMenu, footer, header { visibility: hidden; }

/* Kill all Streamlit padding */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
.element-container {
    margin: 0 !important;
    padding: 0 !important;
}

/* ── TOP NAV BAR ── */
.ap-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--pad);
    height: 52px;
    border-bottom: 1px solid var(--line);
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 100;
}
.ap-nav-logo {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--white);
}
.ap-nav-right {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── CONTENT WRAPPER — applied per block ── */
.ap-wrap {
    padding-left: var(--pad);
    padding-right: var(--pad);
}

/* ── STATION HEADER ── */
.ap-header {
    padding: 4rem var(--pad) 3rem var(--pad);
    border-bottom: 1px solid var(--line);
}
.ap-header-label {
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.25rem;
}
.ap-header-title {
    font-size: clamp(2.4rem, 5vw, 4.5rem);
    font-weight: 300;
    line-height: 1;
    color: var(--white);
    letter-spacing: -0.03em;
    margin: 0;
}
.ap-header-title em {
    font-style: normal;
    color: var(--muted);
}

/* ── STAT INDEX ROW ── */
.ap-index {
    display: flex;
    border-bottom: 1px solid var(--line);
    padding-left: var(--pad);
    padding-right: var(--pad);
}
.ap-index-cell {
    flex: 1;
    padding: 1.5rem 0;
    padding-right: 1.5rem;
    border-right: 1px solid var(--line);
}
.ap-index-cell:last-child {
    border-right: none;
    padding-right: 0;
}
.ap-index-cell + .ap-index-cell {
    padding-left: 1.5rem;
}
.ap-index-label {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.ap-index-value {
    font-size: 2rem;
    font-weight: 300;
    color: var(--white);
    letter-spacing: -0.03em;
    line-height: 1;
}
.ap-index-unit {
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--dim);
    margin-top: 0.4rem;
    text-transform: uppercase;
}

/* ── SECTION LABEL ── */
.ap-section {
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 1px solid var(--line) !important;
    padding: 0 var(--pad) !important;
    margin: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--font) !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 1rem 0 !important;
    margin-right: 2.5rem !important;
    border: none !important;
    border-bottom: 1px solid transparent !important;
    transition: color 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--white) !important;
    border-bottom: 1px solid var(--white) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 0 var(--pad) !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--line) !important;
    width: 240px !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: var(--font) !important;
}
.ap-sidebar-head {
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted) !important;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--line);
    margin: 1.5rem 0 0.75rem 0;
}
.ap-sidebar-logo {
    font-size: 13px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--white) !important;
    font-weight: 500;
    padding-top: 2rem;
}
.ap-sidebar-sub {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted) !important;
    margin-top: 0.3rem;
}
.ap-sidebar-info {
    font-size: 11px;
    color: var(--muted) !important;
    line-height: 2;
}
.ap-sidebar-info span {
    color: var(--text) !important;
}

/* ── SELECTBOX ── */

.stSelectbox {
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}

.stSelectbox [data-baseweb="select"] {
    min-height: 42px;
}

.stSelectbox [data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e8e8 !important;
}

.stSelectbox * {
    color: #e8e8e8 !important;
}

/* ── DATAFRAMES ── */
.stDataFrame {
    border: none !important;
    border-top: 1px solid var(--line) !important;
}

/* ── ALERTS ── */
.stAlert {
    background: var(--surface) !important;
    border: none !important;
    border-left: 2px solid var(--line) !important;
    border-radius: 0 !important;
    font-family: var(--font) !important;
    font-size: 12px !important;
    color: var(--muted) !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: transparent !important;
    border: none !important;
    border-top: 1px solid var(--line) !important;
    border-radius: 0 !important;
    font-family: var(--font) !important;
    font-size: 10px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 0.75rem 0 !important;
}

/* ── PLOTLY ── */
.js-plotly-plot {
    border: none !important;
    border-top: 1px solid var(--line) !important;
}

/* ── PROGRESS ── */
.stProgress > div > div {
    background: var(--white) !important;
    height: 1px !important;
}
.stProgress > div {
    background: var(--line) !important;
    height: 1px !important;
    border-radius: 0 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--line); }

/* ── BODY TEXT ── */
.ap-body {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.8;
    max-width: 560px;
    margin-bottom: 1.5rem;
}

/* ── RISK TABLE ── */
.ap-table-head {
    display: grid;
    grid-template-columns: 64px 60px 1fr 80px 160px 100px;
    padding: 0.6rem 0;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    column-gap: 0;
}
.ap-table-head span {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
}
.ap-table-row {
    display: grid;
    grid-template-columns: 64px 60px 1fr 80px 160px 100px;
    padding: 1rem 0;
    border-bottom: 1px solid var(--line);
    align-items: center;
    column-gap: 0;
}
.ap-table-row:hover { background: var(--surface); }
.ap-col-wk  { font-size: 11px; color: var(--muted); }
.ap-col-mo  { font-size: 12px; color: var(--text); }
.ap-col-bar { display: flex; align-items: center; padding-right: 1rem; }
.ap-bar-track {
    flex: 1;
    height: 1px;
    background: var(--line);
    position: relative;
}
.ap-bar-fill {
    position: absolute;
    left: 0; top: 0;
    height: 1px;
    background: var(--text);
}
.ap-col-prob { font-size: 12px; color: var(--text); font-weight: 500; }
.ap-col-pol  { font-size: 11px; color: var(--muted); padding-left: 0.5rem; }
.ap-col-status {
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: right;
}
.ap-col-status.high { color: var(--white); }

/* ── NO RISK NOTICE ── */
.ap-notice {
    padding: 2rem 0;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    border-top: 1px solid var(--line);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='#121212',
        plot_bgcolor='#121212',
        font=dict(family='Archivo, Helvetica Neue, Helvetica, Arial, sans-serif',
                  color='#666666', size=11),
        xaxis=dict(gridcolor='#2a2a2a', linecolor='#2a2a2a',
                   zerolinecolor='#2a2a2a', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#2a2a2a', linecolor='#2a2a2a',
                   zerolinecolor='#2a2a2a', tickfont=dict(size=10)),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#2a2a2a',
                    font=dict(size=10)),
        colorway=['#e8e8e8', '#666666', '#444444', '#999999', '#bbbbbb', '#333333'],
        margin=dict(t=40, b=40, l=50, r=20),
    )
)

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
DATA_DIR = "./data"

@st.cache_data
def load_and_prep_data():
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        st.error(f"No CSV files found in: {DATA_DIR}")
        st.stop()
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    station_names = sorted(df['station'].unique().tolist())
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    cols_to_drop = [c for c in ['year', 'month', 'day', 'hour', 'No'] if c in df.columns]
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df, station_names, dropped

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_features(df, fit_df=None):
    out = df.copy()
    out['hour']        = out.index.hour
    out['day_of_week'] = out.index.dayofweek
    out['month']       = out.index.month
    for feat in ['PM2.5', 'TEMP', 'WSPM']:
        for lag in [1, 24, 48]:
            out[f'{feat}_lag_{lag}'] = out[feat].shift(lag)
    window = 24
    if fit_df is not None:
        seed = fit_df[['PM2.5']].tail(window).copy()
        combined = pd.concat([seed, out[['PM2.5']]])
        roll_mean = combined['PM2.5'].rolling(window, min_periods=1).mean().iloc[len(seed):]
        roll_std  = combined['PM2.5'].rolling(window, min_periods=1).std().iloc[len(seed):]
        out['PM2.5_roll_mean_24h'] = roll_mean.values
        out['PM2.5_roll_std_24h']  = roll_std.values
    else:
        out['PM2.5_roll_mean_24h'] = out['PM2.5'].rolling(window, min_periods=1).mean()
        out['PM2.5_roll_std_24h']  = out['PM2.5'].rolling(window, min_periods=1).std()
    out['stagnant_air_index']  = (out['PRES'] / (out['WSPM'] + 1)).clip(0, 2000)
    out['is_rush_hour']        = out['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    out['PM2.5_to_PM10_ratio'] = (out['PM2.5'] / (out['PM10'] + 0.001)).clip(0, 10)
    out['NO2_to_CO_ratio']     = (out['NO2']   / (out['CO']   + 0.001)).clip(0, 500)
    return out

# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
MODEL_CACHE_DIR = "./model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def get_cache_key(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:12]

@st.cache_resource
def train_models(df):
    cache_key = get_cache_key(df)
    targets = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    alpha_values = {'PM2.5': 0.88, 'PM10': 0.85, 'SO2': 0.45,
                    'NO2': 0.85, 'CO': 0.8, 'O3': 0.75}

    train_parts, test_parts = [], []
    for station in df['station'].unique():
        sdf = df[df['station'] == station].copy()
        sp  = int(len(sdf) * 0.9)
        train_parts.append(sdf.iloc[:sp])
        test_parts.append(sdf.iloc[sp:])

    train_raw = pd.concat(train_parts).sort_index()
    test_raw  = pd.concat(test_parts).sort_index()
    train_feat = add_features(train_raw).dropna()
    test_feat  = add_features(test_raw, fit_df=train_raw).dropna()
    train_enc  = pd.get_dummies(train_feat, columns=['wd', 'station'], drop_first=True)
    test_enc   = pd.get_dummies(test_feat,  columns=['wd', 'station'], drop_first=True)
    train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)

    X_train = train_enc.drop(columns=targets)
    y_train = train_enc[targets]
    X_test  = test_enc.drop(columns=targets)
    y_test  = test_enc[targets]

    models_to_train = {
        "LightGBM":         lgb.LGBMRegressor(objective='quantile', n_estimators=500,
                                               learning_rate=0.05, num_leaves=63,
                                               random_state=42, n_jobs=-1, verbose=-1),
        "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=12,
                                                   random_state=42, n_jobs=-1),
        "LinearRegression": Pipeline([('scaler', StandardScaler()),
                                      ('model',  LinearRegression(n_jobs=-1))])
    }

    trained_models = {}
    progress = st.progress(0, text="Loading models…")
    total_steps = len(models_to_train) * len(targets)
    step = 0

    for model_name, model_template in models_to_train.items():
        models_for_pollutants = {}
        for target in targets:
            cache_path = os.path.join(MODEL_CACHE_DIR, f"{cache_key}_{model_name}_{target}.pkl")
            if os.path.exists(cache_path):
                current_model = joblib.load(cache_path)
            else:
                current_model = clone(model_template)
                if model_name == "LightGBM":
                    current_model.set_params(alpha=alpha_values.get(target, 0.8))
                current_model.fit(X_train, y_train[target])
                joblib.dump(current_model, cache_path)
            models_for_pollutants[target] = current_model
            step += 1
            progress.progress(step / total_steps,
                               text=f"{model_name} / {target}  ({step}/{total_steps})")
        trained_models[model_name] = models_for_pollutants

    progress.empty()
    return trained_models, X_test, y_test, targets

# ─────────────────────────────────────────────
#  LOAD DATA & TRAIN
# ─────────────────────────────────────────────
df, station_names, dropped_rows = load_and_prep_data()
models, X_test, y_test, targets = train_models(df)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ap-sidebar-logo">AirPulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="ap-sidebar-sub">Air Quality Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="ap-sidebar-head">Station</div>', unsafe_allow_html=True)
    selected_station = st.selectbox("", station_names, label_visibility="collapsed")

    st.markdown('<div class="ap-sidebar-head">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ap-sidebar-info">
    Records &nbsp;<span>{len(df):,}</span><br>
    Stations &nbsp;<span>{len(station_names)}</span><br>
    Span &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>{df.index.min().year}–{df.index.max().year}</span><br>
    Dropped &nbsp;<span>{dropped_rows:,}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ap-sidebar-head">WHO Thresholds</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ap-sidebar-info">
    PM2.5 &nbsp;&nbsp;<span>150.4 µg/m³</span><br>
    NO2 &nbsp;&nbsp;&nbsp;&nbsp;<span>100 µg/m³</span><br>
    SO2 &nbsp;&nbsp;&nbsp;&nbsp;<span>80 µg/m³</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STATION FILTER
# ─────────────────────────────────────────────
station_col = f"station_{selected_station}"
if station_col in X_test.columns:
    mask = X_test[station_col] == 1
else:
    station_cols = [c for c in X_test.columns if c.startswith('station_')]
    mask = X_test[station_cols].sum(axis=1) == 0 if station_cols else pd.Series(True, index=X_test.index)

X_test_filtered = X_test[mask]
y_test_filtered = y_test[mask]

# ─────────────────────────────────────────────
#  HEADER  (self-contained block, no unclosed divs)
# ─────────────────────────────────────────────
selected_station = st.selectbox(
    "Station",
    station_names
)
station_data  = df[df['station'] == selected_station]
avg_pm25      = station_data['PM2.5'].mean()
avg_aqi_label = (
    "Hazardous" if avg_pm25 > 150 else
    "Unhealthy"  if avg_pm25 > 55  else
    "Moderate"   if avg_pm25 > 35  else
    "Good"
)

st.markdown(f"""
<div class="ap-header">
    <div class="ap-header-label">Beijing — Multi-Site Monitoring Network</div>
    <div class="ap-header-title">{selected_station}<em> / </em>{avg_aqi_label}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STAT INDEX ROW  (self-contained block)
# ─────────────────────────────────────────────
avg_no2  = station_data['NO2'].mean()
avg_so2  = station_data['SO2'].mean()
avg_o3   = station_data['O3'].mean()
avg_temp = station_data['TEMP'].mean()

st.markdown(f"""
<div class="ap-index">
    <div class="ap-index-cell">
        <div class="ap-index-label">PM2.5</div>
        <div class="ap-index-value">{avg_pm25:.0f}</div>
        <div class="ap-index-unit">µg/m³</div>
    </div>
    <div class="ap-index-cell">
        <div class="ap-index-label">NO2</div>
        <div class="ap-index-value">{avg_no2:.0f}</div>
        <div class="ap-index-unit">µg/m³</div>
    </div>
    <div class="ap-index-cell">
        <div class="ap-index-label">SO2</div>
        <div class="ap-index-value">{avg_so2:.0f}</div>
        <div class="ap-index-unit">µg/m³</div>
    </div>
    <div class="ap-index-cell">
        <div class="ap-index-label">O3</div>
        <div class="ap-index-value">{avg_o3:.0f}</div>
        <div class="ap-index-unit">µg/m³</div>
    </div>
    <div class="ap-index-cell">
        <div class="ap-index-label">Temp</div>
        <div class="ap-index-value">{avg_temp:.1f}°</div>
        <div class="ap-index-unit">Celsius</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["1-Year Risk Outlook", "Model Performance", "Model Interpretation"])

# ══════════════════════════════════════════════
#  TAB 1 — RISK OUTLOOK
# ══════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="ap-section">
        <span>Weekly Risk Outlook</span>
        <span>Historical exceedance &gt; 25%</span>
    </div>
    <p class="ap-body">
    Weeks flagged when historical exceedance exceeds 25% — more than 1 in 4 years recorded
    at least one unhealthy day. Climatological baseline, not a live forecast.
    </p>
    """, unsafe_allow_html=True)

    numeric_cols = station_data.select_dtypes(include=np.number).columns
    daily_data   = station_data[numeric_cols].resample('D').mean().dropna()
    unhealthy_thresholds = {'PM2.5': 150.4, 'NO2': 100, 'SO2': 80}
    daily_copy = daily_data.copy()
    for pollutant, threshold in unhealthy_thresholds.items():
        daily_copy[f'{pollutant}_unhealthy'] = daily_copy[pollutant] > threshold
    unhealthy_cols = [f'{p}_unhealthy' for p in unhealthy_thresholds]
    daily_copy['is_unhealthy_day'] = daily_copy[unhealthy_cols].any(axis=1)
    daily_copy['week_num']         = daily_copy.index.isocalendar().week.astype(int)
    weekly_risk = daily_copy.groupby('week_num')['is_unhealthy_day'].mean() * 100

    def get_high_pollutants(week_df):
        unhealthy = week_df[week_df['is_unhealthy_day']]
        if unhealthy.empty:
            return "—"
        flags = unhealthy[unhealthy_cols].any()
        return ', '.join(flags.index[flags].str.replace('_unhealthy', '', regex=False))

    pollutant_summary = daily_copy.groupby('week_num').apply(
        get_high_pollutants, include_groups=False
    )

    def week_to_month(week_num):
        try:
            d = datetime.date.fromisocalendar(datetime.date.today().year, int(week_num), 1)
            return d.strftime('%b')
        except Exception:
            return '?'

    current_week   = df.index.max().isocalendar().week
    upcoming_weeks = [((current_week + i - 1) % 52) + 1 for i in range(52)]
    outlook        = weekly_risk.reindex(upcoming_weeks)
    high_risk      = outlook[outlook > 25]

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Bar(
        x=list(weekly_risk.index),
        y=weekly_risk.values,
        marker_color=['#e8e8e8' if v > 25 else '#2a2a2a' for v in weekly_risk.values],
        hovertemplate='Week %{x}<br>%{y:.1f}%<extra></extra>'
    ))
    fig_risk.add_hline(y=25, line_dash='dot', line_color='#444444',
                       annotation_text='25%', annotation_font_size=9,
                       annotation_font_color='#666666')
    fig_risk.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=dict(text='Historical Weekly Exceedance Rate',
                   font=dict(size=11, color='#666666')),
        xaxis_title='ISO Week',
        yaxis_title='% Unhealthy Days',
        showlegend=False,
        height=280,
        bargap=0.3,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("""
    <div class="ap-section"><span>Flagged Weeks</span></div>
    """, unsafe_allow_html=True)

    if not high_risk.empty:
        rows_html = (
            '<div class="ap-table-head">'
            '<span>Week</span>'
            '<span>Month</span>'
            '<span>Risk</span>'
            '<span>Prob.</span>'
            '<span>Pollutants</span>'
            '<span style="text-align:right">Status</span>'
            '</div>'
        )
        for _, row in high_risk.reset_index().iterrows():
            wk         = int(row['week_num'])
            prob       = row['is_unhealthy_day']
            month      = week_to_month(wk)
            pollutants = pollutant_summary.get(wk, "—")
            bar_pct    = round(min(prob, 100), 1)
            label      = 'High' if prob > 50 else 'Elevated'
            status_cls = 'ap-col-status high' if prob > 50 else 'ap-col-status'
            rows_html += (
                '<div class="ap-table-row">'
                f'<div class="ap-col-wk">WK {wk:02d}</div>'
                f'<div class="ap-col-mo">{month}</div>'
                f'<div class="ap-col-bar">'
                f'<div class="ap-bar-track"><div class="ap-bar-fill" style="width:{bar_pct}%"></div></div>'
                f'</div>'
                f'<div class="ap-col-prob">{prob:.1f}%</div>'
                f'<div class="ap-col-pol">{pollutants}</div>'
                f'<div class="{status_cls}">{label}</div>'
                '</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="ap-notice">No high-risk weeks identified for this station.</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════
#  TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="ap-section"><span>Model Performance</span><span>NRMSE across pollutants</span></div>
    """, unsafe_allow_html=True)

    if X_test_filtered.empty:
        st.warning(f"No test data for '{selected_station}'.")
    else:
        all_metrics = []
        for model_name, pollutant_models in models.items():
            for target in targets:
                preds     = pollutant_models[target].predict(X_test_filtered)
                mae       = mean_absolute_error(y_test_filtered[target], preds)
                rmse      = np.sqrt(mean_squared_error(y_test_filtered[target], preds))
                val_range = y_test_filtered[target].max() - y_test_filtered[target].min()
                nrmse     = (rmse / val_range * 100) if val_range > 0 else 0
                all_metrics.append({
                    'Model': model_name, 'Pollutant': target,
                    'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'NRMSE (%)': round(nrmse, 2)
                })
        metrics_df = pd.DataFrame(all_metrics)

        fig_bar = px.bar(
            metrics_df, x='Pollutant', y='NRMSE (%)', color='Model',
            barmode='group',
            color_discrete_sequence=['#e8e8e8', '#666666', '#444444'],
            title='NRMSE by Model & Pollutant'
        )
        fig_bar.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            height=320,
            bargap=0.25,
            title=dict(text='NRMSE by Model & Pollutant',
                       font=dict(size=11, color='#666666'))
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("Full metrics table"):
            st.dataframe(
                metrics_df.style
                    .format({'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'NRMSE (%)': '{:.2f}%'})
                    .background_gradient(subset=['NRMSE (%)'], cmap='Greys'),
                hide_index=True,
                use_container_width=True
            )

        st.markdown("""
        <div class="ap-section"><span>Predictions vs Actuals</span></div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            pollutant_sel = st.selectbox("Pollutant", targets, key='p2')
        with col2:
            model_sel = st.selectbox("Model", list(models.keys()), key='m2')

        preds_plot = models[model_sel][pollutant_sel].predict(X_test_filtered)
        fig_pred   = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=y_test_filtered.index, y=y_test_filtered[pollutant_sel],
            mode='lines', name='Actual',
            line=dict(color='#e8e8e8', width=1)
        ))
        fig_pred.add_trace(go.Scatter(
            x=y_test_filtered.index, y=preds_plot,
            mode='lines', name='Predicted',
            line=dict(color='#555555', width=1, dash='dot')
        ))
        fig_pred.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            title=dict(text=f'{model_sel} · {pollutant_sel}',
                       font=dict(size=11, color='#666666')),
            yaxis_title=f'{pollutant_sel} µg/m³',
            height=360,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        residuals = y_test_filtered[pollutant_sel].values - preds_plot
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Histogram(
            x=residuals, nbinsx=60,
            marker_color='#444444', opacity=1,
            hovertemplate='%{x:.1f}<br>%{y}<extra></extra>'
        ))
        fig_resid.add_vline(x=0, line_dash='dot', line_color='#888888')
        fig_resid.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            title=dict(text=f'Residuals · {model_sel} · {pollutant_sel}',
                       font=dict(size=11, color='#666666')),
            xaxis_title='Actual − Predicted',
            yaxis_title='Count',
            height=240,
            bargap=0.05,
        )
        st.plotly_chart(fig_resid, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 3 — SHAP
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="ap-section"><span>Feature Importance</span><span>SHAP / LightGBM</span></div>
    <p class="ap-body">
    SHAP values attribute each prediction to individual input features.
    Red = high feature value, Blue = low. Position right = pushed prediction higher.
    </p>
    """, unsafe_allow_html=True)

    if X_test_filtered.empty:
        st.warning(f"No test data for '{selected_station}'.")
    else:
        pollutant_shap = st.selectbox("Pollutant", ('PM2.5', 'NO2', 'SO2', 'O3'), key='shap_sel')

        with st.expander(f"Generate SHAP plot — {pollutant_shap}"):
            with st.spinner("Computing SHAP values…"):
                lgbm_model  = models['LightGBM'][pollutant_shap]
                sample      = X_test_filtered.sample(min(500, len(X_test_filtered)), random_state=42)
                explainer   = shap.TreeExplainer(lgbm_model)
                shap_values = explainer.shap_values(sample)

                plt.rcParams.update({
                    'figure.facecolor': '#121212',
                    'axes.facecolor':   '#121212',
                    'text.color':       '#666666',
                    'axes.labelcolor':  '#666666',
                    'xtick.color':      '#444444',
                    'ytick.color':      '#444444',
                    'font.family':      'sans-serif',
                    'font.size':        10,
                })
                fig_shap, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
                fig_shap.patch.set_facecolor('#121212')
                ax.set_facecolor('#121212')
                shap.summary_plot(shap_values, sample, show=False,
                                  max_display=12, plot_size=None,
                                  color_bar_label='Feature Value')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2a2a2a')
                st.pyplot(fig_shap)
                plt.close(fig_shap)