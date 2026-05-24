"""
Hack4Health — CVD Risk Predictor  (Streamlit demo)
Run: streamlit run app.py
"""

import warnings

warnings.filterwarnings("ignore")

import os
import ssl
from io import StringIO
from urllib.request import urlopen

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import shap
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

plt.switch_backend("Agg")

st.set_page_config(
    page_title="CVD Risk Predictor · Hack4Health",
    page_icon="🫀",
    layout="wide",
)


def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    if value:
        return str(value)
    return os.getenv(name.upper(), default)


AUTH_USERNAME = _secret("auth_username")
AUTH_PASSWORD = _secret("auth_password")


TECH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #07111f;
    --panel: rgba(8, 18, 33, 0.72);
    --panel-strong: rgba(12, 24, 43, 0.92);
    --line: rgba(120, 190, 255, 0.18);
    --text: #eaf4ff;
    --muted: #9bb4cc;
    --cyan: #4ed0ff;
    --blue: #6b8bff;
    --teal: #2de2c5;
    --pink: #ff6f91;
    --amber: #ffcb6b;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

stApp {
    background:
        radial-gradient(circle at top left, rgba(78, 208, 255, 0.16), transparent 35%),
        radial-gradient(circle at top right, rgba(107, 139, 255, 0.18), transparent 32%),
        linear-gradient(180deg, #07111f 0%, #081427 48%, #060b14 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(6, 13, 25, 0.98), rgba(8, 20, 37, 0.96));
    border-right: 1px solid rgba(120, 190, 255, 0.12);
}

.hero-shell {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(13, 28, 50, 0.94), rgba(8, 17, 31, 0.90));
    border: 1px solid rgba(120, 190, 255, 0.16);
    border-radius: 26px;
    padding: 1.45rem 1.6rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
}

.hero-shell:before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 0%, rgba(78, 208, 255, 0.10) 25%, transparent 55%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.04em;
    margin: 0;
    line-height: 1.02;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 0.98rem;
    margin: 0.35rem 0 1rem 0;
    max-width: 58rem;
}

.microchip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    padding: 0.45rem 0.75rem;
    border: 1px solid rgba(120, 190, 255, 0.18);
    border-radius: 999px;
    color: var(--text);
    background: rgba(255, 255, 255, 0.03);
    font-size: 0.82rem;
    backdrop-filter: blur(10px);
}

.chip span {
    color: var(--muted);
}

.panel {
    background: var(--panel);
    border: 1px solid rgba(120, 190, 255, 0.14);
    border-radius: 22px;
    padding: 1rem 1.05rem;
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18);
    backdrop-filter: blur(14px);
}

.panel-strong {
    background: var(--panel-strong);
}

.section-kicker {
    color: var(--cyan);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.section-copy {
    color: var(--muted);
    margin: 0.35rem 0 0 0;
    line-height: 1.6;
}

.signal-box {
    background: linear-gradient(135deg, rgba(78, 208, 255, 0.12), rgba(45, 226, 197, 0.08));
    border: 1px solid rgba(78, 208, 255, 0.24);
    border-radius: 18px;
    padding: 1rem;
}

.signal-label {
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.signal-value {
    color: var(--text);
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.05;
}

.signal-note {
    color: var(--muted);
    font-size: 0.9rem;
}

.risk-track {
    width: 100%;
    height: 14px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.7rem;
}

.risk-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--teal), var(--cyan), var(--blue), var(--pink));
}

.detail-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(120, 190, 255, 0.12);
    border-radius: 16px;
    padding: 0.85rem;
}

.detail-name {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.detail-val {
    color: var(--text);
    font-size: 1rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

.footer-note {
    color: var(--muted);
    font-size: 0.82rem;
    text-align: center;
    margin-top: 0.5rem;
}
</style>
"""


def read_csv_from_url(url: str, **kwargs) -> pd.DataFrame:
    with urlopen(url, context=ssl._create_unverified_context()) as response:
        return pd.read_csv(StringIO(response.read().decode("utf-8")), **kwargs)


def make_metric_card(label: str, value: str, accent: str, note: str) -> str:
    return f"""
    <div class="panel panel-strong" style="height:100%;">
      <div class="signal-label">{label}</div>
      <div class="signal-value" style="color:{accent};">{value}</div>
      <div class="signal-note">{note}</div>
    </div>
    """


def auth_gate() -> None:
    if st.session_state.get("app_authenticated"):
        return

    st.markdown(
        """
        <div class="hero-shell">
          <div class="section-kicker">Secure access layer</div>
          <h1 class="hero-title">Protected clinical dashboard</h1>
          <p class="hero-subtitle">This deployment uses an app-level login gate to reduce accidental access to the model and patient-style controls. Configure credentials in Streamlit secrets or environment variables before publishing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not AUTH_USERNAME or not AUTH_PASSWORD:
        st.error(
            "Authentication is not configured. Set auth_username and auth_password in Streamlit secrets or APP_AUTH_USERNAME and APP_AUTH_PASSWORD in the environment."
        )
        st.stop()

    with st.form("login_form", clear_on_submit=False):
        st.subheader("Unlock access")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Unlock dashboard")

    if submitted:
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            st.session_state["app_authenticated"] = True
            st.success("Access granted. Loading the dashboard...")
            st.rerun()
        else:
            st.error("Invalid credentials.")

    st.info("Authentication required before the model and explanations load.")
    st.stop()


st.markdown(TECH_CSS, unsafe_allow_html=True)

auth_gate()

st.markdown(
    """
    <div class="hero-shell">
      <div class="section-kicker">Hack4Health · Neural Risk Console</div>
      <h1 class="hero-title">Cardiovascular Disease Risk Predictor</h1>
      <p class="hero-subtitle">An interactive clinical dashboard that converts patient signals into a risk score, then explains the model's reasoning with SHAP so the output feels like a live diagnostic instrument instead of a plain form.</p>
      <div class="microchip-row">
        <div class="chip">🛰️ <strong>Live inference</strong> <span>from sidebar inputs</span></div>
        <div class="chip">🔬 <strong>Explainable AI</strong> <span>feature-level SHAP impact</span></div>
        <div class="chip">⚡ <strong>Fast preview</strong> <span>cached model training</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


@st.cache_resource(show_spinner="Training model on Cleveland dataset…")
def get_model():
    try:
        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        X.columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
        X = X.apply(pd.to_numeric, errors="coerce").dropna()
        y = (heart.data.targets.squeeze() > 0).astype(int).loc[X.index]
    except Exception:
        try:
            cols = [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
                "target",
            ]
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            df = read_csv_from_url(url, names=cols, na_values="?")
            X = df.drop("target", axis=1)
            y = (df["target"] > 0).astype(int)
        except Exception:
            rng = np.random.default_rng(42)
            n = 303
            X = pd.DataFrame(
                {
                    "age": rng.integers(29, 78, n),
                    "sex": rng.integers(0, 2, n),
                    "cp": rng.integers(0, 4, n),
                    "trestbps": rng.integers(94, 200, n),
                    "chol": rng.integers(126, 564, n),
                    "fbs": rng.integers(0, 2, n),
                    "restecg": rng.integers(0, 3, n),
                    "thalach": rng.integers(70, 205, n),
                    "exang": rng.integers(0, 2, n),
                    "oldpeak": rng.uniform(0, 6.5, n).round(1),
                    "slope": rng.integers(0, 3, n),
                    "ca": rng.integers(0, 4, n),
                    "thal": rng.integers(0, 4, n),
                }
            )
            y = pd.Series(rng.integers(0, 2, n))

    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[X.index]
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr_res, y_tr_res = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ]
    )
    pipe.fit(X_tr_res, y_tr_res)
    return pipe, X_te, y_te


pipe, X_te, y_te = get_model()


FEAT_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest pain type (0–3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 (1=True)",
    "restecg": "Resting ECG (0–2)",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina (1=Yes)",
    "oldpeak": "ST depression",
    "slope": "ST slope (0–2)",
    "ca": "Major vessels coloured (0–3)",
    "thal": "Thalassemia (1=Normal, 2=Fixed, 3=Reversable)",
}
DEFAULTS = {
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}


st.sidebar.markdown("## Patient control deck")
st.sidebar.caption("Tune the scan inputs to simulate a different patient profile.")
vals = {}
for feat, label in FEAT_LABELS.items():
    if feat in ["sex", "fbs", "exang"]:
        vals[feat] = st.sidebar.selectbox(label, [0, 1], index=DEFAULTS[feat])
    elif feat in ["cp", "restecg", "slope", "ca"]:
        vals[feat] = st.sidebar.slider(label, 0, 3, DEFAULTS[feat])
    elif feat == "thal":
        vals[feat] = st.sidebar.slider(label, 1, 3, DEFAULTS[feat])
    elif feat == "oldpeak":
        vals[feat] = st.sidebar.slider(label, 0.0, 6.2, float(DEFAULTS[feat]), step=0.1)
    elif feat == "age":
        vals[feat] = st.sidebar.slider(label, 20, 80, DEFAULTS[feat])
    elif feat == "trestbps":
        vals[feat] = st.sidebar.slider(label, 90, 200, DEFAULTS[feat])
    elif feat == "chol":
        vals[feat] = st.sidebar.slider(label, 100, 600, DEFAULTS[feat])
    elif feat == "thalach":
        vals[feat] = st.sidebar.slider(label, 60, 210, DEFAULTS[feat])

patient_df = pd.DataFrame([vals])


prob = pipe.predict_proba(patient_df)[0, 1]
label = "HIGH RISK" if prob > 0.5 else "LOW RISK"
colour = "#D4537E" if prob > 0.5 else "#1D9E75"

st.markdown(
    f"""
    <div class="signal-box panel">
      <div class="signal-label">Risk telemetry</div>
      <div class="signal-value">{prob:.1%}</div>
      <div class="signal-note">The classifier currently maps this profile to <strong>{label}</strong>.</div>
      <div class="risk-track">
        <div class="risk-fill" style="width:{prob:.0%}"></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Risk probability", f"{prob:.1%}")
metric_b.metric("Risk level", label)
metric_c.metric("Confidence", f"{max(prob, 1-prob):.1%}")

detail_a, detail_b = st.columns(2)
with detail_a:
    st.markdown(
        make_metric_card(
            "Signal state",
            label,
            colour,
            "The model status updates live as you adjust the sliders.",
        ),
        unsafe_allow_html=True,
    )
with detail_b:
    st.markdown(
        make_metric_card(
            "Decision confidence",
            f"{max(prob, 1-prob):.1%}",
            "#4ed0ff",
            "A sharper score means the classifier is less ambiguous.",
        ),
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="panel" style="margin: 1rem 0 0 0;">
      <strong style="color:{colour}">{label}</strong> — model assigns a <strong>{prob:.1%}</strong> probability of cardiovascular disease.
    </div>
    """,
    unsafe_allow_html=True,
)


st.subheader("Why this prediction? (SHAP)")

clf = pipe.named_steps["clf"]
scaler = pipe.named_steps["scaler"]
X_sc = pd.DataFrame(scaler.transform(patient_df), columns=patient_df.columns)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sc)
sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

shap_left, shap_right = st.columns([1.15, 0.85])
with shap_right:
    st.markdown(
        """
        <div class="panel panel-strong">
          <div class="section-kicker">Patient fingerprint</div>
          <p class="section-copy">Red bars push the prediction toward disease. Blue bars pull it back toward the low-risk class. This keeps the model explainable instead of opaque.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

feat_sv = pd.Series(sv, index=patient_df.columns).sort_values()
colors = ["#D4537E" if v > 0 else "#378ADD" for v in feat_sv]

fig, ax = plt.subplots(figsize=(8.2, 4.2))
feat_sv.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP value (impact on prediction)")
ax.set_title("Feature contributions for this patient")
ax.tick_params(labelsize=10)
plt.tight_layout()
with shap_left:
    st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Red bars → pushes toward DISEASE  |  Blue bars → pushes toward NO DISEASE")


st.subheader("Global feature importance (test set)")
X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)
sv_all = explainer.shap_values(X_te_sc)
sv_all = sv_all[1] if isinstance(sv_all, list) else sv_all

mean_abs = pd.Series(np.abs(sv_all).mean(axis=0), index=X_te.columns).sort_values()
fig2, ax2 = plt.subplots(figsize=(8.2, 4.2))
mean_abs.plot(kind="barh", ax=ax2, color="#378ADD")
ax2.set_xlabel("Mean |SHAP value|")
ax2.set_title("Average feature importance across all test patients")
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)


summary_df = pd.DataFrame(
    {
        "Signal": ["Age", "Rest BP", "Cholesterol", "Max HR", "ST Depression", "Chest Pain"],
        "Value": [
            vals["age"],
            vals["trestbps"],
            vals["chol"],
            vals["thalach"],
            vals["oldpeak"],
            f"Type {vals['cp']}",
        ],
    }
)

st.markdown(
    """
    <div class="panel">
      <div class="section-kicker">Scan summary</div>
      <p class="section-copy">The summary below exposes the exact inputs driving this prediction so the interface feels like a control room instead of a basic form.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.dataframe(summary_df, use_container_width=True, hide_index=True)

summary_left, summary_right = st.columns(2)
with summary_left:
    st.markdown(
        f"""
        <div class="detail-card">
          <div class="detail-name">Most likely state</div>
          <div class="detail-val" style="color:{colour};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with summary_right:
    st.markdown(
        f"""
        <div class="detail-card">
          <div class="detail-name">Current confidence</div>
          <div class="detail-val">{max(prob, 1-prob):.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='footer-note'>Adjust the sidebar sliders to re-run the clinical signal chain and watch the risk meter, cards, and SHAP bars reconfigure in real time.</div>",
    unsafe_allow_html=True,
)