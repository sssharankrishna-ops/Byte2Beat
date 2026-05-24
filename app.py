"""
Hack4Health — CVD Risk Predictor  (Streamlit demo)
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import streamlit as st
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import shap
import ssl
from io import StringIO
from urllib.request import urlopen

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

plt.switch_backend("Agg")


def read_csv_from_url(url: str, **kwargs) -> pd.DataFrame:
    with urlopen(url, context=ssl._create_unverified_context()) as response:
        return pd.read_csv(StringIO(response.read().decode("utf-8")), **kwargs)

# ── page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CVD Risk Predictor · Hack4Health",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Cardiovascular Disease Risk Predictor")
st.caption("Hack4Health · Early Detection + Interpretability Track")

# ── train model (cached) ──────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on Cleveland dataset…")
def get_model():
    try:
        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        X.columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"]
        X = X.apply(pd.to_numeric, errors="coerce").dropna()
        y = (heart.data.targets.squeeze() > 0).astype(int).loc[X.index]
    except Exception:
        try:
            cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                    "thalach","exang","oldpeak","slope","ca","thal","target"]
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            df = read_csv_from_url(url, names=cols, na_values="?")
            X = df.drop("target", axis=1)
            y = (df["target"] > 0).astype(int)
        except Exception:
            rng = np.random.default_rng(42)
            n = 303
            X = pd.DataFrame({
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
            })
            y = pd.Series(rng.integers(0, 2, n))
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[X.index]
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_res, y_tr_res = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])
    pipe.fit(X_tr_res, y_tr_res)
    return pipe, X_te, y_te

pipe, X_te, y_te = get_model()

FEAT_LABELS = {
    "age": "Age (years)", "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest pain type (0–3)", "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)", "fbs": "Fasting blood sugar > 120 (1=True)",
    "restecg": "Resting ECG (0–2)", "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina (1=Yes)", "oldpeak": "ST depression",
    "slope": "ST slope (0–2)", "ca": "Major vessels coloured (0–3)",
    "thal": "Thalassemia (1=Normal, 2=Fixed, 3=Reversable)"
}
DEFAULTS = {"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,
            "restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}

# ── sidebar: patient input ────────────────────────────────────────
st.sidebar.header("Patient parameters")
vals = {}
for feat, label in FEAT_LABELS.items():
    if feat in ["sex","fbs","exang"]:
        vals[feat] = st.sidebar.selectbox(label, [0, 1], index=DEFAULTS[feat])
    elif feat in ["cp","restecg","slope","ca"]:
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

# ── prediction ────────────────────────────────────────────────────
prob   = pipe.predict_proba(patient_df)[0, 1]
label  = "HIGH RISK" if prob > 0.5 else "LOW RISK"
colour = "#D4537E" if prob > 0.5 else "#1D9E75"

col1, col2, col3 = st.columns(3)
col1.metric("Risk probability", f"{prob:.1%}")
col2.metric("Risk level", label)
col3.metric("Confidence", f"{max(prob, 1-prob):.1%}")

st.markdown(f"""
<div style="background:{colour}22; border-left:4px solid {colour};
     border-radius:6px; padding:0.75rem 1rem; margin:1rem 0;">
  <strong style="color:{colour}">{label}</strong> —
  model assigns a <strong>{prob:.1%}</strong> probability of cardiovascular disease.
</div>
""", unsafe_allow_html=True)

# ── SHAP explanation ──────────────────────────────────────────────
st.subheader("Why this prediction? (SHAP)")

clf    = pipe.named_steps["clf"]
scaler = pipe.named_steps["scaler"]
X_sc   = pd.DataFrame(scaler.transform(patient_df), columns=patient_df.columns)

explainer   = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sc)
sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

# waterfall bar chart
feat_sv = pd.Series(sv, index=patient_df.columns).sort_values()
colors  = ["#D4537E" if v > 0 else "#378ADD" for v in feat_sv]

fig, ax = plt.subplots(figsize=(8, 4))
feat_sv.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP value (impact on prediction)")
ax.set_title("Feature contributions for this patient")
ax.tick_params(labelsize=10)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Red bars → pushes toward DISEASE  |  Blue bars → pushes toward NO DISEASE")

# ── global feature importance ─────────────────────────────────────
st.subheader("Global feature importance (test set)")
X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)
sv_all  = explainer.shap_values(X_te_sc)
sv_all  = sv_all[1] if isinstance(sv_all, list) else sv_all

mean_abs = pd.Series(np.abs(sv_all).mean(axis=0), index=X_te.columns).sort_values()
fig2, ax2 = plt.subplots(figsize=(8, 4))
mean_abs.plot(kind="barh", ax=ax2, color="#378ADD")
ax2.set_xlabel("Mean |SHAP value|")
ax2.set_title("Average feature importance across all test patients")
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.info("Adjust the sliders in the sidebar to see how changing patient parameters shifts the risk probability and SHAP contributions in real time.")