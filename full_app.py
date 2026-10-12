"""
╔══════════════════════════════════════════════════════════════════════╗
║          MEDICAL COST PREDICTION — FULL APPLICATION                  ║
║          Single-file: Backend (Flask) + Frontend (Streamlit)         ║
╠══════════════════════════════════════════════════════════════════════╣
║  HOW TO RUN (two terminals):                                          ║
║                                                                       ║
║  Terminal 1 — train model (once):                                     ║
║      python full_app.py --train                                       ║
║                                                                       ║
║  Terminal 1 — start Flask API:                                        ║
║      python full_app.py --backend                                     ║
║                                                                       ║
║  Terminal 2 — launch Streamlit UI:                                    ║
║      python full_app.py --frontend                                    ║
║                                                                       ║
║  OR all-in-one (auto-starts Flask in background thread):              ║
║      python full_app.py --all                                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import pickle
import subprocess
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "insurance.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
API_PORT   = 5000
API_BASE   = f"http://127.0.0.1:{API_PORT}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def run_training(force: bool = False) -> None:
    """EDA, feature engineering, train 3 models, save best as model.pkl."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if os.path.exists(MODEL_PATH) and not force:
        print("✅  model.pkl already exists. Use --train --force to retrain.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading data …")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    # ── EDA plots ─────────────────────────────────────────────────────────────
    print("Generating EDA plots …")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["charges"], bins=40, kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribution of Medical Charges")
    ax.set_xlabel("Charges (USD)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "charges_distribution.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="smoker", y="charges", palette="Set2", ax=ax)
    ax.set_title("Charges by Smoker Status")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "charges_by_smoker.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for s, grp in df.groupby("smoker"):
        ax.scatter(grp["bmi"], grp["charges"], alpha=0.4, label=s)
    ax.set_xlabel("BMI"); ax.set_ylabel("Charges (USD)")
    ax.set_title("BMI vs Charges"); ax.legend(title="Smoker")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bmi_vs_charges.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    num_df = df.select_dtypes(include=[np.number])
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)

    print("  Plots saved to ./plots/")

    # ── Feature engineering ───────────────────────────────────────────────────
    df["age_squared"]  = df["age"] ** 2
    df["bmi_obese"]    = (df["bmi"] >= 30).astype(int)
    df["smoker_obese"] = ((df["smoker"] == "yes") & (df["bmi"] >= 30)).astype(int)

    NUM_FEATURES = ["age", "bmi", "children", "age_squared"]
    CAT_FEATURES = ["sex", "smoker", "region"]
    TARGET       = "charges"

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_FEATURES),
    ])

    # ── Train & evaluate ──────────────────────────────────────────────────────
    models = {
        "Ridge":            Ridge(alpha=10),
        "RandomForest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(
                                n_estimators=300, learning_rate=0.05,
                                max_depth=4, random_state=42),
    }

    results = {}
    print("\nTraining models …")
    for name, mdl in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", mdl)])
        cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2    = r2_score(y_test, preds)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        results[name] = {"pipe": pipe, "r2": r2, "mae": mae,
                         "rmse": rmse, "cv_r2": cv.mean()}
        print(f"  {name:20s}  R²={r2:.4f}  MAE=${mae:,.0f}  "
              f"RMSE=${rmse:,.0f}  CV-R²={cv.mean():.4f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    best_pipe = results[best_name]["pipe"]
    print(f"\nBest model: {best_name}  (R²={results[best_name]['r2']:.4f})")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":        best_pipe,
            "num_features": NUM_FEATURES,
            "cat_features": CAT_FEATURES,
            "model_name":   best_name,
            "metrics": {k: {m: v for m, v in v.items() if m != "pipe"}
                        for k, v in results.items()},
        }, f)
    print(f"Model saved → {MODEL_PATH}")

    # feature importance plot
    try:
        enc       = best_pipe.named_steps["pre"].named_transformers_["cat"]
        cat_cols  = list(enc.get_feature_names_out(CAT_FEATURES))
        feat_names = NUM_FEATURES + cat_cols
        fi = best_pipe.named_steps["model"].feature_importances_
        fi_df = pd.DataFrame({"feature": feat_names, "importance": fi})
        fi_df = fi_df.sort_values("importance", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=fi_df, x="importance", y="feature", palette="Blues_r", ax=ax)
        ax.set_title(f"Feature Importances — {best_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=120)
        plt.close(fig)
        print("Feature importance plot saved.")
    except AttributeError:
        pass

    print("\nTraining complete.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FLASK BACKEND
# ═════════════════════════════════════════════════════════════════════════════
def create_flask_app():
    """Build and return the Flask application object."""
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    _bundle = {}

    def get_bundle():
        if "model" not in _bundle:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    "model.pkl not found. Run: python full_app.py --train"
                )
            with open(MODEL_PATH, "rb") as f:
                _bundle.update(pickle.load(f))
        return _bundle

    VALID = {
        "sex":    {"male", "female"},
        "smoker": {"yes", "no"},
        "region": {"northeast", "northwest", "southeast", "southwest"},
    }

    def validate(data):
        try:
            age      = float(data["age"])
            bmi      = float(data["bmi"])
            children = int(data["children"])
            sex      = str(data["sex"]).lower().strip()
            smoker   = str(data["smoker"]).lower().strip()
            region   = str(data["region"]).lower().strip()
        except (KeyError, ValueError) as e:
            return None, f"Missing/invalid field: {e}"
        if not (0 < age <= 120):  return None, "age must be 1–120"
        if not (10 <= bmi <= 70): return None, "bmi must be 10–70"
        if not (0 <= children <= 10): return None, "children must be 0–10"
        if sex    not in VALID["sex"]:    return None, f"sex must be {VALID['sex']}"
        if smoker not in VALID["smoker"]: return None, f"smoker must be {VALID['smoker']}"
        if region not in VALID["region"]: return None, f"region must be {VALID['region']}"
        return {"age": age, "bmi": bmi, "children": children,
                "sex": sex, "smoker": smoker, "region": region,
                "age_squared": age ** 2}, None

    def do_predict(c):
        import pandas as pd
        b = get_bundle()
        row = {k: c[k] for k in b["num_features"] + b["cat_features"]}
        charge = float(b["model"].predict(pd.DataFrame([row]))[0])
        return {"predicted_charge": round(max(charge, 0), 2),
                "model_used": b["model_name"]}

    @app.route("/health")
    def health():
        try:
            b = get_bundle()
            return jsonify({"status": "ok", "model": b["model_name"],
                            "metrics": b["metrics"][b["model_name"]]}), 200
        except FileNotFoundError as e:
            return jsonify({"status": "error", "message": str(e)}), 503

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "JSON body required"}), 400
        cleaned, err = validate(payload)
        if err:
            return jsonify({"error": err}), 422
        try:
            res = do_predict(cleaned)
            res["input"] = {k: v for k, v in cleaned.items() if k != "age_squared"}
            return jsonify(res), 200
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        payload = request.get_json(silent=True)
        if not isinstance(payload, list):
            return jsonify({"error": "JSON array required"}), 400
        out = []
        for i, item in enumerate(payload):
            c, err = validate(item)
            if err:
                out.append({"index": i, "error": err})
            else:
                try:
                    r = do_predict(c); r["index"] = i; out.append(r)
                except Exception as e:
                    out.append({"index": i, "error": str(e)})
        return jsonify(out), 200

    return app


def run_backend(host="0.0.0.0", port=API_PORT, debug=False) -> None:
    """Start the Flask development server (blocking)."""
    app = create_flask_app()
    print(f"Starting Flask API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def start_backend_thread() -> None:
    """Launch Flask in a background daemon thread; wait until ready."""
    import urllib.request, urllib.error

    app = create_flask_app()

    def _run():
        app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{API_BASE}/health", timeout=2)
            print("✅  Flask API is ready.")
            return
        except Exception:
            time.sleep(0.5)
    print("⚠️   Flask API did not respond in 30 s.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STREAMLIT FRONTEND
# ═════════════════════════════════════════════════════════════════════════════
def run_frontend() -> None:
    """Launch Streamlit (blocks until user closes it)."""
    # Write a tiny bootstrap script so Streamlit can import this module
    bootstrap = os.path.join(BASE_DIR, "_st_bootstrap.py")
    with open(bootstrap, "w") as f:
        f.write(f"""
import sys, os
sys.path.insert(0, r"{BASE_DIR}")
from full_app import streamlit_main
streamlit_main()
""")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", bootstrap,
             "--server.headless", "false",
             "--browser.gatherUsageStats", "false"],
            cwd=BASE_DIR,
        )
    finally:
        if os.path.exists(bootstrap):
            os.remove(bootstrap)


def streamlit_main() -> None:
    """All Streamlit UI code — called from the bootstrap script."""
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import requests
    import streamlit as st

    st.set_page_config(
        page_title="Medical Cost Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .main-header{font-size:2.4rem;font-weight:700;color:#1e3a5f;text-align:center;margin-bottom:.2rem}
        .sub-header{font-size:1rem;color:#555;text-align:center;margin-bottom:1.5rem}
        .prediction-box{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:12px;
            padding:1.5rem;text-align:center;border:2px solid #4caf50}
        .prediction-amount{font-size:2.8rem;font-weight:800;color:#1b5e20}
        .risk-high{color:#c62828;font-weight:700}
        .risk-medium{color:#e65100;font-weight:700}
        .risk-low{color:#2e7d32;font-weight:700}
    </style>""", unsafe_allow_html=True)

    # ── helpers ───────────────────────────────────────────────────────────────
    @st.cache_data
    def load_dataset():
        return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

    def call_api(endpoint, payload):
        try:
            r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
            if r.status_code == 200:
                return r.json()
            st.error(f"API error {r.status_code}: {r.json().get('error', r.text)}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach Flask API. Start it with: python full_app.py --backend")
        except requests.exceptions.Timeout:
            st.error("⏱ Request timed out.")
        return None

    def health_check():
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def risk_label(c):
        if c < 8_000:   return "Low Risk",    "risk-low"
        if c < 20_000:  return "Medium Risk",  "risk-medium"
        return "High Risk", "risk-high"

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏥 Medical Cost Predictor")
        st.markdown("---")
        health = health_check()
        if health:
            st.success(f"✅ API Online — {health.get('model','model')}")
        else:
            st.error("❌ API Offline")
            st.info("Start the backend:\n```\npython full_app.py --backend\n```")
        st.markdown("---")
        st.markdown("### 📋 Input Features")
        age      = st.slider("Age", 18, 64, 30)
        sex      = st.selectbox("Sex", ["male", "female"])
        bmi      = st.number_input("BMI", 10.0, 60.0, 26.5, 0.1, format="%.1f")
        children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
        smoker   = st.radio("Smoker", ["no", "yes"], horizontal=True)
        region   = st.selectbox("Region",
                                ["northeast", "northwest", "southeast", "southwest"])
        st.markdown("---")
        predict_btn = st.button("🔮 Predict Cost", use_container_width=True,
                                type="primary")

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="main-header">🏥 Medical Cost Prediction</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict insurance charges using machine '
                'learning on the Medical Cost Personal Dataset</div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["🔮 Predictor", "📊 Data Exploration",
                    "📈 Model Insights", "⚙️ Batch Predict"])

    # TAB 1 — PREDICTOR ────────────────────────────────────────────────────────
    with tabs[0]:
        if predict_btn:
            result = call_api("/predict", {
                "age": age, "sex": sex, "bmi": bmi,
                "children": children, "smoker": smoker, "region": region,
            })
            if result:
                charge = result["predicted_charge"]
                label, css = risk_label(charge)
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.markdown(f"""
                    <div class="prediction-box">
                      <div style="font-size:1.1rem;color:#555;margin-bottom:.4rem">
                          Estimated Annual Medical Cost</div>
                      <div class="prediction-amount">${charge:,.2f}</div>
                      <div style="margin-top:.5rem;font-size:1.1rem">
                          Risk Level: <span class="{css}">{label}</span></div>
                      <div style="font-size:.85rem;color:#666;margin-top:.4rem">
                          Model: {result["model_used"]}</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown("#### Your Profile Summary")
                    st.dataframe(pd.DataFrame({
                        "Field": ["Age","Sex","BMI","Children","Smoker","Region"],
                        "Value": [age, sex.title(), f"{bmi:.1f}",
                                  children, smoker.title(), region.title()],
                    }), use_container_width=True, hide_index=True)
                # gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=charge,
                    title={"text": "Predicted Charge (USD)"},
                    delta={"reference": 13_270, "valueformat": ",.0f"},
                    gauge={
                        "axis": {"range": [0, 65_000]},
                        "bar":  {"color": "#3b82f6"},
                        "steps": [
                            {"range": [0, 8_000],      "color": "#c8e6c9"},
                            {"range": [8_000, 20_000],  "color": "#fff9c4"},
                            {"range": [20_000, 65_000], "color": "#ffccbc"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3},
                                      "thickness": 0.75, "value": 13_270},
                    }))
                fig_g.update_layout(height=280, margin=dict(t=40, b=0))
                st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.info("👈 Fill in the sidebar and click **Predict Cost**.")
            df = load_dataset()
            if df is not None:
                st.markdown("#### Sample from Dataset")
                st.dataframe(df.head(8), use_container_width=True)

    # TAB 2 — DATA EXPLORATION ─────────────────────────────────────────────────
    with tabs[1]:
        df = load_dataset()
        if df is None:
            st.warning("insurance.csv not found.")
        else:
            st.markdown("### Dataset Overview")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("Avg Charge",    f"${df['charges'].mean():,.0f}")
            c3.metric("Smokers",       f"{(df['smoker']=='yes').sum()}")
            c4.metric("Avg BMI",       f"{df['bmi'].mean():.1f}")
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.histogram(df, x="charges", nbins=50,
                                 title="Distribution of Charges",
                                 color_discrete_sequence=["#3b82f6"]),
                    use_container_width=True)
                st.plotly_chart(
                    px.box(df, x="region", y="charges", color="smoker",
                           title="Charges by Region & Smoker",
                           color_discrete_map={"yes":"#ef4444","no":"#22c55e"}),
                    use_container_width=True)
            with c2:
                st.plotly_chart(
                    px.scatter(df, x="bmi", y="charges", color="smoker",
                               title="BMI vs Charges",
                               color_discrete_map={"yes":"#ef4444","no":"#22c55e"},
                               opacity=0.6),
                    use_container_width=True)
                st.plotly_chart(
                    px.violin(df, x="sex", y="charges", color="smoker", box=True,
                              title="Charge Distribution by Sex",
                              color_discrete_map={"yes":"#ef4444","no":"#22c55e"}),
                    use_container_width=True)
            st.plotly_chart(
                px.scatter(df, x="age", y="charges", color="smoker", size="bmi",
                           facet_col="sex",
                           color_discrete_map={"yes":"#ef4444","no":"#22c55e"},
                           title="Age vs Charges by Sex & Smoker", opacity=0.7),
                use_container_width=True)
            st.markdown("### Raw Dataset")
            st.dataframe(df, use_container_width=True, height=300)

    # TAB 3 — MODEL INSIGHTS ───────────────────────────────────────────────────
    with tabs[2]:
        health = health_check()
        if health and "metrics" in health:
            st.markdown("### Model Performance Comparison")
            rows = [{"Model": n, "R²": round(m["r2"],4),
                     "MAE ($)": round(m["mae"],2), "RMSE ($)": round(m["rmse"],2),
                     "CV R²": round(m["cv_r2"],4)}
                    for n, m in health["metrics"].items()]
            mdf = pd.DataFrame(rows)
            st.dataframe(mdf.style
                .highlight_max(subset=["R²","CV R²"], color="#c8e6c9")
                .highlight_min(subset=["MAE ($)","RMSE ($)"], color="#c8e6c9"),
                use_container_width=True)
            st.plotly_chart(
                px.bar(mdf, x="Model", y="R²", color="R²",
                       color_continuous_scale="Blues",
                       title="R² Score by Model", text_auto=".4f"),
                use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.bar(mdf, x="Model", y="MAE ($)", color="MAE ($)",
                           color_continuous_scale="Reds_r",
                           title="MAE by Model", text_auto=",.0f"),
                    use_container_width=True)
            with c2:
                st.plotly_chart(
                    px.bar(mdf, x="Model", y="RMSE ($)", color="RMSE ($)",
                           color_continuous_scale="Oranges_r",
                           title="RMSE by Model", text_auto=",.0f"),
                    use_container_width=True)
        else:
            st.warning("Start the API and train the model to see insights.")
        st.markdown("---")
        st.markdown("### EDA Plots")
        plot_files = {
            "Charges Distribution": "charges_distribution.png",
            "Charges by Smoker":    "charges_by_smoker.png",
            "BMI vs Charges":       "bmi_vs_charges.png",
            "Correlation Heatmap":  "correlation_heatmap.png",
            "Feature Importance":   "feature_importance.png",
        }
        cols = st.columns(2); idx = 0
        for title, fname in plot_files.items():
            fp = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(fp):
                with cols[idx % 2]:
                    st.markdown(f"**{title}**")
                    st.image(fp, use_container_width=True)
                idx += 1

    # TAB 4 — BATCH PREDICT ────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Batch Prediction")
        st.markdown("Upload a CSV with: `age, sex, bmi, children, smoker, region`")
        with st.expander("📋 Example CSV format"):
            ex = pd.DataFrame([
                {"age":25,"sex":"male","bmi":22.5,"children":0,"smoker":"no","region":"northeast"},
                {"age":45,"sex":"female","bmi":31.2,"children":2,"smoker":"yes","region":"southwest"},
            ])
            st.dataframe(ex, use_container_width=True)
            st.download_button("⬇️ Download Template", ex.to_csv(index=False),
                               "sample_input.csv", "text/csv")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                st.dataframe(batch_df.head(), use_container_width=True)
                if st.button("▶ Run Batch Prediction", type="primary"):
                    results = call_api("/predict/batch", batch_df.to_dict("records"))
                    if results:
                        res_df = pd.DataFrame(results)
                        if "predicted_charge" in res_df.columns:
                            batch_df = batch_df.reset_index(drop=True)
                            batch_df["predicted_charge"] = res_df["predicted_charge"]
                            st.success(f"✅ {len(batch_df)} predictions complete!")
                            st.dataframe(batch_df, use_container_width=True)
                            ok = batch_df.dropna(subset=["predicted_charge"])
                            if len(ok):
                                st.plotly_chart(
                                    px.histogram(ok, x="predicted_charge", nbins=30,
                                                 title="Predicted Charge Distribution",
                                                 color_discrete_sequence=["#3b82f6"]),
                                    use_container_width=True)
                            st.download_button("⬇️ Download Results",
                                               batch_df.to_csv(index=False),
                                               "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#888;font-size:.85rem;'>"
                "Medical Cost Predictor · Built with Streamlit + Flask</div>",
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Medical Cost Prediction — single-file launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_app.py --train              # Train / retrain the model
  python full_app.py --backend            # Start Flask API only
  python full_app.py --frontend           # Start Streamlit only (API must be running)
  python full_app.py --all                # Train if needed, then start both
  python full_app.py --train --force      # Force retrain even if model exists
""",
    )
    parser.add_argument("--train",    action="store_true", help="Train the model")
    parser.add_argument("--force",    action="store_true", help="Force retrain")
    parser.add_argument("--backend",  action="store_true", help="Start Flask API")
    parser.add_argument("--frontend", action="store_true", help="Start Streamlit UI")
    parser.add_argument("--all",      action="store_true",
                        help="Train (if needed), start Flask+Streamlit")
    args = parser.parse_args()

    if not any([args.train, args.backend, args.frontend, args.all]):
        parser.print_help()
        return

    if args.train:
        run_training(force=args.force)

    if args.backend:
        run_backend()

    if args.all:
        run_training(force=False)
        start_backend_thread()
        run_frontend()

    if args.frontend:
        run_frontend()


if __name__ == "__main__":
    main()
