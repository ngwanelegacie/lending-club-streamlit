"""
Lending Club Loan Default Predictor
Full-showcase Streamlit app — prediction, EDA, model comparison, feature importance.
Author: Vusumuzi Nkosi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ── Page config ──
st.set_page_config(
    page_title="Lending Club Risk Predictor — Vusumuzi Nkosi",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark navy/green theme) ──
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a192f;
        color: #ccd6f6;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #112240;
        border-right: 1px solid #233554;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #ccd6f6 !important;
    }

    /* Accent text */
    .accent {
        color: #64ffda;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #112240;
        border: 1px solid #233554;
        border-radius: 8px;
        padding: 16px;
    }

    div[data-testid="stMetric"] label {
        color: #8892b0 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #64ffda !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #112240;
        border: 1px solid #233554;
        border-radius: 8px;
        color: #8892b0;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(100, 255, 218, 0.1) !important;
        border-color: #64ffda !important;
        color: #64ffda !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: transparent;
        border: 1px solid #64ffda;
        color: #64ffda;
        border-radius: 4px;
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        background-color: rgba(100, 255, 218, 0.1);
    }

    /* Select boxes and inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div {
        background-color: #112240 !important;
        color: #ccd6f6 !important;
        border-color: #233554 !important;
    }

    /* Divider */
    hr {
        border-color: #233554;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #495670;
        font-size: 13px;
        padding: 20px 0;
        font-family: 'Fira Code', monospace;
    }

    .footer a {
        color: #64ffda;
        text-decoration: none;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #112240 0%, #0a192f 100%);
        border: 1px solid #233554;
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 24px;
    }

    .hero-banner h1 {
        font-size: 2rem;
        margin-bottom: 8px;
    }

    .hero-banner p {
        color: #8892b0;
        font-size: 1rem;
    }

    /* Risk badge */
    .risk-low {
        background-color: rgba(100, 255, 218, 0.15);
        color: #64ffda;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    .risk-medium {
        background-color: rgba(255, 183, 77, 0.15);
        color: #ffb74d;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    .risk-high {
        background-color: rgba(229, 69, 96, 0.15);
        color: #e94560;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a192f",
    plot_bgcolor="#112240",
    font=dict(color="#ccd6f6", family="Inter, sans-serif"),
    title_font=dict(color="#ccd6f6", size=16),
    xaxis=dict(gridcolor="#233554", zerolinecolor="#233554"),
    yaxis=dict(gridcolor="#233554", zerolinecolor="#233554"),
    margin=dict(l=40, r=40, t=50, b=40),
)

COLORS = ["#64ffda", "#e94560", "#ffb74d", "#57cbff", "#f57dff", "#82ca9d"]


# ── Load model and data ──
@st.cache_resource
def load_model():
    """Load the trained model, scaler, and metadata."""
    model_path = os.path.join(os.path.dirname(__file__), "model")
    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return model, scaler, metadata


@st.cache_data
def load_eda_data():
    """Load pre-computed EDA summary data."""
    data_path = os.path.join(os.path.dirname(__file__), "data")
    eda = {}
    for name in ["grade_defaults", "purpose_defaults", "term_defaults",
                  "ownership_defaults", "yearly_defaults", "model_results",
                  "feature_importance", "confusion_lr", "confusion_gb"]:
        filepath = os.path.join(data_path, f"{name}.csv")
        if os.path.exists(filepath):
            eda[name] = pd.read_csv(filepath)
    return eda


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🏦 Navigation")
    page = st.radio(
        "Go to",
        ["Predict Risk", "Explore Data", "Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='color: #8892b0; font-size: 13px;'>
        <strong style='color: #64ffda;'>About this app</strong><br>
        Built on 2.26M real Lending Club loans (2007–2018).
        Gradient Boosting classifier with 0.72 AUC-ROC.<br><br>
        <strong style='color: #64ffda;'>Built by</strong><br>
        <a href='https://ngwanelegacie.github.io' target='_blank' style='color: #64ffda;'>Vusumuzi Nkosi</a><br>
        Data Scientist<br><br>
        <a href='https://github.com/ngwanelegacie/lending-credit-analysis' target='_blank' style='color: #64ffda;'>View source code →</a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 1: PREDICT RISK
# ══════════════════════════════════════════════
if page == "Predict Risk":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🏦 Loan Default Risk Predictor</h1>
        <p>Enter loan application details below. The model predicts the probability
        of default based on patterns learned from 2.26 million real Lending Club loans.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, scaler, metadata = load_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
        term = st.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} months")
        int_rate = st.slider("Interest Rate (%)", 5.0, 31.0, 12.0, 0.5)
        installment = st.number_input("Monthly Installment ($)", min_value=20, max_value=1500, value=330, step=10)

    with col2:
        st.markdown("##### Borrower Profile")
        annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
        dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0, 0.5)
        emp_length = st.selectbox("Employment Length (years)", list(range(0, 11)), index=5)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

    with col3:
        st.markdown("##### Credit History")
        fico_range_low = st.slider("FICO Score", 600, 850, 700, 5)
        revol_util = st.slider("Revolving Utilisation (%)", 0.0, 100.0, 45.0, 1.0)
        open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=10)
        total_acc = st.number_input("Total Credit Lines", min_value=1, max_value=100, value=25)

    st.markdown("---")

    if st.button("⚡ Predict Default Risk", use_container_width=True):
        if model_loaded:
            # Engineer features to match training
            income_to_loan = min(annual_inc / max(loan_amnt, 1), 100)
            loan_to_income = min(loan_amnt / max(annual_inc, 1), 5)
            high_risk_credit = 1 if fico_range_low < 670 else 0
            high_dti = 1 if dti > 30 else 0
            installment_burden = min(installment * 12 / max(annual_inc, 1), 1)

            # Build feature vector (order must match training)
            features = metadata.get("feature_order", [])
            input_data = {
                "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
                "installment": installment, "annual_inc": annual_inc, "dti": dti,
                "emp_length": emp_length, "fico_range_low": fico_range_low,
                "revol_util": revol_util, "open_acc": open_acc, "total_acc": total_acc,
                "income_to_loan": income_to_loan, "loan_to_income": loan_to_income,
                "high_risk_credit": high_risk_credit, "high_dti": high_dti,
                "installment_burden": installment_burden,
            }

            # Fill missing features with defaults
            X_input = np.array([[input_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X_input)

            prob = model.predict_proba(X_scaled)[0][1]
            pred = "Default" if prob >= 0.5 else "Paid"

            # Display result
            st.markdown("### Prediction Result")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Default Probability", f"{prob:.1%}")
            with r2:
                st.metric("Prediction", pred)
            with r3:
                if prob < 0.3:
                    risk_label = "LOW RISK"
                    risk_class = "risk-low"
                elif prob < 0.6:
                    risk_label = "MEDIUM RISK"
                    risk_class = "risk-medium"
                else:
                    risk_label = "HIGH RISK"
                    risk_class = "risk-high"
                st.markdown(f"<div class='{risk_class}'>{risk_label}</div>", unsafe_allow_html=True)

            # Risk factors
            st.markdown("##### Key Risk Factors")
            factors = []
            if fico_range_low < 670:
                factors.append("⚠️ FICO score below 670 (subprime)")
            if dti > 30:
                factors.append("⚠️ High debt-to-income ratio (>30%)")
            if term == 60:
                factors.append("⚠️ 60-month term (2x default rate vs 36-month)")
            if int_rate > 20:
                factors.append("⚠️ High interest rate (>20%)")
            if revol_util > 80:
                factors.append("⚠️ High revolving utilisation (>80%)")
            if not factors:
                factors.append("✅ No major risk flags detected")
            for f in factors:
                st.markdown(f)
        else:
            st.warning("Model files not found. Run the setup script first — see the README.")
            st.markdown("""
            **Demo mode:** The model hasn't been exported yet. To set it up:
            1. Run `python setup_model.py` in the project directory
            2. This exports the trained model, scaler, and EDA data
            3. Restart the app
            """)


# ══════════════════════════════════════════════
# PAGE 2: EXPLORE DATA
# ══════════════════════════════════════════════
elif page == "Explore Data":
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown("*Key patterns from 2.26 million Lending Club loans (2007–2018)*")

    eda = load_eda_data()

    if not eda:
        st.warning("EDA data files not found. Run `python setup_model.py` first.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["By Grade", "By Purpose & Term", "Over Time", "Correlations"])

    with tab1:
        if "grade_defaults" in eda:
            df_g = eda["grade_defaults"]
            fig = px.bar(
                df_g, x="grade", y="default_rate",
                color="default_rate",
                color_continuous_scale=["#64ffda", "#e94560"],
                text=df_g["default_rate"].apply(lambda x: f"{x:.1f}%"),
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Default Rate by Loan Grade",
                              coloraxis_showscale=False, showlegend=False)
            fig.update_traces(textposition="outside")
            fig.update_xaxes(title="Grade")
            fig.update_yaxes(title="Default Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            > **Insight:** Default rates scale almost perfectly with grade —
            from **6.2% (Grade A)** to **48.2% (Grade G)**. Lending Club's
            grading system captures real underlying risk.
            """)

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            if "purpose_defaults" in eda:
                df_p = eda["purpose_defaults"].sort_values("default_rate", ascending=True)
                fig = px.bar(
                    df_p, x="default_rate", y="purpose", orientation="h",
                    color_discrete_sequence=["#3266ad"],
                    text=df_p["default_rate"].apply(lambda x: f"{x:.1f}%"),
                )
                fig.update_layout(**PLOTLY_LAYOUT, title="Default Rate by Loan Purpose", height=400)
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "term_defaults" in eda:
                df_t = eda["term_defaults"]
                fig = px.bar(
                    df_t, x="term", y="default_rate",
                    color="term", color_discrete_sequence=["#1D9E75", "#e94560"],
                    text=df_t["default_rate"].apply(lambda x: f"{x:.1f}%"),
                )
                fig.update_layout(**PLOTLY_LAYOUT, title="Default Rate by Loan Term",
                                  showlegend=False, height=400)
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        > **Insights:** Small business loans have the highest default rate at **~28%** — 3x higher
        than car loans. **60-month loans default at 2x the rate** of 36-month loans (32.3% vs 16.0%).
        """)

    with tab3:
        if "yearly_defaults" in eda:
            df_y = eda["yearly_defaults"]
            fig = px.line(
                df_y, x="year", y="default_rate",
                markers=True, color_discrete_sequence=["#64ffda"],
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Default Rate Over Time (2007–2018)")
            fig.update_xaxes(title="Issue Year")
            fig.update_yaxes(title="Default Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            > **Insight:** Default rates peaked around **2007–2008** (financial crisis),
            dropped through 2009–2012, then rose again in later years as Lending Club
            expanded to riskier borrower segments.
            """)

    with tab4:
        if "ownership_defaults" in eda:
            df_o = eda["ownership_defaults"]
            fig = px.bar(
                df_o, x="home_ownership", y="default_rate",
                color="home_ownership",
                color_discrete_sequence=["#1D9E75", "#ffb74d", "#3266ad"],
                text=df_o["default_rate"].apply(lambda x: f"{x:.1f}%"),
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Default Rate by Home Ownership",
                              showlegend=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            > **Insight:** Renters default at the highest rate (**23.4%**),
            followed by homeowners (20.6%) and mortgage holders (17.0%).
            """)


# ══════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("## 🎯 Model Performance")
    st.markdown("*Comparing three classifiers trained on the Lending Club dataset*")

    eda = load_eda_data()

    if not eda:
        st.warning("Model data files not found. Run `python setup_model.py` first.")
        st.stop()

    # Model comparison table
    if "model_results" in eda:
        st.markdown("### Model Comparison")
        df_m = eda["model_results"]

        m1, m2, m3 = st.columns(3)
        for col_widget, idx in zip([m1, m2, m3], range(min(3, len(df_m)))):
            row = df_m.iloc[idx]
            with col_widget:
                model_name = row.get("model", f"Model {idx+1}")
                st.markdown(f"**{model_name}**")
                st.metric("AUC-ROC", f"{row.get('auc', 0):.3f}")
                st.metric("Accuracy", f"{row.get('acc', 0):.3f}")
                st.metric("F1 Score", f"{row.get('f1', 0):.3f}")

    st.markdown("---")

    # Feature importance
    if "feature_importance" in eda:
        st.markdown("### Feature Importance (Gradient Boosting)")
        df_fi = eda["feature_importance"].sort_values("importance", ascending=True).tail(12)

        fig = px.bar(
            df_fi, x="importance", y="feature", orientation="h",
            color_discrete_sequence=["#3266ad"],
            text=df_fi["importance"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=450,
                          title="Top 12 Most Predictive Features")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        > **Key takeaway:** `sub_grade` dominates at **23.3%** importance — it's essentially
        Lending Club's own risk assessment encoded. After that, DTI (5.5%), interest rate (4.8%),
        and issue year (4.8%) are the strongest predictors.
        """)

    st.markdown("---")

    # Business impact
    st.markdown("### 💰 Business Impact")
    st.markdown("""
    At portfolio scale (100,000 loans/year), the Gradient Boosting model:
    """)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Losses Prevented", "~$114M/year")
    with b2:
        st.metric("Opportunity Cost", "~$19M/year")
    with b3:
        st.metric("Net Benefit", "~$95M/year")

    st.markdown("""
    > The model correctly flags high-risk borrowers, preventing significant losses.
    The opportunity cost reflects good borrowers incorrectly flagged — an acceptable
    trade-off at this scale.
    """)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div class='footer'>
    Built by <a href='https://ngwanelegacie.github.io' target='_blank'>Vusumuzi Nkosi</a> ·
    <a href='https://github.com/ngwanelegacie/lending-credit-analysis' target='_blank'>Source Code</a> ·
    <a href='https://linkedin.com/in/ngwanelegacie' target='_blank'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
