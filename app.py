import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd
from streamlit_option_menu import option_menu
import traceback

# -------------------------------
# CONFIG
# -------------------------------
DEBUG = False

st.set_page_config(
    page_title="Smart Property Advisor",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------
# DEFAULT / CLEAR STATES
# -------------------------------
DEFAULTS = {
    "CRIM": 0.5, "ZN": 10.0, "INDUS": 5.0, "CHAS": 0,
    "NOX": 0.5, "DIS": 3.0, "RM": 6.5, "AGE": 50.0,
    "RAD": 5.0, "TAX": 300.0, "PTRATIO": 18.0,
    "B": 350.0, "LSTAT": 12.5
}

CLEAR_VALUES = {
    "CRIM": 0.0, "ZN": 0.0, "INDUS": 0.0, "CHAS": 0,
    "NOX": 0.0, "DIS": 0.0, "RM": 1.0, "AGE": 0.0,
    "RAD": 1.0, "TAX": 0.0, "PTRATIO": 0.0,
    "B": 0.0, "LSTAT": 0.0
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("🏠 Smart Property Advisor")
    selected = option_menu(
        None,
        ["Predictor", "Insights", "About"],
        icons=["house", "info-circle", "person"],
        default_index=0
    )

# -------------------------------
# PREDICTOR
# -------------------------------
if selected == "Predictor":

    st.title("🏠 Property Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.CRIM = st.slider("Crime Rate (per capita)", 0.0, 10.0, st.session_state.CRIM)
        st.session_state.ZN = st.number_input("Residential Land Zoned (%)", value=st.session_state.ZN)
        st.session_state.INDUS = st.number_input("Non-Retail Business Area (%)", value=st.session_state.INDUS)

        chas = st.selectbox("Located near Charles River?", ["No", "Yes"], index=st.session_state.CHAS)
        st.session_state.CHAS = 1 if chas == "Yes" else 0

        st.session_state.NOX = st.slider("Air Pollution (NOx concentration)", 0.0, 1.0, st.session_state.NOX)
        st.session_state.DIS = st.number_input("Distance to Employment Centers", value=st.session_state.DIS)

    with col2:
        st.session_state.RM = st.slider("Average Rooms per Dwelling", 1.0, 10.0, st.session_state.RM)
        st.session_state.AGE = st.number_input("Old Houses (%) (built before 1940)", value=st.session_state.AGE)
        st.session_state.RAD = st.number_input("Highway Accessibility Index", value=st.session_state.RAD)
        st.session_state.TAX = st.number_input("Property Tax Rate", value=st.session_state.TAX)

        st.session_state.PTRATIO = st.number_input("Student-Teacher Ratio", value=st.session_state.PTRATIO)
        st.session_state.B = st.number_input("Population Diversity Index", value=st.session_state.B)
        st.session_state.LSTAT = st.slider("Lower Income Population (%)", 0.0, 40.0, st.session_state.LSTAT)

    # Buttons
    c1, c2 = st.columns(2)

    if c1.button("🧹 Clear All"):
        st.session_state.update(CLEAR_VALUES)
        st.rerun()

    if c2.button("🔄 Reset Defaults"):
        st.session_state.update(DEFAULTS)
        st.rerun()

    # Prediction
    try:
        pipeline = PredictPipeline()

        data = CustomData(
            st.session_state.CRIM,
            st.session_state.ZN,
            st.session_state.INDUS,
            st.session_state.NOX,
            st.session_state.RM,
            st.session_state.AGE,
            st.session_state.DIS,
            st.session_state.RAD,
            st.session_state.TAX,
            st.session_state.PTRATIO,
            st.session_state.B,
            st.session_state.LSTAT,
            st.session_state.CHAS
        )

        df = data.get_data_as_dataframe()

        df = df.reindex(
            columns=pipeline.preprocessor.feature_names_in_,
            fill_value=0
        )

        prediction = pipeline.predict(df)[0] * 1000

        st.success(f"💰 Estimated Property Price: ${prediction:,.2f}")

    except Exception:
        st.error("Prediction failed. Please adjust inputs.")
        if DEBUG:
            st.text(traceback.format_exc())

# -------------------------------
# INSIGHTS (REPLACEMENT - SIMPLE + STRONG)
# -------------------------------
elif selected == "Insights":

    st.title("📊 Key Property Insights")

    st.markdown("""
    ### 🧠 What drives property prices?

    - **🏠 More Rooms → Higher Price**  
      Larger houses tend to have higher value.

    - **🚨 Higher Crime Rate → Lower Price**  
      Safety plays a major role in valuation.

    - **🌫️ Pollution (NOx) → Negative Impact**  
      Cleaner environments are more desirable.

    - **📉 Lower Income Population (LSTAT) → Lower Price**  
      Socioeconomic factors influence pricing.

    - **📍 Distance to Jobs → Mixed Effect**  
      Accessibility vs peaceful locations trade-off.

    ### 🎯 Why this matters

    This model captures real-world relationships between
    property features and market prices, helping users
    make data-driven decisions.
    """)

    st.info("💡 These insights are derived from historical housing data patterns.")

# -------------------------------
# ABOUT
# -------------------------------
else:

    st.title("👤 About Me")

    st.markdown("""
    ### 👋 Hi, I'm Atharva Sawant

    I am a Machine Learning enthusiast focused on building 
    real-world, production-ready ML applications.

    ### 🚀 About This Project
    - End-to-end ML pipeline
    - Data preprocessing + model prediction
    - Deployed using Streamlit
    - Designed with user-friendly interface

    ### 💼 Skills Demonstrated
    - Machine Learning (Scikit-learn)
    - Data Processing
    - Model Deployment
    - UI/UX Design using Streamlit

    ### 📬 Contact Me
    - 📧 Email: atharvasawant3183@gmail.com  
    - 📱 Phone: +91 9653320569  

    ### 🎯 Goal
    To build scalable ML solutions that solve real-world problems.
    """)

    st.success("✅ Portfolio-ready ML project")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Smart Property Advisor | Built with Streamlit")