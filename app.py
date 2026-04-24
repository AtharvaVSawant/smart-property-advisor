import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------
# Custom CSS (Clean & Professional)
# -------------------------------
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #f5f7fb;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    
    .main-header h1 {
        color: #1e3c72;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #6c757d;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    .result-card h3 {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .result-card .price {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .result-card .range {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2a5298;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #1e3c72;
        transform: translateY(-1px);
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    
    /* Info box */
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #6c757d;
        font-size: 0.85rem;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Navigation
# -------------------------------
with st.sidebar:
    st.markdown("## 🏠 Property Predictor")
    
    selected = option_menu(
        menu_title=None,
        options=["Predictor", "Market Trends", "About"],
        icons=["calculator", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"font-size": "18px"},
            "nav-link": {"font-size": "15px", "margin": "2px 0"},
            "nav-link-selected": {"background-color": "#2a5298"},
        }
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.caption("Model trained on Boston Housing Dataset")
    st.caption("Features include crime rate, rooms, location, etc.")

# -------------------------------
# Main Content
# -------------------------------
if selected == "Predictor":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Property Price Predictor</h1>
        <p>Enter property details below to get an estimated market value</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown("### 📋 Property Information")
    
    # Create tabs for organized input
    tab1, tab2, tab3 = st.tabs(["📍 Location & Area", "🏢 Property Features", "🌍 Neighborhood"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            CRIM = st.number_input(
                "Crime Rate", 
                min_value=0.0, 
                step=0.1, 
                value=0.5, 
                format="%.2f",
                help="Per capita crime rate by town"
            )
            
            ZN = st.number_input(
                "Residential Land (acres)", 
                min_value=0, 
                value=10,
                help="Proportion of residential land zoned for lots over 25,000 sq.ft"
            )
            
            INDUS = st.number_input(
                "Industrial Area (%)", 
                min_value=0.0,
                value=5.0, 
                format="%.2f",
                help="Proportion of non-retail business acres per town"
            )
        
        with col2:
            CHAS = st.selectbox(
                "Located on Charles River?", 
                ["No", "Yes"],
                help="Property location relative to Charles River"
            )
            CHAS = 1 if CHAS == "Yes" else 0
            
            NOX = st.number_input(
                "Air Pollution Level (NOx)", 
                min_value=0.0, 
                step=0.01,
                value=0.5, 
                format="%.3f",
                help="Nitric oxides concentration"
            )
            
            DIS = st.number_input(
                "Distance to Employment Centers", 
                min_value=0.0,
                value=3.0, 
                format="%.3f",
                help="Weighted distance to Boston employment centers"
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            RM = st.number_input(
                "Average Rooms per Dwelling", 
                min_value=1.0, 
                step=0.1,
                value=6.5, 
                format="%.2f",
                help="Average number of rooms"
            )
            
            AGE = st.number_input(
                "Property Age (years)", 
                min_value=0, 
                max_value=150,
                value=50,
                help="Age of the property"
            )
        
        with col2:
            TAX = st.number_input(
                "Property Tax Rate", 
                min_value=0,
                value=300,
                help="Full-value property-tax rate per $10,000"
            )
            
            RAD = st.number_input(
                "Highway Access Index", 
                min_value=1,
                value=5,
                help="Accessibility to radial highways (1-24)"
            )
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            PTRATIO = st.number_input(
                "Student-Teacher Ratio", 
                min_value=0.0,
                value=18.0, 
                format="%.2f",
                help="Pupil-teacher ratio by town"
            )
            
            LSTAT = st.number_input(
                "Lower Income Population (%)", 
                min_value=0.0,
                value=12.5, 
                format="%.2f",
                help="Percentage of lower status population"
            )
        
        with col2:
            B = st.number_input(
                "Population Diversity Index", 
                min_value=0.0,
                value=350.0, 
                format="%.2f",
                help="Proportion of Black population (0-500)"
            )
    
    # Prediction Button
    st.markdown("---")
    predict_button = st.button("Calculate Estimated Price", use_container_width=True)
    
    if predict_button:
        with st.spinner("Calculating..."):
            time.sleep(0.5)  # Smooth loading effect
            
            try:
                # Prepare data for prediction
                data = CustomData(
                    CRIM, ZN, INDUS, NOX, RM, AGE,
                    DIS, RAD, TAX, PTRATIO, B, LSTAT, CHAS
                )
                
                input_df = data.get_data_as_dataframe()
                pipeline = PredictPipeline()
                result = pipeline.predict(input_df)
                
                price = result[0] * 1000  # Convert to dollars
                
                # Price range (±15%)
                min_price = price * 0.85
                max_price = price * 1.15
                
                # Display result
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>Estimated Property Value</h3>
                        <div class="price">${price:,.2f}</div>
                        <div class="range">Estimated range: ${min_price:,.0f} - ${max_price:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Price per Room", f"${price/RM:,.0f}")
                with col2:
                    st.metric("Tax Impact (Annual)", f"${(price * TAX/100000):,.0f}")
                with col3:
                    premium = "Premium" if price > 300000 else "Standard"
                    st.metric("Property Class", premium)
                
                # Quick insight
                st.info("💡 **Insight**: This estimate is based on comparable properties in the area and current market conditions.")
                
            except Exception as e:
                st.error(f"Unable to calculate estimate. Please check your inputs.")
                if st.checkbox("Show error details"):
                    st.code(str(e))

elif selected == "Market Trends":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Market Trends</h1>
        <p>Latest real estate market insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample trend visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price by Area Type")
        df_prices = pd.DataFrame({
            'Area': ['Downtown', 'Suburban', 'Rural', 'Waterfront'],
            'Avg Price ($K)': [450, 320, 250, 520]
        })
        
        # Define custom colors for each bar
        custom_colors = {
            'Downtown': '#1e3c72',    # Dark blue
            'Suburban': '#2a5298',     # Medium blue
            'Rural': '#28a745',        # Green (changed from white)
            'Waterfront': '#17a2b8'    # Teal/cyan
        }
        
        fig = px.bar(df_prices, x='Area', y='Avg Price ($K)', 
                     color='Area',
                     color_discrete_map=custom_colors,
                     title="Average Property Values")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Market Indicators")
        st.metric("Current Market", "Stable", "Neutral")
        st.metric("Inventory Level", "Moderate", "↓ 5%")
        st.metric("Avg Days on Market", "45 days", "↑ 3 days")
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>📊 Market Note:</strong> Property values are influenced by location, property condition, 
        and local economic factors. Consider getting a professional appraisal for final decisions.
    </div>
    """, unsafe_allow_html=True)

else:  # About
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About This Tool</h1>
        <p>Understanding property valuation through data</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    
    This property price predictor uses machine learning to estimate real estate values based on multiple factors including:
    
    - **Location metrics**: Crime rates, industrial area proportion, distance to employment centers
    - **Property characteristics**: Number of rooms, age, tax rate
    - **Neighborhood features**: Student-teacher ratio, demographic data
    - **Accessibility**: Highway access, proximity to amenities
    
    ### 🔧 How It Works
    
    The model analyzes relationships between property features and actual sale prices to provide data-driven estimates.
    
    ### ⚠️ Important Notes
    
    - Estimates are based on historical data patterns
    - Actual market prices may vary
    - Consider multiple sources for final decisions
    - Professional appraisals recommended for official use
    
    ### 📧 Contact
    
    For questions or feedback, please reach out to our support team.
    Email : atharvasawant3183@gmail.com
    Contact : +91 9653320569
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Fill in all property details accurately for the best estimate.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="footer">
    <p>Property Price Predictor | Estimates based on market data</p>
</div>
""", unsafe_allow_html=True)