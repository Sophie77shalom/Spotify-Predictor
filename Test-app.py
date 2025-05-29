import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# --- APP CONFIG ---
st.set_page_config(
    page_title="Spotify Analytics Suite",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    models = {
        "Popularity Predictor": joblib.load("models/popularity_regressor.pkl"),
        "Explicit Content Detector": joblib.load("models/explicit_predictor.pkl"),
        "Genre Classifier": joblib.load("models/genre_classifier.pkl"),
        "Popularity Tier (XGBoost)": joblib.load("models/popularity_classifier.pkl")
    }
    return models

models = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎧 Spotify Model Hub")
    
    model_choice = st.radio(
        "SELECT MODEL",
        list(models.keys()),
        index=0,
        format_func=lambda x: f"🎵 {x}"
    )
    
    st.markdown("---")
    st.markdown("""
    **How to Use:**
    1. Select a model
    2. Adjust input parameters
    3. View predictions & insights
    """)

# --- MAIN DASHBOARD ---
st.title(f"🔮 {model_choice}")
st.caption("Powered by Machine Learning | Spotify Data Analytics")

# --- MODEL INPUT SECTION ---
with st.expander("🎛️ TRACK PARAMETERS", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        duration = st.slider("Duration (min)", 0.5, 10.0, 3.5, 0.1,
                           help="Track length in minutes")
        
    with col2:
        popularity = st.slider("Popularity Score", 0, 100, 70,
                             help="Spotify's popularity metric (0-100)")
        
    with col3:
        explicit = st.toggle("Explicit Content", value=False,
                           help="Contains explicit lyrics")

# --- PREDICTION ENGINE ---
input_data = pd.DataFrame({
    "duration_min": [duration],
    "popularity": [popularity],
    "explicit": [int(explicit)],
    "genre": "pop"

})

# --- MODEL-SPECIFIC OUTPUT ---
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Insights", "🛠️ Model Info"])

with tab1:
    model = models[model_choice]
    
    if "Predictor" in model_choice:
        pred = model.predict(input_data)[0]
        st.metric("PREDICTED POPULARITY SCORE", 
                 f"{pred:.1f}",
                 delta=f"{(pred-popularity):.1f} vs Input")
        
    elif "Detector" in model_choice:
        proba = model.predict_proba(input_data)[0][1]
        gauge = px.indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=proba,
            title="Explicit Content Probability"
        )
        st.plotly_chart(gauge, use_container_width=True)
        
    elif "Classifier" in model_choice:
        pred = model.predict(input_data)[0]
        if "Genre" in model_choice:
            decoded = encoders["genre"].inverse_transform([pred])[0]
            st.metric("PREDICTED GENRE", decoded)
        else:
            levels = ["Low", "Medium", "High"]
            st.metric("POPULARITY TIER", levels[pred])
            
        # Show probabilities
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(input_data)[0]
            fig = px.bar(
                x=encoders["popularity"].classes_ if "Tier" in model_choice else encoders["genre"].classes_,
                y=probas,
                title="Class Probabilities"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Feature importance visualization
    if hasattr(model, 'feature_importances_'):
        features = input_data.columns
        importances = model.feature_importances_
        
        fig = px.bar(
            x=features,
            y=importances,
            title="Feature Importance",
            labels={'x': 'Features', 'y': 'Importance'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # SHAP analysis (if available)
    if st.checkbox("Show advanced explanation"):
        import shap
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        shap_values = explainer.shap_values(input_data)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_data, plot_type="bar")
        st.pyplot(fig)

with tab3:
    st.subheader("Model Performance")
    
    # Mock metrics - replace with your actual metrics
    metrics = {
        "Accuracy": 0.92,
        "Precision": 0.89,
        "Recall": 0.94,
        "F1 Score": 0.91
    } if "Detector" in model_choice else {
        "RMSE": 12.3,
        "R² Score": 0.85,
        "MAE": 9.1
    }
    
    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
    
    st.subheader("Training Details")
    st.markdown("""
    - **Algorithm**: XGBoost
    - **Training Data**: 10,000 tracks
    - **Last Updated**: November 2023
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
}
</style>
<div class="footer">
<p>© 2023 Spotify Analytics Suite | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)