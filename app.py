import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# --- APP CONFIG ---
st.set_page_config(
    page_title="Spotify Analytics Suite",
    page_icon="🎧",
    layout="wide"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    try:
        models = {}
        model_files = {
            "Popularity Predictor": "models/popularity_regressor.pkl",
            "Explicit Content Detector": "models/explicit_predictor.pkl", 
            "Genre Classifier": "models/genre_classifier.pkl",
            "Popularity Tier": "models/popularity_classifier.pkl"
        }
        
        for name, file_path in model_files.items():
            try:
                models[name] = joblib.load(file_path)
                st.success(f"✅ {name} loaded")
            except FileNotFoundError:
                st.warning(f"⚠️ {name} file not found")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

# Load models
with st.spinner("Loading models..."):
    models = load_models()

if not models:
    st.error("❌ No models could be loaded. Check your models/ directory.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🎧 Spotify Analytics")
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.info("Adjust parameters and get instant predictions!")

# --- MAIN APP ---
st.title(f"🎵 {model_choice}")

# --- INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎛️ Track Features")
    duration = st.slider("Duration (minutes)", 1.0, 8.0, 3.5, 0.1)
    popularity = st.slider("Current Popularity", 0, 100, 50)
    explicit = st.checkbox("Explicit Content")
    
    # Additional features that your models might need
    st.subheader("🎭 Additional Features")
    genre = st.selectbox("Genre", [
        "Pop", "Rock", "Hip-Hop", "Electronic", "Jazz", "Classical", 
        "Country", "R&B", "Reggae", "Folk", "Alternative", "Indie"
    ])
    
    artist_popularity = st.slider("Artist Popularity", 0, 100, 50)
    artist_frequency = st.slider("Artist Track Frequency", 1, 50, 10)

with col2:
    st.subheader("📊 Prediction Results")
    
    # Create comprehensive input data with all possible features
    input_data = pd.DataFrame({
        "duration_min": [duration],
        "popularity": [popularity], 
        "explicit": [int(explicit)],
        "genre": [genre],
        "artist_popularity": [artist_popularity],
        "artist_frequency": [artist_frequency]
    })
    
    try:
        model = models[model_choice]
        
        # Get the features the model actually expects
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            # Filter input_data to only include expected features
            available_features = [col for col in expected_features if col in input_data.columns]
            model_input = input_data[available_features]
        else:
            # Fallback: try with all features, let the model decide
            model_input = input_data
        
        # Make prediction
        if "Predictor" in model_choice:
            # Regression model
            prediction = model.predict(model_input)[0]
            st.metric(
                "Predicted Popularity", 
                f"{prediction:.1f}",
                delta=f"{prediction - popularity:.1f}"
            )
            
            # Simple visualization
            fig = px.bar(
                x=["Current", "Predicted"],
                y=[popularity, prediction],
                title="Popularity Comparison",
                color=["Current", "Predicted"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif "Detector" in model_choice:
            # Binary classification
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(model_input)[0]
                explicit_prob = proba[1] if len(proba) > 1 else proba[0]
                
                st.metric("Explicit Probability", f"{explicit_prob:.1%}")
                
                # Simple gauge using progress bar
                st.progress(explicit_prob)
                
                if explicit_prob > 0.5:
                    st.error("🚫 Likely contains explicit content")
                else:
                    st.success("✅ Likely clean content")
            else:
                prediction = model.predict(model_input)[0]
                result = "Explicit" if prediction == 1 else "Clean"
                st.metric("Classification", result)
                
        elif "Classifier" in model_choice:
            # Multi-class classification
            prediction = model.predict(model_input)[0]
            
            if "Genre" in model_choice:
                # You might need to map numbers to genre names
                genres = ["Pop", "Rock", "Hip-Hop", "Electronic", "Jazz", "Classical"]
                genre_result = genres[prediction] if prediction < len(genres) else f"Genre {prediction}"
                st.metric("Predicted Genre", genre_result)
                
            else:  # Popularity tier
                tiers = ["Low Popularity", "Medium Popularity", "High Popularity"]
                tier = tiers[prediction] if prediction < len(tiers) else f"Tier {prediction}"
                st.metric("Popularity Tier", tier)
            
            # Show probabilities if available
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(model_input)[0]
                
                # Create simple bar chart
                labels = [f"Class {i}" for i in range(len(probas))]
                fig = px.bar(
                    x=labels,
                    y=probas,
                    title="Class Probabilities"
                )
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Check if your input features match the model's expected format.")
        
        # Debug information
        if hasattr(models[model_choice], 'feature_names_in_'):
            st.write("**Expected features:**", list(models[model_choice].feature_names_in_))
        st.write("**Provided features:**", list(input_data.columns))

# --- SIMPLE INSIGHTS ---
st.markdown("---")
st.subheader("💡 Quick Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"**Duration Impact**\n\n{'Longer' if duration > 4 else 'Shorter'} tracks tend to have {'higher' if duration > 4 else 'lower'} complexity.")

with col2:
    st.info(f"**Popularity Level**\n\n{popularity}/100 is {'high' if popularity > 70 else 'moderate' if popularity > 40 else 'low'} popularity.")

with col3:
    st.info(f"**Content Rating**\n\n{'Explicit' if explicit else 'Clean'} content affects audience reach.")

# --- MODEL INFO ---
with st.expander("ℹ️ Model Information"):
    st.write(f"**Selected Model:** {model_choice}")
    st.write(f"**Model Type:** {type(models[model_choice]).__name__}")
    
    if hasattr(models[model_choice], 'feature_names_in_'):
        st.write(f"**Expected Features:** {list(models[model_choice].feature_names_in_)}")
    else:
        st.write(f"**Input Features:** {list(input_data.columns)}")
    
    # Show feature importance if available
    if hasattr(models[model_choice], 'feature_importances_'):
        importances = models[model_choice].feature_importances_
        if hasattr(models[model_choice], 'feature_names_in_'):
            feature_names = models[model_choice].feature_names_in_
        else:
            feature_names = input_data.columns[:len(importances)]
        
        fig = px.bar(
            x=feature_names,
            y=importances,
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "🎧 Spotify Analytics Suite | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)