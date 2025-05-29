import joblib
import os

# Create models directory if not exists
os.makedirs("models/label_encoders", exist_ok=True)

# Conversion mapping (old -> new)
conversion_map = {
    "explicit_predictor.joblib": "models/explicit_predictor.pkl",
    "spotify_genre_classifier.joblib": "models/genre_classifier.pkl",
    "spotify_popularity_predictor.joblib": "models/popularity_regressor.pkl",
    "spotify_pop_predictor.joblib": "models/popularity_classifier.pkl"
}

# Convert all files
for old_name, new_name in conversion_map.items():
    try:
        model = joblib.load(old_name)
        joblib.dump(model, new_name)
        print(f"Converted {old_name} -> {new_name}")
    except Exception as e:
        print(f"Failed to convert {old_name}: {str(e)}")