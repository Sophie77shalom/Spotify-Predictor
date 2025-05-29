# Mini Flask app (save as app.py)
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('spotify_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data['danceability'], data['energy'], data['duration_ms'], data['tempo']]
    prediction = model.predict([features])[0]
    return jsonify({'liked': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)