from flask import Flask, request, jsonify
import joblib
import requests
import os

from libml import process_data # TODO


MODEL_URL = os.getenv("MODEL_URL", "http://localhost:5000")
app = Flask(__name__)
model = None

def load_model():
    global model
    model_path = "model.joblib"
    if not os.path.exists(model_path):
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")

    model = joblib.load(model_path)
    print("Model loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input"}), 400

    text = data['text']
    processed_text = process_data(text)

    prediction = model.predict([processed_text])
    return jsonify({"prediction": prediction[0]})

