from flask import Flask, request, jsonify
import joblib
import pickle
import requests
import os
import csv
from flasgger import Swagger

from libml.text_preprocessing import  preprocess_input
from libml import __version__ as lib_ml_version
import subprocess

CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./cache")
TRAINED_MODEL_VERSION = os.getenv("TRAINED_MODEL_VERSION", "v0.1.0")

MODEL_URL = f"https://github.com/remla25-team12/model-training/releases/download/{TRAINED_MODEL_VERSION}/Classifier_Sentiment_Model.joblib"
VEC_URL = f"https://github.com/remla25-team12/model-training/releases/download/{TRAINED_MODEL_VERSION}/c1_BoW_Sentiment_Model.pkl"
FEEDBACK_FILE_PATH = os.getenv("FEEDBACK_FILE_PATH","./feedback/feedback_dump.tsv")
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)
swagger = Swagger(app)
model = None
new_data = []

def get_version():
    try:
        with open("version.txt") as f:
            version = f.read().strip()
            # Strip -pre.* if present
            base_version = version.split('-')[0]

            # Split into MAJOR.MINOR.PATCH
            parts = base_version.split('.')
            major = parts[0]
            minor = parts[1]
            patch = int(parts[2])

            # If version.txt was pre-release → subtract 1 from PATCH
            if '-' in version or 'pre' in version:
                patch = max(0, patch - 1)

            return f"{major}.{minor}.{patch}"
    except Exception:
        return "unknown"
    
def load_model():
    """
    Method for loading the model and vectorizer from the specified URLs if they do not exist locally.
    """
    global model, cv, new_data

    versioned_cache_dir = os.path.join(CACHE_DIR, TRAINED_MODEL_VERSION)
    os.makedirs(versioned_cache_dir, exist_ok=True)

    model_path = os.path.join(versioned_cache_dir, "Classifier_Sentiment_Model.joblib")
    vec_path = os.path.join(versioned_cache_dir, "c1_BoW_Sentiment_Model.pkl")

    # get model if non existent
    if not os.path.exists(model_path):
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")

    # get vc non existent
    if not os.path.exists(vec_path):
        print(f"Downloading vectorizer from {VEC_URL}...")
        response = requests.get(VEC_URL)
        if response.status_code == 200:
            with open(vec_path, "wb") as f:
                f.write(response.content)
            print("Vectorizer downloaded successfully.")
        else:
            raise Exception(f"Failed to download vectorizer. Status code: {response.status_code}")

    # load both
    model = joblib.load(model_path)
    with open(vec_path, "rb") as f:
        cv = pickle.load(f)
    print("Model and vectorizer loaded successfully.")

@app.route('/version', methods=['GET'])
def version():
    """
    Get library version from lib-ml
    ---
    summary: Get library version
    parameters: []
    responses:
      200:
        description: Successfully returns version
    """
    return jsonify({"version": get_version()})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment from input text
    ---
    summary: Predict sentiment
    parameters:
    - in: body
        name: body
        required: true
        schema:
        type: object
        required:
            - text
        properties:
            text:
            type: string
    responses:
      200:
        description: Sentiment prediction result
      400:
        description: Invalid input
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Input is invalid"}), 400

    text = data['text']
    processed_text = preprocess_input(text,cv)
    print("My preprocessed text is: ", processed_text.shape)
    print("Predicting...")
    prediction = model.predict(processed_text)
    print("Prediction complete: ", str(prediction[0]))
    return jsonify({"prediction": str(prediction[0])}), 200


@app.route('/new_data', methods=['POST'])
def new_data_save():
    """
    Submit new data for feedback
    ---
    summary: Submit new data
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - text
            - sentiment
          properties:
            text:
              type: string
            sentiment:
              type: integer
    responses:
      200:
        description: Data added successfully
      400:
        description: Invalid input
      500:
        description: Feedback file write error
    """
    data = request.get_json()
    if not data or 'text' not in data or 'sentiment' not in data:
        return jsonify({"error": "Input is invalid"}), 400
    text = data['text']
    sentiment = data['sentiment']
    if not isinstance(sentiment, int):
        return jsonify({"error": "Sentiment must be an integer"}), 400
    if sentiment not in [0, 1]:
        return jsonify({"error": "Sentiment must be 0 or 1"}), 400
    if not isinstance(text, str):
        return jsonify({"error": "Text must be a string"}), 400
    if len(text) < 1:
        return jsonify({"error": "Text must be at least 1 character long"}), 400

    new_data.append({"text": text, "sentiment": sentiment})
    print(f"[NEW DATA] {text} -> {sentiment}")

    # Append to .tsv file
    try:
        os.makedirs(os.path.dirname(FEEDBACK_FILE_PATH), exist_ok=True)
        file_exists = os.path.isfile(FEEDBACK_FILE_PATH)
        with open(FEEDBACK_FILE_PATH, mode='a', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            if not file_exists:
                writer.writerow(["Review", "Liked"])
            writer.writerow([text.strip(), sentiment])
    except Exception as e:
        print(f"Error writing to feedback file: {e}")
        return jsonify({"error": f"Failed to write to TSV: {str(e)}"}), 500

    return jsonify({"message": "Data added successfully"}), 200

if __name__ == "__main__":
    load_model()
    port = int(os.getenv("MODEL_SERVICE_PORT", 5000))
    app.run(host="0.0.0.0", port=port)
