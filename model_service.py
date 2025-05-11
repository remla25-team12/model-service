from flask import Flask, request, jsonify
import joblib
import pickle
import requests
import os
import csv

from libml.text_preprocessing import  preprocess_input
from libml import __version__ as lib_ml_version

CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./cache")
MODEL_URL = os.getenv("MODEL_URL", "https://github.com/remla25-team12/model-training/releases/download/v0.1.0/Classifier_Sentiment_Model.joblib")
VEC_URL = os.getenv("VEC_URL", "https://github.com/remla25-team12/model-training/releases/download/v0.1.0/c1_BoW_Sentiment_Model.pkl")
FEEDBACK_FILE_PATH = os.getenv("FEEDBACK_FILE_PATH","./feedback/feedback_dump.tsv")
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)
model = None
new_data = []

def load_model():
    """
    Loads the model and vectorizer from the specified URLs if they do not exist locally.
    """
    global model, cv, new_data
    model_path = os.path.join(CACHE_DIR, "Classifier_Sentiment_Model.joblib")
    vec_path = os.path.join(CACHE_DIR, "c1_BoW_Sentiment_Model.pkl")

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
    Get the current version of lib-ml
    ---
    summary: Get library version
    responses:
      200:
        description: Successfully returns version
        content:
          application/json:
            schema:
              type: object
              properties:
                version:
                  type: string
                  example: "1.2.3"
    """
    return jsonify({"version": lib_ml_version})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to the /predict endpoint.
    This method validates the input JSON payload to ensure it contains the required field:
    - 'text': A non-empty string representing the text data.
    If the input is valid, the text is preprocessed and passed to the model for prediction.
    Otherwise, an appropriate error message is returned.
    Returns:
        - 200: If the prediction is successful, with the prediction result.
        - 400: If the input is invalid, with an error message explaining the issue.
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
    Handles POST requests to the /new_data endpoint.

    This method validates the input JSON payload to ensure it contains the required fields:
    - 'text': A non-empty string representing the text data.
    - 'sentiment': An integer (0 or 1) representing the sentiment.

    If the input is valid, the data is appended to the global `new_data` list.
    Otherwise, an appropriate error message is returned.

    Returns:
        - 200: If the data is successfully added.
        - 400: If the input is invalid, with an error message explaining the issue.
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
        print(f"❌ Error writing to feedback file: {e}")
        return jsonify({"error": f"Failed to write to TSV: {str(e)}"}), 500

    return jsonify({"message": "Data added successfully"}), 200

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
