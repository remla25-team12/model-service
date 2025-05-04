from flask import Flask, request, jsonify
import joblib
import pickle
import requests
import os

from libml.text_preprocessing import  preprocess_input

MODEL_URL = os.getenv("MODEL_URL", "https://github.com/remla25-team12/model-training/releases/download/v0.1.0/Classifier_Sentiment_Model.joblib")
VEC_URL = os.getenv("VEC_URL", "https://github.com/remla25-team12/model-training/releases/download/v0.1.0/c1_BoW_Sentiment_Model.pkl")

app = Flask(__name__)
model = None

def load_model():
    global model, cv
    model_path = "Classifier_Sentiment_Model.joblib"
    vec_path = "c1_BoW_Sentiment_Model.pkl"

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

@app.route('/predict', methods=['POST'])
def predict():
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

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
