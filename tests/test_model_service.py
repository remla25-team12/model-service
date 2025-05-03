import pytest
from flask import json
from model_service import app, load_model

@pytest.fixture
def client():
    # Set up the Flask test client
    app.testing = True
    with app.test_client() as client:
        yield client

def test_load_model():
    # Test if the model loads successfully
    try:
        load_model()
    except Exception as e:
        pytest.fail(f"Model failed to load: {e}")

def test_predict_valid_input(client):
    # Test the /predict endpoint with valid input
    load_model()  # Ensure the model is loaded
    response = client.post('/predict', json={"text": "This is a great product!"})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data

def test_predict_invalid_input(client):
    # Test the /predict endpoint with invalid input
    response = client.post('/predict', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["error"] == "Input is invalid"
