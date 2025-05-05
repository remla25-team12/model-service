# Model Service

This Flask-based REST API exposes the trained sentiment model for prediction.

It loads the model from a versioned release and uses the `lib-ml` for preprocessing.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo>/model-service.git
   cd model-service
    ```
2. Install dependencies:
    ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
