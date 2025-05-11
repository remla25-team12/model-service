# Stage 1: Build dependencies
FROM python:3.12.9-slim as builder
RUN apt-get update && apt-get install -y git && apt-get clean
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Stage 2: Final image
FROM python:3.12.9-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY model_service.py .

# Set default env variables (can be overridden by Kubernetes/Compose)
ENV MODEL_URL=https://github.com/remla25-team12/model-training/releases/download/v0.1.0/Classifier_Sentiment_Model.joblib
ENV VEC_URL=https://github.com/remla25-team12/model-training/releases/download/v0.1.0/c1_BoW_Sentiment_Model.pkl
ENV MODEL_CACHE_DIR=/app/cache
ENV FEEDBACK_FILE_PATH=data/feedback_dump.tsv

# Ensure the cache directory exists
RUN mkdir -p $MODEL_CACHE_DIR

EXPOSE 5000
CMD ["python", "model_service.py"]