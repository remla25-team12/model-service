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
EXPOSE 5000
CMD ["python", "model_service.py"]