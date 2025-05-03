FROM python:3.12.9-slim

RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /root/
COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
	pip install -r requirements.txt

COPY model_service.py .

# WHAT PORTS TO EXPOSE? Can this be set in the docker-compose file?
EXPOSE 50000
CMD ["python", "model_service.py"]
# CMD ["python", "model_service.py", "--host=

