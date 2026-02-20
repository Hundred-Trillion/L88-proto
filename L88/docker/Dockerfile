# L88/docker/Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app
COPY L88/requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir -r requirements.txt
COPY L88 /app/L88
ENV PYTHONPATH=/app/L88
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
