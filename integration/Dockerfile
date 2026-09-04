FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies required for LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files and install python packages
COPY integration/api/requirements.txt /app/integration/api/requirements.txt
COPY integration/dashboard/requirements.txt /app/integration/dashboard/requirements.txt

RUN pip install --no-cache-dir -r /app/integration/api/requirements.txt
RUN pip install --no-cache-dir -r /app/integration/dashboard/requirements.txt

# Copy project files into container
COPY . /app

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Launch both FastAPI backend service and Streamlit dashboard concurrently
CMD uvicorn integration.api.main:app --host 0.0.0.0 --port 8000 & streamlit run integration/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
