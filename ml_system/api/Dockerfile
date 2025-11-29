# -----------------------------------------------------------
# Base Image (Small Footprint Python + Required Libs)
# -----------------------------------------------------------
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# Working Directory
# -----------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------
# Copy Requirements First (Allows Docker Layer Caching)
# -----------------------------------------------------------
COPY requirements.txt .

# -----------------------------------------------------------
# Install Python Dependencies
# -----------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------
# Copy Application Files
# -----------------------------------------------------------
COPY main.py .
COPY model.joblib .
COPY scaler.joblib .
COPY feature_list.json .

# -----------------------------------------------------------
# Expose FastAPI Port
# -----------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------
# Default command (Uvicorn Server)
# -----------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
