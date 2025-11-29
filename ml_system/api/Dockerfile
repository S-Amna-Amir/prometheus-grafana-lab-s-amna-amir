# -----------------------------------------------------------
# Base Image
# -----------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------
# Install system libs
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# Working directory
# -----------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------
# Copy requirements
# (the requirements.txt is inside ml_system/api)
# -----------------------------------------------------------
COPY ml_system/api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------
# Copy application code
# -----------------------------------------------------------
COPY ml_system/api/main.py .

# -----------------------------------------------------------
# Copy ML artifacts from project root
# -----------------------------------------------------------
COPY model.joblib .
COPY scaler.joblib .
COPY feature_list.json .

# -----------------------------------------------------------
# Expose port
# -----------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------
# Start API server
# -----------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
