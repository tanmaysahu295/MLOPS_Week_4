# Use lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies only if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (to use Docker layer caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only required app files (not whole repo)
COPY app.py .
COPY artifacts/model_1.joblib ./artifacts/model_1.joblib
COPY data/iris.csv ./data/iris.csv

# Expose port
ENV PORT=5000
EXPOSE 5000

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
