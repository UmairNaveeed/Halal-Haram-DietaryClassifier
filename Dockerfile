# Dockerfile for Hugging Face Spaces (Docker SDK)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache base model to ensure instant cold start on Hugging Face Spaces
RUN python3 -c "from transformers import BertTokenizer, BertForSequenceClassification; BertTokenizer.from_pretrained('bert-base-uncased'); BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)"

# Copy application files
COPY . .

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Start FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
