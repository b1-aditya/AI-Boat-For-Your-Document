# Python 3.12 slim base image (stable wheels for PyTorch, ChromaDB, FlashRank, PyMuPDF)
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logs for Render
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_TYPE=streamlit

WORKDIR /app

# Install minimal system dependencies needed by some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached Docker layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code and starter data
COPY ingest.py rag.py app.py api.py .env.example ./
COPY data/ ./data/

# Create directory for persistent ChromaDB mount (on Render Persistent Disk)
RUN mkdir -p /app/chroma_db

# Default ports (Render injects $PORT dynamically at runtime)
EXPOSE 8501 8000

# Run Streamlit UI by default, or FastAPI if APP_TYPE=api is set
CMD ["sh", "-c", "if [ \"$APP_TYPE\" = \"api\" ]; then uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}; else streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true; fi"]
