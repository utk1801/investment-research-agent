FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence-transformers + chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject + install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY grafana/ ./grafana/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Streamlit
EXPOSE 8501

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]