# ─────────────────────────────────────────────────────────────────
#  AI Interview Simulator — Dockerfile
#  Compatible with Hugging Face Spaces (docker SDK)
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces expects the app to listen on port 7860
ENV PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user (HF Spaces security requirement)
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Install dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY interview_env/ ./interview_env/
COPY app.py         ./app.py
COPY inference.py   ./inference.py
COPY openenv.yaml   ./openenv.yaml
COPY README.md      ./README.md

# Give ownership to appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

EXPOSE ${PORT}

# Start the FastAPI + Gradio app
CMD ["python", "app.py"]
