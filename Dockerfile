# ==============================================================================
# Stage 1: Builder - Compile dependencies into an isolated virtual environment
# ==============================================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip install -r requirements.txt

# ==============================================================================
# Stage 2: Production Runtime - Hardened, non-root image for Google Cloud Run
# ==============================================================================
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="argon-learn" \
      org.opencontainers.image.description="ArgonLearn: ELI5-to-Hero Multi-Agent AI Visualizer built with Google ADK" \
      org.opencontainers.image.vendor="ArgonLearn AI"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    APP_ENV="production"

# Install runtime multimedia libraries required for Manim vector rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root system user for least-privilege execution on Cloud Run
RUN groupadd --gid 10001 argon && \
    useradd --uid 10001 --gid argon --shell /bin/bash --create-home argon

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=argon:argon app/ /app/app/
COPY --chown=argon:argon tests/ /app/tests/
COPY --chown=argon:argon README.md /app/README.md

USER argon

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/healthz || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 2 --proxy-headers"]
