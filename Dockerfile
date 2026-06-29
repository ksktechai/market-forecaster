# market-forecaster
#
# Target host: Apple-silicon M2 Max (or any x86/arm box with enough RAM).
# NOT a Raspberry Pi -- TimesFM + torch are too heavy for the Pi.
#
# Python 3.11 (not the host's 3.14): torch/timesfm lag the newest CPython.
FROM python:3.11-slim

# git is needed to pip-install timesfm from GitHub; build-essential for any
# native wheels that need compiling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

# Install the lightweight core first (better layer caching), then the heavy
# model deps (torch + timesfm from GitHub).
COPY requirements.txt requirements-model.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-model.txt

COPY market_forecaster ./market_forecaster
COPY pyproject.toml README.md ./

# HF cache lives on a mounted volume (see docker-compose.yml) so weights persist.
RUN mkdir -p /models/hf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run via our entrypoint so uvicorn uses our JSON logging (log_config=None) and
# binds the configured host/port.
CMD ["python", "-m", "market_forecaster"]
