# syntax=docker/dockerfile:1
#
# Container image for the Feature Prioritisation Assistant Streamlit app.
#
# This is for running the app locally in a container -- the Streamlit Community
# Cloud deployment does not use it; that builds straight from requirements.txt.
# Keep it around for testing the app in a clean environment, or as the starting
# point for a container host (Cloud Run, Fly, ECS, a VM).
#
# Build and run:
#   docker build -t ai-feature-prioritisation-assistant .
#   docker run --rm -p 8505:8505 --env-file .env ai-feature-prioritisation-assistant

FROM python:3.11-slim

# PYTHONUNBUFFERED keeps logs streaming into `docker logs` instead of sitting in
# a buffer until the process exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as a normal user rather than root: nothing here needs root, and a stray
# `pip install` or file write inside a running container should not be able to
# modify the system image.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Dependencies first, in their own layer: application edits then rebuild in
# seconds instead of reinstalling the framework every time.
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user . .

# 8505 rather than Streamlit's default 8501, so this container can run alongside
# the other apps in this repo (8501-8504 are taken). Run it locally with
# `--server.port=8505` to match. STREAMLIT_* variables are the env form of
# .streamlit/config.toml options, set here rather than in that file so the
# committed theme stays deployment-neutral.
ENV STREAMLIT_SERVER_PORT=8505 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8505

# Streamlit's own health endpoint: the honest signal that the script has loaded,
# not merely that the port is open.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8505/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8505", "--server.address=0.0.0.0"]
