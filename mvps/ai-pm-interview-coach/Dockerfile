# syntax=docker/dockerfile:1
#
# Container image for the AI PM Interview Coach Streamlit app.
#
# Build and run:
#   docker build -t pm-interview-coach .
#   docker run --rm -p 8502:8502 --env-file .env pm-interview-coach
#
# Then open http://localhost:8502.
#
# Nothing else has to run alongside it. Interviews live in Streamlit's per-browser
# session state, so there is no database, no cache, and no volume to mount -- and
# nothing to clean up, because a conversation ends with the browser session or the
# container, whichever goes first.
#
# --env-file rather than baked-in values: .env holds a provider API key and is
# excluded by .dockerignore precisely so it never reaches the image or a registry.
#
# One caveat if this ever leaves a single container: because state is per-session
# and in-process, replicas need sticky sessions, or a reconnect lands on a replica
# with no interview in it. Streamlit's websocket already wants stickiness, so this
# is inherent rather than a new constraint.

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

# 8502 rather than Streamlit's default 8501, so this app and the PRD generator
# can run side by side without a port clash. STREAMLIT_* variables are the env
# form of .streamlit/config.toml options, set here rather than in that file so
# the committed theme stays deployment-neutral.
ENV STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8502

# Streamlit's own health endpoint: the honest signal that the script has loaded,
# not merely that the port is open.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8502/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]
