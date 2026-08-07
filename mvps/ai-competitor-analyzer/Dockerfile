# syntax=docker/dockerfile:1
#
# Container image for the AI Competitor Analyzer.
#
#   docker build -t competitor-analyzer .
#   docker run --rm -p 8501:8501 --env-file .env competitor-analyzer
#   # then open http://localhost:8501
#
# Small and fast by design. This app holds no models and no corpus -- everything
# it knows arrives from a search API at request time -- so the image is the
# Python base plus a handful of pure-Python wheels, and the build is seconds
# rather than the minutes an embedding-model image takes.
#
# The same properties are what let it run on Streamlit Community Cloud inside
# ~1GB of RAM. If a future change adds a local model here, that deployment target
# is what it breaks.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as a normal user: nothing here needs root, and a stray write inside a
# running container should not be able to modify the system image.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Dependencies first, so editing application code does not reinstall them.
COPY --chown=user:user requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user:user streamlit_app.py pyproject.toml ./
COPY --chown=user:user app ./app
COPY --chown=user:user ui ./ui

# No .env is copied in: keys arrive at run time via --env-file or -e, so an
# image built here can be shared without carrying credentials inside it.

EXPOSE 8501

# Streamlit binds loopback by default, which is unreachable from outside the
# container. headless=true suppresses the browser-open attempt and the email
# prompt on first run.
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
