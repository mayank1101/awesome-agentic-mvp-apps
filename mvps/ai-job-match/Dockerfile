# syntax=docker/dockerfile:1
#
# Container image for the resume/job fit app.
#
#   docker build -t ai-job-match .
#   docker run --rm -p 8503:8503 --env-file .env ai-job-match
#   # then open http://localhost:8503
#
# Every dependency is a pure-Python wheel: no model weights, no vector store, no
# native toolchain. The build is seconds rather than the minutes an
# embedding-model image takes, and the same property is what lets this run on
# Streamlit Community Cloud inside ~1GB of RAM. If a future change adds a local
# model here, that deployment target is what it breaks.

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

# No .env is copied in: keys arrive at run time via --env-file or -e, so an image
# built here can be shared without carrying credentials inside it.

EXPOSE 8503

# Streamlit binds loopback by default, which is unreachable from outside the
# container. headless=true suppresses the browser-open attempt and the email
# prompt on first run. uploadMaxSize matches MAX_UPLOAD_BYTES so a file the app
# would reject is refused by the server first, before it is buffered in memory.
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8503", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.maxUploadSize=5"]
