# syntax=docker/dockerfile:1
#
# Container image for the AI Travel Agent.
#
#   docker build -t ai-travel-agent .
#   docker run --rm -p 8506:8506 --env-file .env ai-travel-agent
#   # then open http://localhost:8506

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

EXPOSE 8506

# Streamlit binds loopback by default, which is unreachable from outside the
# container. headless=true suppresses the browser-open attempt and the email
# prompt on first run.
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8506", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
