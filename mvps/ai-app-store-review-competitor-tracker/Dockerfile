# syntax=docker/dockerfile:1
#
# Container image for the App Store Review Competitor Tracker.
#
#   docker build -t review-tracker .
#   docker run --rm -p 8501:8501 --env-file .env review-tracker
#   # then open http://localhost:8501
#
# Small and fast by design: this app holds no models and no corpus, so the
# image is the Python base plus a handful of pure-Python wheels.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as a normal user: nothing here needs root.
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

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
