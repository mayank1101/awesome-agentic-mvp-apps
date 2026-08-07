"""Application package: domain models, agent layer, and supporting services.

Layering runs one way only -- ``agents`` may use ``models``, ``core``, and
``services``; nothing in here imports from the Streamlit ``ui`` package.
"""
