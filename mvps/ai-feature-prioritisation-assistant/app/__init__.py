"""Application package: domain models, the estimator agent, and scoring services.

Layering runs one way only -- ``agents`` may use ``models``, ``core``, and
``services``; nothing in here imports from the Streamlit ``ui`` package.

The split that matters most in this app is inside ``services``:
:mod:`app.services.scoring` holds every number that appears on screen, and it is
pure arithmetic over a factor set. The model layer in ``agents`` produces the
factors and never a score.
"""
