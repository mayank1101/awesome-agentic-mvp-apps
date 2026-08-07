"""Streamlit UI layer.

Every module here may import from :mod:`app`; nothing in :mod:`app` imports from
here. That one-way dependency is what keeps the agents, the session store, and
the transcript projection testable without a Streamlit runtime.
"""
