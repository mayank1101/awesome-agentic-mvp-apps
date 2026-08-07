"""Streamlit presentation layer.

Split by screen region rather than by widget type:

* :mod:`ui.state` -- the session-state keys and their accessors.
* :mod:`ui.sidebar` -- the brief form and the progress checklist.
* :mod:`ui.document` -- the main pane: outline header, streaming sections,
  export.

Nothing here talks to a model directly; it all goes through :mod:`app.agents`.
"""
