"""Streamlit presentation layer.

Split by screen region rather than by widget type:

* :mod:`ui.state` -- the session-state keys and their accessors.
* :mod:`ui.input_form` -- the paste box, the product context, the estimate button.
* :mod:`ui.results` -- the ranking, the divergence notes, the factor editor, the
  per-feature reasoning, and the exports.

Nothing here talks to a model directly; it all goes through :mod:`app.agents`.
And nothing here computes a score: :mod:`ui.results` renders what
:func:`app.services.scoring.score_backlog` returned and does no arithmetic of
its own, so there is exactly one place in the codebase where a RICE number is
produced.
"""
