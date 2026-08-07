"""Supporting services: the scales, the arithmetic, parsing, guardrails, export.

:mod:`app.services.scales` and :mod:`app.services.scoring` between them own every
number the app displays. Neither imports a model client, and that is the point --
scoring a backlog is a pure function of its factors, which is what makes an edit
re-rank instantly and what makes the whole thing testable without a provider.
"""
