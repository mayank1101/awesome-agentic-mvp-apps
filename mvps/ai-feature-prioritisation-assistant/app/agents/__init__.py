"""The agent layer: one estimator, one call, no arithmetic.

Deliberately small. The interesting logic in this app is not here -- it is in
:mod:`app.services.scoring`, which is pure and testable. This package exists to
turn rough prose into an anchored factor set and then get out of the way.

Public surface:

* :func:`~app.agents.estimator.estimate_backlog` -- blocking, for the UI.
* :func:`~app.agents.estimator.aestimate_backlog` -- the coroutine underneath.
"""

from app.agents.estimator import aestimate_backlog, estimate_backlog

__all__ = ["aestimate_backlog", "estimate_backlog"]
