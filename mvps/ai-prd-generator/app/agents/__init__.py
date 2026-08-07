"""Agent layer: everything that talks to a model.

This is the package's public surface. Import from here rather than reaching into
the submodules, so the internal split (client / presets / prompts / agents) can
change without touching callers::

    from app.agents import SCOPE_PRESETS, generate_outline, stream_section
"""

from app.agents.prd_agents import (
    agenerate_outline,
    agenerate_section,
    astream_section,
    generate_outline,
    generate_section,
    stream_section,
)
from app.agents.presets import (
    LENGTH_PRESETS,
    SCOPE_PRESETS,
    WORDS_PER_PAGE,
    LengthPreset,
    ScopePreset,
)

__all__ = [
    "LENGTH_PRESETS",
    "SCOPE_PRESETS",
    "WORDS_PER_PAGE",
    "LengthPreset",
    "ScopePreset",
    "agenerate_outline",
    "agenerate_section",
    "astream_section",
    "generate_outline",
    "generate_section",
    "stream_section",
]
