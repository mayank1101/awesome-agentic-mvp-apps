"""Main pane: the outline header, the streaming sections, and the export.

Rendering happens in two passes, and the order matters.

**Pass 1** lays out every section up front -- real content where it exists, a
skeleton where it does not -- so the reader sees the shape of the whole document
immediately rather than watching it grow. The Regenerate buttons must also be
created in this pass: a Streamlit button only reports the click that happened on
the run that drew it.

**Pass 2** walks the gaps and streams into the placeholders pass 1 reserved,
updating the sidebar checklist as each section lands.
"""

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from app.agents import LENGTH_PRESETS, SCOPE_PRESETS, WORDS_PER_PAGE, stream_section
from app.core.config import get_settings
from app.models.schemas import PRDDocument, PRDInput, PRDOutline, PRDSection
from app.services.guardrails import redact_secrets, sanitize_markdown
from app.services.markdown_renderer import count_words, render_markdown
from ui import state
from ui.sidebar import SidebarSlots

#: Status icon per section state, in Streamlit's material-icon markdown syntax.
STATUS_ICONS = {
    "done": ":green[:material/check_circle:]",
    "writing": ":violet[:material/stylus_note:]",
    "pending": ":gray[:material/radio_button_unchecked:]",
    "error": ":red[:material/error:]",
}

_SKELETON_HEIGHT = 170
_EMPTY_STATE_HEIGHT = 520


def render_empty_state() -> None:
    """Draw the placeholder shown before any PRD has been generated."""
    with st.container(
        height=_EMPTY_STATE_HEIGHT,
        horizontal_alignment="center",
        vertical_alignment="center",
    ):
        st.markdown("## :gray[:material/edit_document:]", text_alignment="center")
        st.markdown("#### No PRD yet", text_alignment="center")
        st.caption(
            "Fill in the brief in the sidebar and hit **Generate PRD**.",
            text_alignment="center",
        )


def render_document(prd_input: PRDInput, outline: PRDOutline, slots: SidebarSlots) -> None:
    """Render the document, writing any sections that are still missing.

    Args:
        prd_input: The brief this document was generated from.
        outline: The plan being filled in.
        slots: Sidebar placeholders to report progress into.
    """
    _render_header(prd_input, outline)

    # Progress lives in the main pane, not the sidebar: Streamlit auto-collapses
    # the sidebar on narrower windows, and progress must stay visible.
    progress_slot = st.empty()

    section_slots, regenerate_flags = _render_section_layout(outline)
    _render_initial_checklist(outline, slots, regenerate_flags)
    _write_missing_sections(
        prd_input, outline, slots, section_slots, regenerate_flags, progress_slot
    )
    _render_progress_or_export(outline, slots, progress_slot)


def _render_header(prd_input: PRDInput, outline: PRDOutline) -> None:
    """Draw the badge row and the document title."""
    scope_icon = "extension" if prd_input.scope == "feature" else "inventory_2"
    st.markdown(
        f":green-badge[:material/{scope_icon}: {SCOPE_PRESETS[prd_input.scope]['label']}] "
        f":violet-badge[:material/group: {prd_input.audience}] "
        f":blue-badge[:material/straighten: {LENGTH_PRESETS[prd_input.length]['label']}] "
        f":gray-badge[:material/list: {len(outline.sections)} sections]"
    )
    st.title(outline.title)


def _render_section_layout(
    outline: PRDOutline,
) -> tuple[list[DeltaGenerator], list[bool]]:
    """Pass 1: draw every section heading, body-or-skeleton, and button.

    Returns:
        A placeholder per section for pass 2 to stream into, and a flag per
        section saying whether its Regenerate button was clicked this run.
    """
    section_slots: list[DeltaGenerator] = []
    regenerate_flags: list[bool] = []
    sections = state.get_sections()

    for index, spec in enumerate(outline.sections):
        with st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="bottom",
        ):
            st.subheader(spec.title)
            regenerate_flags.append(
                st.button(
                    "Regenerate",
                    key=f"regen_{index}",
                    icon=":material/refresh:",
                    type="tertiary",
                    help="Rewrite this section",
                )
            )

        slot = st.empty()
        existing = sections[index]
        if existing is not None and not regenerate_flags[index]:
            slot.markdown(existing.content)
        else:
            with slot:
                st.skeleton(height=_SKELETON_HEIGHT)
        section_slots.append(slot)

    return section_slots, regenerate_flags


def _render_initial_checklist(
    outline: PRDOutline, slots: SidebarSlots, regenerate_flags: list[bool]
) -> None:
    """Show each section's status as it stands before this run writes anything."""
    sections = state.get_sections()
    for index, spec in enumerate(outline.sections):
        written = sections[index] is not None and not regenerate_flags[index]
        slots.checklist[index].markdown(
            _checklist_markdown(spec.title, "done" if written else "pending")
        )


def _write_missing_sections(
    prd_input: PRDInput,
    outline: PRDOutline,
    slots: SidebarSlots,
    section_slots: list[DeltaGenerator],
    regenerate_flags: list[bool],
    progress_slot: DeltaGenerator,
) -> None:
    """Pass 2: stream every section that is missing or was asked to be rewritten.

    A failure is contained to its own section: the error is shown in place, the
    checklist marks it, and the loop moves on. Free-tier models rate-limit
    often, and losing a nine-section document to one 429 is not a useful
    outcome -- the Regenerate button is the retry.
    """
    total = len(outline.sections)
    sections = state.get_sections()

    for index, spec in enumerate(outline.sections):
        if sections[index] is not None and not regenerate_flags[index]:
            continue

        done_count = sum(1 for section in sections if section is not None)
        progress_slot.progress(done_count / total, text=f"Writing section {index + 1} of {total}")
        slots.checklist[index].markdown(_checklist_markdown(spec.title, "writing"))

        with section_slots[index]:
            try:
                content = st.write_stream(
                    stream_section(prd_input, outline, spec.title, spec.summary)
                )
            except Exception as exc:  # noqa: BLE001 - one section must not sink the document
                st.error(
                    f"Section generation failed: {redact_secrets(str(exc))}",
                    icon=":material/error:",
                )
                content = None

        if content is None:
            slots.checklist[index].markdown(_checklist_markdown(spec.title, "error"))
            continue

        # Sanitise on the way into state, so both the re-render on the next run
        # and the exported file carry the cleaned text. The stream itself is
        # rendered raw -- Streamlit escapes HTML, and holding tokens back to
        # sanitise them would cost the live typing effect for no real gain.
        content = _clean(content)
        state.set_section(index, PRDSection(title=spec.title, content=content))
        section_slots[index].markdown(content)
        slots.checklist[index].markdown(_checklist_markdown(spec.title, "done"))


def _render_progress_or_export(
    outline: PRDOutline, slots: SidebarSlots, progress_slot: DeltaGenerator
) -> None:
    """Show the download once every section has landed, otherwise the progress bar."""
    sections = state.get_sections()
    written = sum(1 for section in sections if section is not None)
    total = len(sections)

    if written < total:
        progress_slot.progress(written / total, text=f"{written} of {total} sections written")
        return

    progress_slot.empty()
    document = PRDDocument(title=outline.title, sections=sections)
    markdown = render_markdown(document)
    words = count_words(markdown)

    with slots.export.container():
        st.download_button(
            "Download Markdown",
            data=markdown,
            file_name=f"{outline.title.replace(' ', '_')}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )
        st.caption(f"{words:,} words · ~{max(1, round(words / WORDS_PER_PAGE))} pages")


def _clean(markdown: str) -> str:
    """Run generated Markdown through the output guardrails, if enabled."""
    if not get_settings().guardrails_enabled:
        return markdown
    return sanitize_markdown(markdown)


def _checklist_markdown(title: str, status: str) -> str:
    """Format one sidebar checklist line for a section in the given status."""
    icon = STATUS_ICONS[status]
    if status == "writing":
        return f"{icon} **{title}**"
    if status == "pending":
        return f"{icon} :gray[{title}]"
    return f"{icon} {title}"
