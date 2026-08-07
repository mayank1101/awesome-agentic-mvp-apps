"""Tests for the fabrication guard.

This is the app's one hard guarantee, so these tests are the ones that matter
most. They are written from both sides: a faithful rewrite must pass cleanly (a
guard that cries wolf gets turned off), and each class of invention must be
caught.
"""

from app.services import provenance


def test_faithful_rewrite_passes(resume_text: str) -> None:
    tailored = (
        "# Priya Raman\n"
        "priya.raman@example.com | +91 98765 43210 | Bengaluru\n\n"
        "## Summary\n"
        "Backend engineer with 6 years building payment services in Python.\n\n"
        "## Experience\n"
        "### Senior Backend Engineer, Fintrail\n"
        "*Mar 2021 - Present, Bengaluru*\n"
        "- Designed REST APIs serving 12000 requests per minute\n"
        "- Rebuilt the settlement pipeline in Python, cutting reconciliation time by 40%\n"
    )
    assert provenance.check(resume_text, tailored) == []


def test_invented_number_is_caught(resume_text: str) -> None:
    tailored = "- Cut reconciliation time by 65%\n"
    violations = provenance.check(resume_text, tailored)
    assert [v.kind for v in violations] == ["number"]
    assert violations[0].text == "65%"


def test_invented_tool_is_caught(resume_text: str) -> None:
    tailored = "- Ran production workloads on Kubernetes\n"
    violations = provenance.check(resume_text, tailored)
    assert any(v.kind == "name" and v.text == "Kubernetes" for v in violations)


def test_invented_employer_is_caught(resume_text: str) -> None:
    tailored = "### Staff Engineer, Northwind Pay\n"
    violations = provenance.check(resume_text, tailored)
    assert any(v.text == "Northwind" for v in violations)


def test_invented_contact_details_are_caught(resume_text: str) -> None:
    tailored = "priya@northwind.example | https://github.com/someone-else\n"
    kinds = {v.kind for v in provenance.check(resume_text, tailored)}
    assert kinds == {"contact"}


def test_sentence_initial_words_are_not_flagged(resume_text: str) -> None:
    """A capitalised first word is grammar, not a name -- flagging it kills the guard."""
    tailored = "- Delivered payment services. Reduced deploy time using Docker.\n"
    assert provenance.check(resume_text, tailored) == []


def test_lowercase_technology_present_in_original_passes(resume_text: str) -> None:
    tailored = "## Skills\nPython, Django, PostgreSQL, Docker\n"
    assert provenance.check(resume_text, tailored) == []


def test_number_formatting_differences_are_not_inventions(resume_text: str) -> None:
    """`12,000` and `12000` are the same fact written twice."""
    tailored = "- Designed REST APIs serving 12,000 requests per minute\n"
    assert provenance.check(resume_text, tailored) == []


def test_markdown_syntax_is_not_treated_as_content(resume_text: str) -> None:
    tailored = "## Skills\n**Python**, *Django*, `Docker`\n"
    assert provenance.check(resume_text, tailored) == []


def test_violations_are_deduplicated(resume_text: str) -> None:
    tailored = "- Used Kubernetes daily\n- Kubernetes operators shipped\n"
    violations = provenance.check(resume_text, tailored)
    assert len([v for v in violations if v.text.lower() == "kubernetes"]) == 1


def test_describe_names_the_offender(resume_text: str) -> None:
    violations = provenance.check(resume_text, "- Cut latency by 99%\n")
    described = provenance.describe(violations)
    assert described and "99%" in described[0]


# --------------------------------------------------------------------------- #
# Extraction artefacts
#
# Every case below comes from a real resume that the guard wrongly refused: a
# LaTeX template glued icon names onto the contact line, and pypdf split a word
# at a kerning pair. A guard that rejects the candidate's own email address is
# worse than no guard, because strict mode then refuses the rewrite.
# --------------------------------------------------------------------------- #


def test_a_word_split_by_kerning_is_not_an_invention() -> None:
    original = "Education\nIndian Institute of T echnology (IIT) Jammu\n2019 - 2021"
    tailored = "### Indian Institute of Technology (IIT) Jammu\n"

    assert provenance.check(original, tailored) == []


def test_an_email_glued_to_a_template_icon_is_not_an_invention() -> None:
    original = "Mayank Sharma\n/phone(+91) - 9691314634 /envelopename.surname@example.com"
    tailored = "name.surname@example.com\n"

    assert provenance.check(original, tailored) == []


def test_squashing_does_not_excuse_a_real_invention() -> None:
    """The fallback must not turn the guard off."""
    original = "Built Django services in Python at Fintrail."
    tailored = "- Shipped Kubernetes operators\n"

    assert any(v.text == "Kubernetes" for v in provenance.check(original, tailored))


def test_short_tokens_still_need_an_exact_match() -> None:
    """Containment is meaningless for short tokens: "SAP" sits inside "sapling"."""
    original = "Worked on sapling data pipelines."
    tailored = "- Migrated SAP financials\n"

    assert any(v.text == "SAP" for v in provenance.check(original, tailored))


def test_a_sentence_final_word_is_not_read_as_a_technical_name() -> None:
    """A word ending a sentence carries a full stop, which is not evidence of a name."""
    original = "Built LLM retrieval systems and evaluation harnesses."
    tailored = "- Proven track record with LLM applications. Experience in evaluation.\n"

    assert provenance.check(original, tailored) == []


def test_real_dotted_names_are_still_checked() -> None:
    original = "Built services in Python."
    tailored = "- Shipped Node.js and scikit-learn pipelines\n"

    flagged = {v.text for v in provenance.check(original, tailored)}
    assert "Node.js" in flagged
    assert "scikit-learn" in flagged
