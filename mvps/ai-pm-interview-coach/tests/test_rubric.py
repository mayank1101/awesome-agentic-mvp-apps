"""Rubric and preset integrity.

These are data-entry tests, and they matter more than they look. A missing anchor
means the evaluator is asked to distinguish a level it was never described, and a
preset naming a dimension that no longer exists fails at report time rather than
here.
"""

import pytest

from app.agents import presets, rubric
from app.models.schemas import ALL_DIMENSIONS, CompanyArchetype, InterviewType, Seniority

# --- rubric ------------------------------------------------------------------


def test_every_dimension_is_covered_exactly_once():
    keys = [dimension.key for dimension in rubric.RUBRIC]
    assert keys == list(ALL_DIMENSIONS)
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("dimension", rubric.RUBRIC, ids=lambda d: d.key)
def test_every_dimension_anchors_all_four_levels(dimension):
    assert set(dimension.anchors) == set(rubric.LEVELS)


@pytest.mark.parametrize("dimension", rubric.RUBRIC, ids=lambda d: d.key)
def test_no_anchor_is_empty_or_a_placeholder(dimension):
    for score, text in dimension.anchors.items():
        assert text.strip(), f"{dimension.key} level {score} is empty"
        # An anchor short enough to be a single word cannot discriminate between
        # levels, which is the one job it has.
        assert len(text.split()) >= 5, f"{dimension.key} level {score} is too vague to apply"


@pytest.mark.parametrize("dimension", rubric.RUBRIC, ids=lambda d: d.key)
def test_anchors_are_distinct_within_a_dimension(dimension):
    texts = list(dimension.anchors.values())
    assert len(texts) == len(set(texts))


@pytest.mark.parametrize("dimension", rubric.RUBRIC, ids=lambda d: d.key)
def test_every_dimension_has_a_label_and_a_test_question(dimension):
    assert dimension.label.strip()
    assert dimension.question.strip().endswith("?")


def test_scale_has_no_midpoint():
    # Four levels split evenly into below-bar and at-bar. A middle value is the
    # failure mode the whole design is built against.
    assert rubric.LEVELS == (1, 2, 3, 4)
    assert set(rubric.BELOW_BAR) | set(rubric.AT_BAR) == set(rubric.LEVELS)
    assert not set(rubric.BELOW_BAR) & set(rubric.AT_BAR)


def test_lookup_by_key_resolves_every_dimension():
    for key in ALL_DIMENSIONS:
        assert rubric.dimension(key).key == key


def test_instruction_text_carries_every_anchor():
    # The anchors *are* the calibration. A summary would leave the model with
    # dimension names and nothing to calibrate against.
    text = rubric.format_for_instructions()
    for dimension in rubric.RUBRIC:
        assert dimension.key in text
        assert dimension.question in text
        for anchor in dimension.anchors.values():
            assert anchor in text


# --- presets -----------------------------------------------------------------


def test_every_interview_type_has_a_preset():
    assert set(presets.INTERVIEW_TYPE_PRESETS) == set(InterviewType.__args__)


def test_every_seniority_has_a_preset():
    assert set(presets.SENIORITY_PRESETS) == set(Seniority.__args__)


def test_every_archetype_has_a_preset():
    assert set(presets.ARCHETYPE_PRESETS) == set(CompanyArchetype.__args__)


@pytest.mark.parametrize("interview_type", InterviewType.__args__)
def test_primaries_are_real_dimensions_and_not_all_of_them(interview_type):
    primary = presets.primary_dimensions(interview_type)

    assert primary, "an interview type with no primaries has nothing to lead with"
    assert set(primary) <= set(ALL_DIMENSIONS)
    assert len(set(primary)) == len(primary)
    # If everything were primary the distinction would buy nothing.
    assert len(primary) < len(ALL_DIMENSIONS)


@pytest.mark.parametrize("interview_type", InterviewType.__args__)
def test_primary_and_secondary_partition_the_rubric(interview_type):
    primary = presets.primary_dimensions(interview_type)
    secondary = presets.secondary_dimensions(interview_type)

    assert not set(primary) & set(secondary)
    assert set(primary) | set(secondary) == set(ALL_DIMENSIONS)


def test_behavioral_does_not_lead_on_metrics():
    # The case that motivated primary dimensions: a behavioural interview cannot
    # legitimately exercise metrics rigour, so next_focus must never point there.
    assert "metrics" not in presets.primary_dimensions("behavioral")


def test_metrics_interview_leads_on_metrics():
    assert "metrics" in presets.primary_dimensions("execution_metrics")


@pytest.mark.parametrize("interview_type", InterviewType.__args__)
def test_every_interview_type_supplies_a_brief_and_probe_angles(interview_type):
    preset = presets.INTERVIEW_TYPE_PRESETS[interview_type]

    assert preset.question_brief.strip()
    assert len(preset.probe_angles) >= 2
    assert all(angle.strip() for angle in preset.probe_angles)


@pytest.mark.parametrize("seniority", Seniority.__args__)
def test_every_seniority_states_a_bar(seniority):
    assert presets.SENIORITY_PRESETS[seniority].bar.strip()


@pytest.mark.parametrize("archetype", CompanyArchetype.__args__)
def test_every_archetype_states_what_it_optimises_for(archetype):
    preset = presets.ARCHETYPE_PRESETS[archetype]

    assert preset.optimises_for.strip()
    assert preset.context.strip()


def test_archetypes_optimise_for_different_things():
    # The axis exists precisely because the same answer is strong at one
    # archetype and weak at another. Identical values would make it decorative.
    values = [preset.optimises_for for preset in presets.ARCHETYPE_PRESETS.values()]
    assert len(values) == len(set(values))
