"""Tests for parsing the estimator's reply.

No provider is contacted here. These exercise the coercion path, which is where
a large single-shot JSON reply actually goes wrong: a fence around the object, a
sentence in front of it, one row out of twenty-five with a string where a number
belongs.
"""

import json
from types import SimpleNamespace

import pytest

from app.agents.estimator import _coerce_estimate, _payload_from
from app.core.exceptions import EstimateParseError
from app.models.schemas import BacklogEstimate

_ENTRY = {
    "id": "F1",
    "reach": 1200,
    "reach_rationale": "the note says every seller, and the context says 1,200 sellers",
    "impact": 2,
    "impact_rationale": "core job, clearly better",
    "confidence": 0.8,
    "confidence_rationale": "reasoned but no data cited",
    "effort_months": 1.5,
    "effort_rationale": "a sprint for a pair",
    "assumptions": ["assumed sellers are the addressable base"],
}


def _response(*, value=None, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(value=value, text=text)


def test_a_dict_value_is_used_directly():
    estimate = _coerce_estimate(_response(value={"estimates": [_ENTRY]}))

    assert [item.id for item in estimate.estimates] == ["F1"]


def test_an_already_validated_value_is_accepted():
    parsed = BacklogEstimate.model_validate({"estimates": [_ENTRY]})

    estimate = _coerce_estimate(_response(value=parsed))

    assert estimate.estimates[0].reach == 1200


def test_bare_json_text_is_parsed():
    estimate = _coerce_estimate(_response(text=json.dumps({"estimates": [_ENTRY]})))

    assert estimate.estimates[0].impact == 2


def test_a_fenced_object_with_prose_around_it_is_recovered():
    text = (
        "Sure! Here are the estimates:\n```json\n"
        + json.dumps({"estimates": [_ENTRY]})
        + "\n```\nHope that helps."
    )

    estimate = _coerce_estimate(_response(text=text))

    assert estimate.estimates[0].id == "F1"


def test_one_bad_row_does_not_discard_the_good_ones():
    broken = {**_ENTRY, "id": "F2", "impact": "high"}

    estimate = _coerce_estimate(_response(value={"estimates": [_ENTRY, broken]}))

    assert [item.id for item in estimate.estimates] == ["F1"]


def test_a_reply_with_no_valid_row_is_an_error():
    with pytest.raises(EstimateParseError, match="none valid"):
        _coerce_estimate(_response(value={"estimates": [{"id": "F1"}]}))


def test_a_reply_with_no_estimates_key_is_an_error():
    with pytest.raises(EstimateParseError, match="no 'estimates' list"):
        _coerce_estimate(_response(value={"features": []}))


def test_a_reply_with_no_json_at_all_is_an_error():
    with pytest.raises(EstimateParseError, match="was not JSON"):
        _coerce_estimate(_response(text="I cannot help with that."))


def test_malformed_json_is_an_error_not_a_crash():
    with pytest.raises(EstimateParseError):
        _payload_from(_response(text="{ estimates: [ }"))


def test_factors_are_snapped_on_the_way_in():
    loose = {**_ENTRY, "impact": 2.4, "confidence": 75, "effort_months": 2.3, "reach": 1200.7}

    estimate = _coerce_estimate(_response(value={"estimates": [loose]}))

    item = estimate.estimates[0]
    assert (item.impact, item.confidence, item.effort_months, item.reach) == (2.0, 0.8, 2.0, 1201.0)


def test_model_written_prose_is_sanitised():
    nasty = {**_ENTRY, "reach_rationale": "see ![x](http://evil/p.png)"}

    estimate = _coerce_estimate(_response(value={"estimates": [nasty]}))

    assert "![" not in estimate.estimates[0].reach_rationale


def test_the_estimate_schema_has_nowhere_to_put_a_score():
    """The structural guarantee: a model cannot hand back a RICE or ICE number."""
    fields = set(BacklogEstimate.model_json_schema()["$defs"]["FeatureEstimate"]["properties"])

    assert not fields & {"rice", "ice", "score", "rank", "priority"}


def test_the_reach_unit_is_read_from_the_reply():
    estimate = _coerce_estimate(_response(value={"reach_unit": "accounts", "estimates": [_ENTRY]}))

    assert estimate.reach_unit == "accounts"


def test_a_missing_reach_unit_falls_back_rather_than_blanking_the_header():
    for payload in ({"estimates": [_ENTRY]}, {"reach_unit": "   ", "estimates": [_ENTRY]}):
        assert _coerce_estimate(_response(value=payload)).reach_unit == "users"
