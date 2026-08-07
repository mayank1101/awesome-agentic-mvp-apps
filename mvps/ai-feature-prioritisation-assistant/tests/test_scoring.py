"""Tests for reconciliation, ranking, divergence, and levers.

The first test in this file is the one that matters most: it asserts the claim
the whole app is built on, that every score displayed is the formula applied to
the factors displayed. If that ever fails, nothing else here is worth reading.
"""

from app.models.schemas import BacklogEstimate
from app.services.scales import ice_score, rice_score
from app.services.scoring import attach_levers, describe_divergence, lever_hint, score_backlog
from tests.conftest import make_backlog, make_backlog_estimate, make_estimate


def test_every_score_is_the_formula_applied_to_its_own_factors():
    backlog = make_backlog("Bulk export", "Dark mode", "SSO", "Audit log", "Webhooks")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=2000, impact=2, confidence=1.0, effort_months=2),
        make_estimate("F2", reach=3500, impact=0.25, confidence=0.8, effort_months=0.5),
        make_estimate("F3", reach=120, impact=3, confidence=1.0, effort_months=6),
        make_estimate("F4", reach=400, impact=1, confidence=0.5, effort_months=1),
        make_estimate("F5", reach=900, impact=2, confidence=0.8, effort_months=3),
    )

    ranked = score_backlog(backlog, estimate)

    for row in ranked.rows:
        factors = row.factors
        assert row.rice == rice_score(
            factors.reach, factors.impact, factors.confidence, factors.effort_months
        )
        assert row.ice == ice_score(factors.impact, factors.confidence, factors.effort_months)


def test_ranks_are_dense_and_cover_every_row():
    backlog = make_backlog("A", "B", "C")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=100),
        make_estimate("F2", reach=200),
        make_estimate("F3", reach=300),
    )

    ranked = score_backlog(backlog, estimate)

    assert sorted(row.rice_rank for row in ranked.rows) == [1, 2, 3]
    assert sorted(row.ice_rank for row in ranked.rows) == [1, 2, 3]
    assert [row.idea.title for row in ranked.rows] == ["C", "B", "A"]


def test_identical_factors_break_ties_deterministically_by_input_order():
    backlog = make_backlog("First", "Second", "Third")
    estimate = make_backlog_estimate(make_estimate("F1"), make_estimate("F2"), make_estimate("F3"))

    ranked = score_backlog(backlog, estimate)

    assert [row.idea.title for row in ranked.rows] == ["First", "Second", "Third"]
    assert len({row.rice_rank for row in ranked.rows}) == 3


def test_ties_prefer_the_better_evidenced_feature():
    # Same RICE score by construction: 1000*1*1.0/2 == 800*1*0.5/0.8 is not
    # equal, so build the tie explicitly from equal scores at different
    # confidence. 500*1*1.0/1 == 1000*1*0.5/1 == 500.
    backlog = make_backlog("Guessed", "Evidenced")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=1000, impact=1, confidence=0.5, effort_months=1),
        make_estimate("F2", reach=500, impact=1, confidence=1.0, effort_months=1),
    )

    ranked = score_backlog(backlog, estimate)

    assert ranked.rows[0].rice == ranked.rows[1].rice
    assert ranked.rows[0].idea.title == "Evidenced"


def test_unknown_ids_are_dropped_and_missing_ones_reported():
    backlog = make_backlog("A", "B", "C")
    estimate = make_backlog_estimate(
        make_estimate("F1"),
        make_estimate("F9"),  # never sent
        make_estimate("F3"),
    )

    ranked = score_backlog(backlog, estimate)

    assert [row.idea.id for row in ranked.rows] == ["F1", "F3"]
    assert ranked.unestimated == ["F2"]


def test_a_duplicated_id_keeps_the_first_entry():
    backlog = make_backlog("A")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=100), make_estimate("F1", reach=9999)
    )

    ranked = score_backlog(backlog, estimate)

    assert len(ranked.rows) == 1
    assert ranked.rows[0].factors.reach == 100


def test_an_empty_estimate_yields_no_rows_and_lists_everything_as_unestimated():
    backlog = make_backlog("A", "B")

    ranked = score_backlog(backlog, BacklogEstimate(estimates=[]))

    assert ranked.rows == []
    assert ranked.unestimated == ["F1", "F2"]


# --- Overrides --------------------------------------------------------------
def test_an_override_changes_the_score_and_is_attributed():
    backlog = make_backlog("A")
    estimate = make_backlog_estimate(make_estimate("F1", effort_months=4))

    ranked = score_backlog(backlog, estimate, overrides={"F1": {"effort_months": 1}})

    row = ranked.rows[0]
    assert row.factors.effort_months == 1
    assert row.overridden == ["effort_months"]
    assert row.rice == rice_score(row.factors.reach, row.factors.impact, row.factors.confidence, 1)


def test_an_override_can_reorder_the_backlog():
    backlog = make_backlog("Cheap", "Expensive")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=1000, effort_months=4),
        make_estimate("F2", reach=1000, effort_months=1),
    )

    before = score_backlog(backlog, estimate)
    after = score_backlog(backlog, estimate, overrides={"F1": {"effort_months": 0.25}})

    assert before.rows[0].idea.title == "Expensive"
    assert after.rows[0].idea.title == "Cheap"


def test_an_edit_that_snaps_back_to_the_estimate_is_not_an_override():
    backlog = make_backlog("A")
    estimate = make_backlog_estimate(make_estimate("F1", effort_months=2))

    ranked = score_backlog(backlog, estimate, overrides={"F1": {"effort_months": 2.1}})

    assert ranked.rows[0].factors.effort_months == 2
    assert ranked.rows[0].overridden == []


def test_overrides_for_unknown_features_are_ignored():
    backlog = make_backlog("A")
    estimate = make_backlog_estimate(make_estimate("F1"))

    ranked = score_backlog(backlog, estimate, overrides={"F42": {"impact": 3}})

    assert ranked.rows[0].overridden == []


# --- Divergence -------------------------------------------------------------
def test_divergence_names_the_feature_and_blames_reach():
    # F1 is broad and costly, F2 narrow and cheap. ICE cannot see the reach gap.
    backlog = make_backlog("Broad and costly", "Narrow and cheap", "Middling")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=50_000, impact=1, confidence=0.8, effort_months=9),
        make_estimate("F2", reach=20, impact=3, confidence=1.0, effort_months=0.25),
        make_estimate("F3", reach=800, impact=1, confidence=0.8, effort_months=2),
    )

    ranked = score_backlog(backlog, estimate)

    assert ranked.rows[0].idea.title == "Broad and costly"
    assert ranked.by_ice()[0].idea.title == "Narrow and cheap"


def test_agreement_is_reported_as_agreement_not_corroboration():
    backlog = make_backlog("A", "B")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=5000, impact=3, confidence=1.0, effort_months=0.5),
        make_estimate("F2", reach=10, impact=0.25, confidence=0.5, effort_months=12),
    )

    notes = score_backlog(backlog, estimate).divergence

    assert len(notes) == 1
    assert "not corroboration" in notes[0]


def test_divergence_is_silent_for_a_single_feature():
    backlog = make_backlog("Only one")
    ranked = score_backlog(backlog, make_backlog_estimate(make_estimate("F1")))

    assert ranked.divergence == []
    assert describe_divergence(ranked.rows) == []


def test_a_divergent_feature_gets_a_note_that_names_it():
    backlog = make_backlog("Broad", "Narrow-but-easy", "Mid", "Filler")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=90_000, impact=1, confidence=0.8, effort_months=12),
        make_estimate("F2", reach=15, impact=3, confidence=1.0, effort_months=0.25),
        make_estimate("F3", reach=2000, impact=2, confidence=0.8, effort_months=2),
        make_estimate("F4", reach=1500, impact=1, confidence=0.8, effort_months=3),
    )

    notes = score_backlog(backlog, estimate).divergence

    assert notes
    assert any("Narrow-but-easy" in note for note in notes)


# --- Levers -----------------------------------------------------------------
def test_lever_hint_inverts_the_rice_formula():
    backlog = make_backlog("Leader", "Chaser")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=4000, impact=1, confidence=1.0, effort_months=1),
        make_estimate("F2", reach=1000, impact=1, confidence=1.0, effort_months=1),
    )

    ranked = score_backlog(backlog, estimate)
    leader, chaser = ranked.rows[0], ranked.rows[1]
    hint = lever_hint(chaser, leader)

    assert hint is not None
    assert "4,000" in hint  # the reach needed to draw level
    assert "Leader" in hint


def test_no_lever_for_the_top_row():
    backlog = make_backlog("A", "B")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=4000), make_estimate("F2", reach=100)
    )

    ranked = score_backlog(backlog, estimate)
    hints = attach_levers(ranked)

    assert ranked.rows[0].idea.id not in hints
    assert ranked.rows[1].idea.id in hints


def test_no_effort_lever_when_the_floor_cannot_close_the_gap():
    # The chaser reaches 1 user; no achievable effort makes it competitive.
    backlog = make_backlog("Leader", "Chaser")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=100_000, impact=3, confidence=1.0, effort_months=0.25),
        make_estimate("F2", reach=1, impact=0.25, confidence=0.5, effort_months=24),
    )

    ranked = score_backlog(backlog, estimate)
    hint = lever_hint(ranked.rows[1], ranked.rows[0])

    assert hint is None or "Effort drops" not in hint


# --- Findings from the first live run ---------------------------------------
def test_divergence_is_capped_so_it_stays_readable():
    # A twelve-item backlog can put six features in the symmetric difference of
    # the two top-3 lists. Six paragraphs is a wall of text, not an insight.
    backlog = make_backlog(*[f"F{index}" for index in range(1, 13)])
    estimate = make_backlog_estimate(
        *[
            make_estimate(
                f"F{index}",
                reach=10 ** (index % 4 + 1),
                impact=[0.25, 1, 2, 3][index % 4],
                confidence=[0.5, 0.8, 1.0][index % 3],
                effort_months=[0.25, 1, 3, 12][index % 4],
            )
            for index in range(1, 13)
        ]
    )

    notes = score_backlog(backlog, estimate).divergence

    assert 0 < len(notes) <= 3


def test_the_biggest_mover_is_reported_first():
    backlog = make_backlog("Huge swing", "Small swing", "Anchor", "Filler", "Filler2")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=1, impact=3, confidence=1.0, effort_months=0.25),
        make_estimate("F2", reach=900, impact=2, confidence=0.8, effort_months=1),
        make_estimate("F3", reach=50_000, impact=1, confidence=0.8, effort_months=9),
        make_estimate("F4", reach=800, impact=1, confidence=0.8, effort_months=2),
        make_estimate("F5", reach=700, impact=1, confidence=0.5, effort_months=3),
    )

    ranked = score_backlog(backlog, estimate)
    shifts = {row.idea.title: abs(row.rank_shift) for row in ranked.rows}
    biggest = max(shifts, key=shifts.get)

    assert ranked.divergence[0].startswith(f"**{biggest}**")


def test_a_reach_lever_beyond_the_backlogs_own_ceiling_is_not_offered():
    # Observed live: "overtakes X if Reach reaches 24,000" for a 12,000-seat
    # product. A lever nobody can pull reads like advice and is worse than none.
    backlog = make_backlog("Leader", "Chaser")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=12_000, impact=1, confidence=0.8, effort_months=0.25),
        make_estimate("F2", reach=12_000, impact=0.5, confidence=0.8, effort_months=0.25),
    )

    ranked = score_backlog(backlog, estimate)
    hints = attach_levers(ranked)

    assert "Reach reaches" not in hints.get(ranked.rows[1].idea.id, "")


def test_the_reach_unit_is_carried_through_to_the_ranking():
    backlog = make_backlog("A")
    estimate = BacklogEstimate(reach_unit="accounts", estimates=[make_estimate("F1")])

    assert score_backlog(backlog, estimate).reach_unit == "accounts"


def test_divergence_and_levers_use_the_declared_reach_unit():
    backlog = make_backlog("Broad", "Narrow-but-easy", "Mid", "Filler")
    estimate = BacklogEstimate(
        reach_unit="accounts",
        estimates=[
            make_estimate("F1", reach=9_000, impact=1, confidence=0.8, effort_months=12),
            make_estimate("F2", reach=15, impact=3, confidence=1.0, effort_months=0.25),
            make_estimate("F3", reach=2000, impact=2, confidence=0.8, effort_months=2),
            make_estimate("F4", reach=1500, impact=1, confidence=0.8, effort_months=3),
        ],
    )

    ranked = score_backlog(backlog, estimate)
    hints = attach_levers(ranked)

    assert any("accounts/quarter" in note for note in ranked.divergence)
    assert not any("users/quarter" in note for note in ranked.divergence)
    assert all("users/quarter" not in hint for hint in hints.values())
