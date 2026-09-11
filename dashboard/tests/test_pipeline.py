"""Test suite.

These are not decorative. Each test pins down a claim the report makes, so
that a change to the pipeline that would quietly invalidate a headline number
fails here instead of in a presentation.

Run:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import analysis as A
from src import config as C
from src import simulate as S
from src.features import haversine_km


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def facts() -> pd.DataFrame:
    if not C.FACTS_PATH.exists():
        pytest.skip("Run `make all` first to build the fact table")
    return A.load_facts()


@pytest.fixture(scope="module")
def base(facts) -> pd.DataFrame:
    return A.analysis_base(facts)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
def test_grain_is_one_row_per_order(facts):
    assert facts.order_id.is_unique


def test_no_negative_durations(base):
    assert (base.delivery_days >= 0).all()
    assert (base.transit_days.dropna() >= -1).all()


def test_lateness_is_never_silently_false(facts):
    """An order with no delivery date must have an unknown lateness flag.

    This is the bug that would most easily flatter the on-time rate, so it
    gets its own test.
    """
    missing = facts[facts.delivered_at.isna()]
    assert missing.is_late.notna().sum() == 0


def test_analysis_base_is_delivered_and_reviewed(base):
    assert base.is_delivered.all()
    assert base.review_score.notna().all()


def test_continuous_columns_are_numeric(base):
    """Object-dtype numerics silently break aggregation and plotting."""
    for col in ["delivery_days", "delay_days", "is_late", "is_bad_review",
                "item_total", "review_score"]:
        assert pd.api.types.is_numeric_dtype(base[col]), f"{col} is not numeric"


# --------------------------------------------------------------------------
# Metric logic
# --------------------------------------------------------------------------
def test_bad_review_flag_matches_definition(base):
    expected = base.review_score <= C.BAD_REVIEW_MAX
    assert (base.is_bad_review.astype(bool) == expected).all()


def test_sar_is_zero_at_the_baseline():
    """A segment performing exactly at the baseline carries no excess risk."""
    g = pd.DataFrame({"orders": [1000], "bad_rate": [0.13], "gmv": [50_000.0]})
    out = A.satisfaction_at_risk(g, baseline_bad_rate=0.13)
    assert out.excess_detractors.iloc[0] == pytest.approx(0.0)
    assert out.gmv_exposure.iloc[0] == pytest.approx(0.0)


def test_sar_scales_with_volume_and_gap():
    g = pd.DataFrame({"orders": [1000, 2000], "bad_rate": [0.20, 0.20],
                      "gmv": [100_000.0, 200_000.0]})
    out = A.satisfaction_at_risk(g, baseline_bad_rate=0.10)
    # Twice the orders at the same rate means twice the excess detractors.
    assert out.excess_detractors.iloc[1] == pytest.approx(
        2 * out.excess_detractors.iloc[0])


def test_better_than_baseline_segments_carry_no_exposure():
    g = pd.DataFrame({"orders": [500], "bad_rate": [0.05], "gmv": [20_000.0]})
    out = A.satisfaction_at_risk(g, baseline_bad_rate=0.13)
    assert out.excess_detractors.iloc[0] < 0
    assert out.gmv_exposure.iloc[0] == 0.0  # clipped: no negative exposure


def test_seller_scorecard_assigns_every_seller_an_action(base):
    sc = A.seller_scorecard(base, float(base.is_bad_review.mean()))
    assert sc.action.notna().all()
    assert sc.seller_id.is_unique


def test_low_volume_sellers_are_not_judged_harshly(base):
    sc = A.seller_scorecard(base, float(base.is_bad_review.mean()))
    small = sc[sc.orders < C.SELLER_MIN_ORDERS]
    assert (small.action == "Monitor (low volume)").all()


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def test_damage_curve_is_monotone(base):
    """Being earlier can never increase the chance of a bad review."""
    curve = S.fit_delay_curve(base)
    assert (np.diff(curve.rate.values) >= -1e-12).all()


def test_do_nothing_scenario_changes_nothing(base):
    curve = S.fit_delay_curve(base)
    r = S.run(base, S.Scenario(), curve)
    assert r["detractors_avoided"] == pytest.approx(0.0, abs=1e-6)
    assert r["gmv_lost"] == 0.0
    assert r["sellers_removed"] == 0


def test_faster_transit_reduces_detractors(base):
    curve = S.fit_delay_curve(base)
    r = S.run(base, S.Scenario(transit_cut_pct=30, transit_target="all"), curve)
    assert r["detractors_avoided"] > 0
    assert r["after"]["late_rate"] < r["baseline"]["late_rate"]


def test_tightening_the_promise_increases_lateness(base):
    """Counterintuitive but important: the ETA padding is load-bearing."""
    curve = S.fit_delay_curve(base)
    r = S.run(base, S.Scenario(promise_trim_days=3), curve)
    assert r["after"]["late_rate"] > r["baseline"]["late_rate"]
    assert r["detractors_avoided"] < 0


def test_removing_sellers_costs_gmv(base):
    curve = S.fit_delay_curve(base)
    r = S.run(base, S.Scenario(remove_below_score=3.5), curve)
    assert r["sellers_removed"] > 0
    assert r["gmv_lost"] > 0
    assert r["after"]["orders"] < r["baseline"]["orders"]


def test_stronger_lever_does_more(base):
    curve = S.fit_delay_curve(base)
    weak = S.run(base, S.Scenario(transit_cut_pct=10, transit_target="all"), curve)
    strong = S.run(base, S.Scenario(transit_cut_pct=40, transit_target="all"), curve)
    assert strong["detractors_avoided"] > weak["detractors_avoided"]


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def test_haversine_known_distance():
    # Sao Paulo to Rio de Janeiro is roughly 360 km.
    d = haversine_km(np.array([-23.55]), np.array([-46.63]),
                     np.array([-22.91]), np.array([-43.17]))
    assert 330 < d[0] < 380


def test_haversine_zero_for_same_point():
    d = haversine_km(np.array([-10.0]), np.array([-50.0]),
                     np.array([-10.0]), np.array([-50.0]))
    assert d[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Headline claims made in the report
# --------------------------------------------------------------------------
def test_late_orders_score_far_worse(base):
    on_time = base[base.is_late == 0].review_score.mean()
    late = base[base.is_late == 1].review_score.mean()
    assert on_time - late > 1.0, "the central finding of the report"


def test_carrier_owns_more_of_the_journey_than_the_seller(base):
    assert base.transit_days.mean() > 2 * base.handling_days.mean()


def test_split_orders_score_worse(base):
    single = base[~base.is_multi_seller].review_score.mean()
    multi = base[base.is_multi_seller].review_score.mean()
    assert single - multi > 0.5
