"""Stage 5 - Scenario simulation.

Three levers leadership can actually pull, applied to the real order history:

  1. Speed up the carrier network (cut transit time on the worst lanes).
  2. Tighten the delivery promise (remove padding from the quoted ETA).
  3. Remove underperforming sellers from the catalogue.

Method
------
Rather than assume an effect, we learn the empirical relationship between how
late an order was and how likely it was to draw a one/two-star review, then
re-run history with the lever applied and read the new detractor count off
that curve.

    P(bad review) = f(delay_days)

`f` is estimated by binning delay into fine buckets and taking the observed
detractor rate, with monotone smoothing so the curve cannot wiggle upward as
orders get earlier.

Assumptions, stated plainly because they bound how far the numbers travel:
  * The delay-to-detractor relationship is treated as causal. Lateness is
    plausibly causal here, but part of it may proxy for other problems.
  * Removing a seller removes their orders outright. In reality some demand
    would be recaptured by other sellers, so the GMV loss shown is a
    worst case.
  * Tightening the promise changes only the lateness flag, not customer
    behaviour. In practice a shorter quoted ETA may also lift conversion,
    which this model does not credit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config as C


# --------------------------------------------------------------------------
# The damage curve
# --------------------------------------------------------------------------
def fit_delay_curve(base: pd.DataFrame, n_bins: int = 40) -> pd.DataFrame:
    """Estimate P(bad review | delay_days) from the observed data."""
    d = base[["delay_days", "is_bad_review"]].dropna().copy()
    # Clip the tails: beyond these points the sample is thin and the rate is
    # already saturated.
    d["clipped"] = d.delay_days.clip(-40, 30)
    d["bin"] = pd.qcut(d.clipped, n_bins, duplicates="drop")

    curve = d.groupby("bin", observed=True).agg(
        centre=("clipped", "mean"),
        rate=("is_bad_review", "mean"),
        n=("is_bad_review", "size"),
    ).reset_index(drop=True).sort_values("centre")

    # Enforce monotonicity: being earlier can never raise detractor risk.
    curve["rate"] = np.maximum.accumulate(np.asarray(curve.rate.values, dtype="float64"))
    curve["centre"] = np.asarray(curve.centre.values, dtype="float64")
    return curve.reset_index(drop=True)


def predict_bad_rate(curve: pd.DataFrame, delay: np.ndarray) -> np.ndarray:
    """Look up detractor probability for arbitrary delay values."""
    delay = np.asarray(delay, dtype="float64")
    xp = np.asarray(curve.centre.values, dtype="float64")
    fp = np.asarray(curve.rate.values, dtype="float64")
    return np.interp(np.clip(delay, -40, 30), xp, fp, left=fp[0], right=fp[-1])


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    """Lever settings for one simulation run."""

    transit_cut_pct: float = 0.0        # % reduction in carrier transit time
    transit_target: str = "worst_lanes"  # 'worst_lanes' | 'all' | 'cross_state'
    lane_late_threshold: float = 0.10    # which lanes count as 'worst'
    promise_trim_days: float = 0.0       # days removed from the quoted ETA
    remove_below_score: float | None = None   # drop sellers under this score
    remove_min_orders: int = C.SELLER_MIN_ORDERS
    notes: str = field(default="")


def _targeted_mask(base: pd.DataFrame, scenario: Scenario) -> pd.Series:
    """Which orders the logistics investment reaches."""
    if scenario.transit_target == "all":
        return pd.Series(True, index=base.index)
    if scenario.transit_target == "cross_state":
        return base.is_cross_state.fillna(0).astype(bool)

    # 'worst_lanes': lanes whose late rate exceeds the threshold
    lane = base.seller_state.fillna("?") + ">" + base.customer_state.fillna("?")
    lane_rate = base.assign(lane=lane).groupby("lane").is_late.mean()
    bad_lanes = set(lane_rate[lane_rate > scenario.lane_late_threshold].index)
    return lane.isin(bad_lanes)


def run(base: pd.DataFrame, scenario: Scenario, curve: pd.DataFrame) -> dict:
    """Apply a scenario to the order history and report the delta.

    Returns a dict of before/after metrics plus the deltas that the dashboard
    displays.
    """
    d = base.copy()

    baseline = {
        "orders": len(d),
        "gmv": float(d.item_total.sum()),
        "detractors": float(predict_bad_rate(curve, d.delay_days.values).sum()),
        "late_rate": float(d.is_late.mean()),
        "avg_delivery_days": float(d.delivery_days.mean()),
    }
    baseline["bad_rate"] = baseline["detractors"] / baseline["orders"]

    # ---- Lever 1: faster carrier on targeted lanes ----
    saved = np.zeros(len(d))
    if scenario.transit_cut_pct > 0:
        mask = _targeted_mask(d, scenario).values
        transit = d.transit_days.fillna(0).clip(lower=0).values
        saved = np.where(mask, transit * scenario.transit_cut_pct / 100.0, 0.0)

    new_delay = d.delay_days.values - saved
    new_delivery = d.delivery_days.values - saved

    # ---- Lever 2: tighter promise ----
    # Trimming the ETA moves the deadline earlier, so delay rises.
    new_delay = new_delay + scenario.promise_trim_days

    d["sim_delay"] = new_delay
    d["sim_delivery"] = new_delivery
    d["sim_late"] = new_delay > 0
    d["sim_bad_p"] = predict_bad_rate(curve, new_delay)

    # ---- Lever 3: remove sellers ----
    removed_gmv = 0.0
    removed_orders = 0
    removed_sellers = 0
    if scenario.remove_below_score is not None:
        perf = d.groupby("main_seller_id").agg(
            orders=("order_id", "size"), score=("review_score", "mean")
        )
        drop = perf[(perf.orders >= scenario.remove_min_orders)
                    & (perf.score < scenario.remove_below_score)].index
        removed_sellers = len(drop)
        gone = d.main_seller_id.isin(drop)
        removed_gmv = float(d.loc[gone, "item_total"].sum())
        removed_orders = int(gone.sum())
        d = d[~gone]

    after = {
        "orders": len(d),
        "gmv": float(d.item_total.sum()),
        "detractors": float(d.sim_bad_p.sum()),
        "late_rate": float(d.sim_late.mean()),
        "avg_delivery_days": float(d.sim_delivery.mean()),
    }
    after["bad_rate"] = after["detractors"] / max(after["orders"], 1)

    # Value of the goodwill saved, on the same basis as SAR: a detractor in a
    # market with ~3% repeat purchase is a lost customer worth one AOV.
    aov = baseline["gmv"] / baseline["orders"]
    detractors_avoided = baseline["detractors"] - after["detractors"]

    return {
        "baseline": baseline,
        "after": after,
        "detractors_avoided": detractors_avoided,
        "detractors_avoided_pct": detractors_avoided / baseline["detractors"],
        "gmv_lost": removed_gmv,
        "gmv_lost_pct": removed_gmv / baseline["gmv"],
        "orders_lost": removed_orders,
        "sellers_removed": removed_sellers,
        "goodwill_value": detractors_avoided * aov,
        "net_value": detractors_avoided * aov - removed_gmv,
        "late_rate_delta": after["late_rate"] - baseline["late_rate"],
        "score_proxy_delta": -(after["bad_rate"] - baseline["bad_rate"]),
    }


def preset_scenarios() -> dict[str, Scenario]:
    """Named scenarios used in the report and offered in the dashboard."""
    return {
        "Do nothing": Scenario(notes="Current state, for reference"),
        "Fix the worst lanes": Scenario(
            transit_cut_pct=25, transit_target="worst_lanes",
            notes="25% faster transit on lanes with a late rate above 10%"),
        "Remove the worst sellers": Scenario(
            remove_below_score=3.5,
            notes="Delist sellers with 30+ orders averaging under 3.5 stars"),
        "Tighten the promise by 3 days": Scenario(
            promise_trim_days=3,
            notes="Quote a 3-day shorter ETA with no operational change"),
        "Combined programme": Scenario(
            transit_cut_pct=25, transit_target="worst_lanes",
            remove_below_score=3.5, promise_trim_days=2,
            notes="Logistics investment, seller cleanup, and a modest "
                  "promise tightening funded by the faster network"),
    }
