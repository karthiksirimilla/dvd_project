"""Stage 3 - Analysis.

Turns the fact table into the aggregate tables that feed the report and the
dashboard. Nothing here plots; nothing downstream re-derives a number.

The central construct is Satisfaction-at-Risk (SAR), defined in
`satisfaction_at_risk` below.

Run:  python -m src.analysis
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as C


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_facts() -> pd.DataFrame:
    return pd.read_parquet(C.FACTS_PATH)


def analysis_base(df: pd.DataFrame) -> pd.DataFrame:
    """Delivered orders that carry a review.

    This is the population every satisfaction statistic is computed on. It is
    defined once, here, so that no two figures in the report can disagree
    about their denominator.
    """
    base = df[df.is_delivered & df.review_score.notna()].copy()
    base = base[
        (base.purchased_at >= C.ANALYSIS_START) & (base.purchased_at <= C.ANALYSIS_END)
    ]
    return base


# --------------------------------------------------------------------------
# The headline metric
# --------------------------------------------------------------------------
def satisfaction_at_risk(
    g: pd.DataFrame, baseline_bad_rate: float
) -> pd.DataFrame:
    """Attach Satisfaction-at-Risk columns to a grouped summary.

    SAR answers one question: how many one- and two-star reviews would
    disappear if this segment merely behaved like the marketplace average?

        excess_detractors = orders * (bad_rate - baseline_bad_rate)

    A positive value is a segment destroying more goodwill than its size
    warrants. A negative value is a segment doing better than average.

    `gmv_exposure` prices that risk. With a repeat-purchase rate near 3%, a
    detractor is effectively a lost customer, so we value each excess
    detractor at the segment's average order value. It is a floor, not a
    forecast: it ignores word of mouth and the orders never placed.

    Expects `g` to already contain `orders`, `bad_rate` and `gmv`.
    """
    g = g.copy()
    g["excess_detractors"] = g["orders"] * (g["bad_rate"] - baseline_bad_rate)
    aov = np.where(g["orders"] > 0, g["gmv"] / g["orders"], 0.0)
    g["gmv_exposure"] = g["excess_detractors"].clip(lower=0) * aov
    total = g["gmv_exposure"].sum()
    g["sar_share"] = g["gmv_exposure"] / total if total else 0.0
    return g


def _summarise(base: pd.DataFrame, key) -> pd.DataFrame:
    """Standard segment summary. Every league table in the project uses this."""
    g = base.groupby(key, observed=True).agg(
        orders=("order_id", "size"),
        avg_score=("review_score", "mean"),
        bad_rate=("is_bad_review", "mean"),
        late_rate=("is_late", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_handling_days=("handling_days", "mean"),
        avg_transit_days=("transit_days", "mean"),
        avg_delay_days=("delay_days", "mean"),
        gmv=("item_total", "sum"),
        avg_order_value=("order_value", "mean"),
        avg_freight_ratio=("freight_ratio", "mean"),
    )
    return g.reset_index()


# --------------------------------------------------------------------------
# Aggregate builders
# --------------------------------------------------------------------------
def headline_kpis(base: pd.DataFrame, df: pd.DataFrame) -> dict:
    """The dozen numbers the presentation opens with."""
    on_time = base[~base.is_late.astype(bool)]
    late = base[base.is_late.astype(bool)]

    # Bad reviews attributable to lateness: the excess detractor rate among
    # late orders, applied to the number of late orders.
    lift = late.is_bad_review.mean() - on_time.is_bad_review.mean()
    attributable = lift * len(late)

    repeat = df.groupby("customer_unique_id").size()

    single = base[~base.is_multi_seller]
    multi = base[base.is_multi_seller]

    return {
        "orders_total": int(len(df)),
        "orders_analysed": int(len(base)),
        "gmv": float(base.item_total.sum()),
        "avg_score": float(base.review_score.mean()),
        "bad_rate": float(base.is_bad_review.mean()),
        "late_rate": float(base.is_late.mean()),
        "avg_delivery_days": float(base.delivery_days.mean()),
        "median_delivery_days": float(base.delivery_days.median()),
        "median_promised_days": float(base.promised_days.median()),
        "median_padding_days": float(base.promise_padding_days.median()),
        "avg_handling_days": float(base.handling_days.mean()),
        "avg_transit_days": float(base.transit_days.mean()),
        "score_on_time": float(on_time.review_score.mean()),
        "score_late": float(late.review_score.mean()),
        "bad_rate_on_time": float(on_time.is_bad_review.mean()),
        "bad_rate_late": float(late.is_bad_review.mean()),
        "detractors_total": int(base.is_bad_review.sum()),
        "detractors_from_lateness": float(attributable),
        "detractor_share_from_lateness": float(attributable / base.is_bad_review.sum()),
        "repeat_customer_rate": float((repeat > 1).mean()),
        "n_sellers": int(base.main_seller_id.nunique()),
        "score_single_seller": float(single.review_score.mean()),
        "score_multi_seller": float(multi.review_score.mean()),
        "multi_seller_share": float(base.is_multi_seller.mean()),
        "cross_state_share": float(base.is_cross_state.mean()),
        "corr_score_handling": float(base.review_score.corr(base.handling_days)),
        "corr_score_transit": float(base.review_score.corr(base.transit_days)),
        "corr_score_delivery": float(base.review_score.corr(base.delivery_days)),
        "corr_score_distance": float(base.review_score.corr(base.distance_km)),
    }


def monthly_trend(base: pd.DataFrame) -> pd.DataFrame:
    g = base.groupby("purchase_month").agg(
        orders=("order_id", "size"),
        gmv=("item_total", "sum"),
        avg_score=("review_score", "mean"),
        bad_rate=("is_bad_review", "mean"),
        late_rate=("is_late", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        active_sellers=("main_seller_id", "nunique"),
    ).reset_index()
    g["gmv_per_seller"] = g.gmv / g.active_sellers
    return g


def delay_curve(base: pd.DataFrame) -> pd.DataFrame:
    """The damage curve: review score against how late the order was."""
    g = base.groupby("delay_bucket", observed=True).agg(
        orders=("order_id", "size"),
        avg_score=("review_score", "mean"),
        bad_rate=("is_bad_review", "mean"),
    ).reset_index()
    g["delay_bucket"] = g.delay_bucket.astype(str)
    return g


def journey_stages(base: pd.DataFrame) -> pd.DataFrame:
    """Where the days go, split by whether the order ended up late.

    This is the chart that reframes the problem from "slow sellers" to
    "slow carriers".
    """
    rows = []
    for label, sub in [("On time", base[~base.is_late.astype(bool)]),
                       ("Late", base[base.is_late.astype(bool)])]:
        for stage, col in [("Payment approval", "approval_days"),
                           ("Seller handling", "handling_days"),
                           ("Carrier transit", "transit_days")]:
            rows.append({"group": label, "stage": stage,
                         "days": sub[col].mean(),
                         "median_days": sub[col].median()})
    return pd.DataFrame(rows)


def category_view(base: pd.DataFrame, baseline: float) -> pd.DataFrame:
    g = _summarise(base, "main_category")
    g = g[g.orders >= C.SEGMENT_MIN_ORDERS]
    return satisfaction_at_risk(g, baseline).sort_values(
        "gmv_exposure", ascending=False
    )


def state_view(base: pd.DataFrame, baseline: float) -> pd.DataFrame:
    g = _summarise(base, "customer_state")
    return satisfaction_at_risk(g, baseline).sort_values(
        "gmv_exposure", ascending=False
    )


def lane_view(base: pd.DataFrame, baseline: float) -> pd.DataFrame:
    """Origin-destination performance: which shipping lanes are broken."""
    sub = base[base.seller_state.notna()]
    g = sub.groupby(["seller_state", "customer_state"], observed=True).agg(
        orders=("order_id", "size"),
        avg_score=("review_score", "mean"),
        bad_rate=("is_bad_review", "mean"),
        late_rate=("is_late", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_transit_days=("transit_days", "mean"),
        avg_distance_km=("distance_km", "mean"),
        gmv=("item_total", "sum"),
    ).reset_index()
    g = g[g.orders >= 100]
    g["lane"] = g.seller_state + " to " + g.customer_state
    return satisfaction_at_risk(g, baseline).sort_values(
        "gmv_exposure", ascending=False
    )


def seller_scorecard(base: pd.DataFrame, baseline: float) -> pd.DataFrame:
    """Per-seller performance with an Invest / Coach / Monitor / Remove call.

    The quadrant is the operational output of the whole project: it says what
    to do with each seller rather than only how they scored.
    """
    g = base.groupby("main_seller_id", observed=True).agg(
        orders=("order_id", "size"),
        avg_score=("review_score", "mean"),
        bad_rate=("is_bad_review", "mean"),
        late_rate=("is_late", "mean"),
        avg_handling_days=("handling_days", "mean"),
        avg_transit_days=("transit_days", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        gmv=("item_total", "sum"),
        avg_order_value=("order_value", "mean"),
        avg_freight_ratio=("freight_ratio", "mean"),
        state=("seller_state", "first"),
        top_category=("main_category",
                      lambda s: s.value_counts().idxmax() if len(s) else None),
        first_order=("purchased_at", "min"),
        last_order=("purchased_at", "max"),
    ).reset_index().rename(columns={"main_seller_id": "seller_id"})

    g["tenure_days"] = (g.last_order - g.first_order).dt.days
    g["orders_per_month"] = g.orders / ((g.tenure_days / 30.44).clip(lower=1))
    g["gmv_share"] = g.gmv / g.gmv.sum()
    g["rated"] = g.orders >= C.SELLER_MIN_ORDERS

    g = satisfaction_at_risk(g, baseline)

    def verdict(r):
        if not r.rated:
            return "Monitor (low volume)"
        quality_bad = r.avg_score < C.SELLER_BAD_SCORE
        delivery_bad = r.late_rate > C.SELLER_LATE_RATE
        if quality_bad and delivery_bad:
            return "Remove"
        if quality_bad:
            return "Coach: product quality"
        if delivery_bad:
            return "Coach: fulfilment"
        return "Invest"

    g["action"] = g.apply(verdict, axis=1)
    return g.sort_values("gmv", ascending=False)


def basket_effects(base: pd.DataFrame) -> pd.DataFrame:
    """Small structural effects that turn out to matter: split shipments,
    distance, freight burden, payment method, installments."""
    frames = []

    def block(name, series):
        g = base.groupby(series, observed=True).agg(
            orders=("order_id", "size"),
            avg_score=("review_score", "mean"),
            bad_rate=("is_bad_review", "mean"),
            late_rate=("is_late", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
        ).reset_index()
        g.columns = ["level"] + list(g.columns[1:])
        g.insert(0, "dimension", name)
        g["level"] = g["level"].astype(str)
        frames.append(g)

    block("Sellers in order", base.n_sellers.clip(upper=3))
    block("Items in order", base.n_items.clip(upper=5))
    block("Distance band", base.distance_band)
    block("Freight ratio quintile",
          pd.qcut(base.freight_ratio.clip(0, 3), 5, duplicates="drop").astype(str))
    block("Payment type", base.payment_type)
    block("Installments", base.installments.clip(upper=7))
    block("Cross-state shipment",
          base.is_cross_state.map({1.0: "Different state", 0.0: "Same state"}))
    return pd.concat(frames, ignore_index=True)


def promise_analysis(base: pd.DataFrame) -> pd.DataFrame:
    """How much slack is built into the delivery promise, by state.

    `safe_trim_days` is the padding that could be removed while still hitting
    the current on-time rate for that state: the gap between the promise and
    the 92nd percentile of actual delivery time.
    """
    g = base.groupby("customer_state").agg(
        orders=("order_id", "size"),
        median_promised=("promised_days", "median"),
        median_actual=("delivery_days", "median"),
        p92_actual=("delivery_days", lambda s: s.quantile(0.92)),
        late_rate=("is_late", "mean"),
        avg_score=("review_score", "mean"),
    ).reset_index()
    g["padding_days"] = g.median_promised - g.median_actual
    g["safe_trim_days"] = (g.median_promised - g.p92_actual).clip(lower=0)
    return g.sort_values("safe_trim_days", ascending=False)


def marketing_funnel(con_df: dict) -> pd.DataFrame:
    """Acquisition quality by lead origin and business segment.

    Only 45% of closed deals ever transact, so this is scoped to matched
    sellers and reported with that caveat attached.
    """
    mql = con_df["mql"]
    deals = con_df["closed_deals"]
    sellers = con_df["seller_perf"]

    merged = deals.merge(mql, on="mql_id", how="left").merge(
        sellers, on="seller_id", how="left"
    )
    merged["transacted"] = merged.orders.notna()

    by_origin = merged.groupby("origin", dropna=False).agg(
        leads_won=("mql_id", "size"),
        activation_rate=("transacted", "mean"),
        avg_orders=("orders", "mean"),
        avg_score=("avg_score", "mean"),
        avg_late_rate=("late_rate", "mean"),
        gmv=("gmv", "sum"),
    ).reset_index().rename(columns={"origin": "segment"})
    by_origin.insert(0, "dimension", "Lead origin")

    by_seg = merged.groupby("business_segment", dropna=False).agg(
        leads_won=("mql_id", "size"),
        activation_rate=("transacted", "mean"),
        avg_orders=("orders", "mean"),
        avg_score=("avg_score", "mean"),
        avg_late_rate=("late_rate", "mean"),
        gmv=("gmv", "sum"),
    ).reset_index().rename(columns={"business_segment": "segment"})
    by_seg.insert(0, "dimension", "Business segment")

    out = pd.concat([by_origin, by_seg], ignore_index=True)
    return out.sort_values(["dimension", "gmv"], ascending=[True, False])


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_all() -> dict:
    df = load_facts()
    base = analysis_base(df)
    baseline = float(base.is_bad_review.mean())

    kpis = headline_kpis(base, df)
    kpis["baseline_bad_rate"] = baseline

    tables = {
        "monthly": monthly_trend(base),
        "delay_curve": delay_curve(base),
        "journey_stages": journey_stages(base),
        "category": category_view(base, baseline),
        "state": state_view(base, baseline),
        "lane": lane_view(base, baseline),
        "seller_scorecard": seller_scorecard(base, baseline),
        "basket_effects": basket_effects(base),
        "promise": promise_analysis(base),
    }

    # Marketing funnel needs the seller performance table
    import duckdb

    con = duckdb.connect(str(C.DB_PATH))
    mkt = marketing_funnel({
        "mql": con.execute("SELECT * FROM raw_mql").df(),
        "closed_deals": con.execute("SELECT * FROM raw_closed_deals").df(),
        "seller_perf": tables["seller_scorecard"][
            ["seller_id", "orders", "avg_score", "late_rate", "gmv"]
        ],
    })
    con.close()
    tables["marketing"] = mkt

    for name, t in tables.items():
        t.to_csv(C.TABLE_DIR / f"{name}.csv", index=False)
    with open(C.TABLE_DIR / "kpis.json", "w") as fh:
        json.dump(kpis, fh, indent=2)

    return {"kpis": kpis, **tables}


def main() -> None:
    out = build_all()
    k = out["kpis"]
    print("Headline KPIs")
    for key in ("orders_analysed", "avg_score", "bad_rate", "late_rate",
                "score_on_time", "score_late", "detractor_share_from_lateness",
                "avg_handling_days", "avg_transit_days", "median_padding_days",
                "repeat_customer_rate"):
        v = k[key]
        print(f"  {key:32s} {v:,.4f}" if isinstance(v, float) else f"  {key:32s} {v:,}")
    sc = out["seller_scorecard"]
    print("\nSeller actions:")
    print(sc.action.value_counts().to_string())
    print("\nGMV share by action:")
    print((sc.groupby("action").gmv.sum() / sc.gmv.sum()).round(4).to_string())


if __name__ == "__main__":
    main()
