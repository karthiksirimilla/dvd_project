"""Stage 2 - Feature engineering.

Collapses the relational schema into one order-level fact table. Every later
stage (analysis, model, dashboard) reads only this table, which keeps the
joins in one place and makes the numbers reproducible.

Grain: one row per order_id.

Run:  python -m src.features
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src import config as C
from src.ingest import connect


# --------------------------------------------------------------------------
# SQL: item-level rollup
# --------------------------------------------------------------------------
ITEM_ROLLUP = """
CREATE OR REPLACE TABLE agg_items AS
WITH itm AS (
    SELECT
        i.order_id,
        i.order_item_id,
        i.product_id,
        i.seller_id,
        i.price,
        i.freight_value,
        i.shipping_limit_date,
        coalesce(t.product_category_name_english,
                 p.product_category_name,
                 'unknown')                       AS category,
        p.product_weight_g,
        p.product_photos_qty,
        p.product_description_lenght              AS description_length,
        s.seller_state,
        s.seller_zip_code_prefix,
        -- rank items by value so the "headline" product of the order is the
        -- most expensive one, not an arbitrary row
        row_number() OVER (PARTITION BY i.order_id
                           ORDER BY i.price DESC, i.order_item_id) AS value_rank
    FROM raw_order_items i
    LEFT JOIN raw_products p USING (product_id)
    LEFT JOIN raw_category_translation t
           ON p.product_category_name = t.product_category_name
    LEFT JOIN raw_sellers s USING (seller_id)
)
SELECT
    order_id,
    count(*)                                    AS n_items,
    count(DISTINCT seller_id)                   AS n_sellers,
    count(DISTINCT category)                    AS n_categories,
    sum(price)                                  AS item_total,
    sum(freight_value)                          AS freight_total,
    avg(product_weight_g)                       AS avg_weight_g,
    avg(product_photos_qty)                     AS avg_photos,
    avg(description_length)                     AS avg_description_length,
    max(shipping_limit_date)                    AS shipping_limit_date,
    any_value(category)      FILTER (WHERE value_rank = 1) AS main_category,
    any_value(seller_id)     FILTER (WHERE value_rank = 1) AS main_seller_id,
    any_value(seller_state)  FILTER (WHERE value_rank = 1) AS seller_state,
    any_value(seller_zip_code_prefix) FILTER (WHERE value_rank = 1)
                                                AS seller_zip
FROM itm
GROUP BY order_id
"""

PAYMENT_ROLLUP = """
CREATE OR REPLACE TABLE agg_payments AS
SELECT
    order_id,
    sum(payment_value)                          AS payment_value,
    max(payment_installments)                   AS installments,
    count(*)                                    AS n_payment_rows,
    -- the payment method carrying the largest value defines the order
    any_value(payment_type ORDER BY payment_value DESC) AS payment_type
FROM raw_order_payments
GROUP BY order_id
"""

# --------------------------------------------------------------------------
# SQL: the fact table
# --------------------------------------------------------------------------
FACTS = """
CREATE OR REPLACE TABLE fact_orders AS
SELECT
    o.order_id,
    o.order_status,
    c.customer_unique_id,
    c.customer_state,
    c.customer_city,
    c.customer_zip_code_prefix                  AS customer_zip,

    -- lifecycle timestamps
    o.order_purchase_timestamp                  AS purchased_at,
    o.order_approved_at                         AS approved_at,
    o.order_delivered_carrier_date              AS carrier_at,
    o.order_delivered_customer_date             AS delivered_at,
    o.order_estimated_delivery_date             AS promised_at,

    -- durations, in days
    date_diff('minute', o.order_purchase_timestamp, o.order_approved_at)
        / 1440.0                                AS approval_days,
    date_diff('minute', o.order_approved_at, o.order_delivered_carrier_date)
        / 1440.0                                AS handling_days,
    date_diff('minute', o.order_delivered_carrier_date,
              o.order_delivered_customer_date) / 1440.0
                                                AS transit_days,
    date_diff('minute', o.order_purchase_timestamp,
              o.order_delivered_customer_date) / 1440.0
                                                AS delivery_days,
    date_diff('minute', o.order_purchase_timestamp,
              o.order_estimated_delivery_date) / 1440.0
                                                AS promised_days,
    date_diff('minute', o.order_estimated_delivery_date,
              o.order_delivered_customer_date) / 1440.0
                                                AS delay_days,
    -- did the seller miss the contractual handover deadline?
    date_diff('minute', i.shipping_limit_date,
              o.order_delivered_carrier_date) / 1440.0
                                                AS handover_slack_days,

    -- basket
    i.n_items, i.n_sellers, i.n_categories,
    i.item_total, i.freight_total,
    i.item_total + i.freight_total              AS order_value,
    i.avg_weight_g, i.avg_photos, i.avg_description_length,
    i.main_category, i.main_seller_id, i.seller_state,

    -- payment
    p.payment_value, p.installments, p.payment_type,

    -- outcome
    r.review_score,
    r.review_creation_date,
    r.review_answer_timestamp,
    r.review_comment_message,
    date_diff('minute', r.review_creation_date, r.review_answer_timestamp)
        / 60.0                                  AS review_response_hours,

    -- geography
    gc.lat AS customer_lat, gc.lng AS customer_lng,
    gs.lat AS seller_lat,   gs.lng AS seller_lng
FROM raw_orders o
LEFT JOIN raw_customers c USING (customer_id)
LEFT JOIN agg_items     i USING (order_id)
LEFT JOIN agg_payments  p USING (order_id)
LEFT JOIN dim_reviews   r USING (order_id)
LEFT JOIN dim_geo      gc ON gc.zip_prefix = c.customer_zip_code_prefix
LEFT JOIN dim_geo      gs ON gs.zip_prefix = i.seller_zip
"""


# --------------------------------------------------------------------------
# Python-side derived columns
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, vectorised over numpy arrays."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add the analysis flags that are easier to express in pandas than SQL."""
    df = df.copy()

    # ---- Timestamp integrity -------------------------------------------
    # A small share of orders have lifecycle timestamps out of order: handed
    # to the carrier before payment approval, or delivered to the customer
    # before the carrier receipt was logged. These are recording faults, not
    # real events. The order itself is kept (its purchase-to-delivery time is
    # still valid) but the impossible stage durations are set to null so they
    # cannot drag the stage averages around.
    df["timestamp_inconsistent"] = (
        (df["handling_days"] < 0).fillna(False)
        | (df["transit_days"] < 0).fillna(False)
        | (df["approval_days"] < 0).fillna(False)
    )
    for col in ("approval_days", "handling_days", "transit_days"):
        df.loc[df[col] < 0, col] = np.nan

    # Calendar helpers
    df["purchase_date"] = df["purchased_at"].dt.date
    df["purchase_month"] = df["purchased_at"].dt.to_period("M").dt.to_timestamp()
    df["purchase_dow"] = df["purchased_at"].dt.day_name()
    df["purchase_hour"] = df["purchased_at"].dt.hour

    # Delivery outcome flags. Note the guard: an order without a delivery
    # timestamp has an unknown outcome and must stay NaN rather than False,
    # otherwise every undelivered order silently counts as "on time".
    delivered = df["delivered_at"].notna()
    df["is_delivered"] = df["order_status"].eq("delivered") & delivered
    # Cast to float rather than leaving object dtype: these columns are
    # averaged everywhere downstream, and an object-dtype mean silently
    # breaks plotting and arithmetic.
    df["is_late"] = np.where(
        delivered, (df["delay_days"] > 0).astype("float64"), np.nan
    ).astype("float64")
    df["is_very_late"] = np.where(
        delivered, (df["delay_days"] > 5).astype("float64"), np.nan
    ).astype("float64")

    df["delay_bucket"] = pd.cut(
        df["delay_days"], bins=C.DELAY_BUCKETS, labels=C.DELAY_LABELS
    )

    # Promise padding: how much slack the platform builds into the ETA.
    df["promise_padding_days"] = df["promised_days"] - df["delivery_days"]

    # Share of the total journey spent with the seller vs with the carrier.
    total = df["handling_days"] + df["transit_days"]
    df["handling_share"] = np.where(total > 0, df["handling_days"] / total, np.nan)

    # Commercial features
    df["freight_ratio"] = np.where(
        df["item_total"] > 0, df["freight_total"] / df["item_total"], np.nan
    )
    df["is_multi_seller"] = df["n_sellers"] > 1
    df["is_cross_state"] = np.where(
        df["seller_state"].notna() & df["customer_state"].notna(),
        (df["seller_state"] != df["customer_state"]).astype("float64"),
        np.nan,
    ).astype("float64")

    # Distance between seller and customer
    df["distance_km"] = haversine_km(
        df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"]
    )
    df["distance_band"] = pd.cut(
        df["distance_km"],
        bins=[-1, 50, 200, 500, 1000, 2000, 100_000],
        labels=["<50 km", "50-200", "200-500", "500-1000", "1000-2000", "2000+ km"],
    )

    # Outcome
    df["is_bad_review"] = np.where(
        df["review_score"].notna(),
        (df["review_score"] <= C.BAD_REVIEW_MAX).astype("float64"),
        np.nan,
    ).astype("float64")
    df["has_comment"] = df["review_comment_message"].notna()

    # Guarantee float dtype on every continuous column so that downstream
    # aggregation, plotting and modelling never receive object arrays.
    for col in ("approval_days", "handling_days", "transit_days",
                "delivery_days", "promised_days", "delay_days",
                "handover_slack_days", "item_total", "freight_total",
                "order_value", "payment_value", "freight_ratio",
                "distance_km", "review_score", "promise_padding_days",
                "handling_share", "review_response_hours",
                "avg_weight_g", "avg_photos", "avg_description_length"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def sanity_checks(df: pd.DataFrame) -> None:
    """Fail loudly if the fact table violates an invariant we depend on."""
    assert df["order_id"].is_unique, "fact table grain broken: duplicate order_id"

    neg = (df.loc[df.is_delivered, "delivery_days"] < 0).sum()
    assert neg == 0, f"{neg} delivered orders have negative delivery time"

    # Every delivered order must have a resolvable lateness flag.
    bad = df.loc[df.is_delivered, "is_late"].isna().sum()
    assert bad == 0, f"{bad} delivered orders have an unknown lateness flag"

    print("Sanity checks passed")


def main() -> None:
    con = connect()
    con.execute(ITEM_ROLLUP)
    con.execute(PAYMENT_ROLLUP)
    con.execute(FACTS)

    df = con.execute("SELECT * FROM fact_orders").df()
    con.close()

    df = derive(df)
    sanity_checks(df)

    # review_comment_message is only needed by the text module; keeping it in
    # the main parquet triples the file size, so it is written separately.
    text = df.loc[df.has_comment, ["order_id", "review_score",
                                   "review_comment_message", "main_category",
                                   "is_late"]]
    text.to_parquet(C.PROCESSED_DIR / "review_text.parquet", index=False)

    df.drop(columns=["review_comment_message"]).to_parquet(
        C.FACTS_PATH, index=False
    )

    print(f"fact_orders: {len(df):,} rows x {df.shape[1]} cols -> {C.FACTS_PATH}")
    print(f"  delivered with review: "
          f"{int((df.is_delivered & df.review_score.notna()).sum()):,}")
    print(f"  review comments kept:  {len(text):,}")


if __name__ == "__main__":
    main()
