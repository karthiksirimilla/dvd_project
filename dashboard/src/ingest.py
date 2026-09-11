"""Stage 1 - Ingest.

Loads the eleven raw CSVs into a DuckDB database and runs a data quality
audit. DuckDB is used rather than plain pandas because the geolocation table
has ~1M rows and the joins that build the fact table are far cheaper in SQL.

Run:  python -m src.ingest
"""

from __future__ import annotations

import duckdb
import pandas as pd

from src import config as C


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def connect() -> duckdb.DuckDBPyConnection:
    """Open (or create) the project database."""
    return duckdb.connect(str(C.DB_PATH))


def load_raw(con: duckdb.DuckDBPyConnection) -> None:
    """Register every raw CSV as a table named `raw_<logical name>`.

    read_csv_auto handles the quoted headers and the UTF-8 BOM present in
    product_category_name_translation.csv.
    """
    for name, filename in C.RAW_TABLES.items():
        path = C.RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw file {path}. Place the CSVs in {C.RAW_DIR} "
                f"or set MI_DATA_DIR."
            )
        con.execute(
            f"CREATE OR REPLACE TABLE raw_{name} AS "
            f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true, "
            f"sample_size=-1)"
        )
    print(f"Loaded {len(C.RAW_TABLES)} raw tables into {C.DB_PATH.name}")


def build_geo_lookup(con: duckdb.DuckDBPyConnection) -> None:
    """Collapse the 1M-row geolocation table to one point per zip prefix.

    The raw table holds many rows per prefix with noticeable GPS noise, and a
    handful of points fall outside Brazil's bounding box. We take the median
    coordinate per prefix after clipping to that box, which is robust to both.
    """
    con.execute(
        """
        CREATE OR REPLACE TABLE dim_geo AS
        SELECT
            geolocation_zip_code_prefix          AS zip_prefix,
            median(geolocation_lat)              AS lat,
            median(geolocation_lng)              AS lng,
            any_value(geolocation_state)         AS state
        FROM raw_geolocation
        WHERE geolocation_lat BETWEEN -34.0 AND 5.3
          AND geolocation_lng BETWEEN -74.0 AND -34.8
        GROUP BY 1
        """
    )
    n = con.execute("SELECT count(*) FROM dim_geo").fetchone()[0]
    print(f"dim_geo: {n:,} unique zip prefixes")


def dedupe_reviews(con: duckdb.DuckDBPyConnection) -> None:
    """One review per order.

    A small number of orders carry more than one review row (a follow-up
    survey). We keep the earliest review, which is the one that reflects the
    delivery experience being analysed.
    """
    con.execute(
        """
        CREATE OR REPLACE TABLE dim_reviews AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY order_id
                       ORDER BY review_creation_date, review_id
                   ) AS rn
            FROM raw_order_reviews
        ) WHERE rn = 1
        """
    )
    before = con.execute("SELECT count(*) FROM raw_order_reviews").fetchone()[0]
    after = con.execute("SELECT count(*) FROM dim_reviews").fetchone()[0]
    print(f"Reviews deduplicated: {before:,} -> {after:,}")


# --------------------------------------------------------------------------
# Quality audit
# --------------------------------------------------------------------------
def quality_audit(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Produce the missingness / integrity report used in the report and note.

    Each row is one finding: what it is, how big it is, and what we did about
    it. Writing the treatment down next to the finding is what makes this
    auditable rather than a wall of null counts.
    """
    checks: list[dict] = []

    def add(area, finding, n, pct, treatment):
        checks.append(
            {"area": area, "finding": finding, "rows": n,
             "share": round(pct, 4), "treatment": treatment}
        )

    n_orders = con.execute("SELECT count(*) FROM raw_orders").fetchone()[0]

    # 1. Timestamp gaps in the order lifecycle
    for col in ("order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date"):
        n = con.execute(
            f"SELECT count(*) FROM raw_orders WHERE {col} IS NULL"
        ).fetchone()[0]
        add("orders", f"{col} missing", n, n / n_orders,
            "Retained; excluded only from metrics that require the timestamp")

    # 2. Non-delivered orders
    n = con.execute(
        "SELECT count(*) FROM raw_orders WHERE order_status <> 'delivered'"
    ).fetchone()[0]
    add("orders", "order_status not 'delivered'", n, n / n_orders,
        "Excluded from the satisfaction model; kept for the funnel view")

    # 3. Delivered orders with no delivery timestamp (contradiction)
    n = con.execute(
        """SELECT count(*) FROM raw_orders
           WHERE order_status = 'delivered'
             AND order_delivered_customer_date IS NULL"""
    ).fetchone()[0]
    add("orders", "status='delivered' but no delivery date", n, n / n_orders,
        "Dropped from delivery analysis: the outcome date is unknowable")

    # 4. Lifecycle timestamps recorded out of order
    n = con.execute(
        """SELECT count(*) FROM raw_orders
           WHERE order_delivered_customer_date < order_delivered_carrier_date
              OR order_delivered_carrier_date < order_approved_at"""
    ).fetchone()[0]
    add("orders", "lifecycle timestamps out of order", n, n / n_orders,
        "Order kept; the impossible stage duration is nulled so it cannot "
        "distort stage averages")

    # 5. Products missing a category
    n_prod = con.execute("SELECT count(*) FROM raw_products").fetchone()[0]
    n = con.execute(
        "SELECT count(*) FROM raw_products WHERE product_category_name IS NULL"
    ).fetchone()[0]
    add("products", "product_category_name missing", n, n / n_prod,
        "Mapped to 'unknown' so the orders are not silently lost in joins")

    # 5. Categories with no English translation
    n = con.execute(
        """SELECT count(DISTINCT p.product_category_name)
           FROM raw_products p
           LEFT JOIN raw_category_translation t
             ON p.product_category_name = t.product_category_name
           WHERE p.product_category_name IS NOT NULL
             AND t.product_category_name_english IS NULL"""
    ).fetchone()[0]
    add("products", "categories missing English translation", n, float("nan"),
        "Fall back to the Portuguese name rather than dropping the row")

    # 6. Duplicate reviews
    n = con.execute(
        """SELECT count(*) FROM (
               SELECT order_id FROM raw_order_reviews
               GROUP BY 1 HAVING count(*) > 1)"""
    ).fetchone()[0]
    add("reviews", "orders with more than one review", n, float("nan"),
        "Kept the earliest review per order")

    # 7. Free-text coverage
    n_rev = con.execute("SELECT count(*) FROM raw_order_reviews").fetchone()[0]
    n = con.execute(
        "SELECT count(*) FROM raw_order_reviews WHERE review_comment_message IS NULL"
    ).fetchone()[0]
    add("reviews", "review with no written comment", n, n / n_rev,
        "Text analysis runs on the commented subset only; stated as a caveat")

    # 8. Geolocation noise
    n_geo = con.execute("SELECT count(*) FROM raw_geolocation").fetchone()[0]
    n = con.execute(
        """SELECT count(*) FROM raw_geolocation
           WHERE geolocation_lat NOT BETWEEN -34.0 AND 5.3
              OR geolocation_lng NOT BETWEEN -74.0 AND -34.8"""
    ).fetchone()[0]
    add("geolocation", "coordinates outside Brazil bounding box", n, n / n_geo,
        "Clipped before taking the median coordinate per zip prefix")

    # 9. Orders with no payment record
    n = con.execute(
        """SELECT count(*) FROM raw_orders o
           LEFT JOIN raw_order_payments p USING (order_id)
           WHERE p.order_id IS NULL"""
    ).fetchone()[0]
    add("payments", "orders with no payment row", n, n / n_orders,
        "Payment fields left null; order retained")

    # 10. Marketing funnel linkage
    n_deals = con.execute("SELECT count(*) FROM raw_closed_deals").fetchone()[0]
    n = con.execute(
        """SELECT count(*) FROM raw_closed_deals d
           WHERE NOT EXISTS (
               SELECT 1 FROM raw_order_items i WHERE i.seller_id = d.seller_id)"""
    ).fetchone()[0]
    add("marketing", "closed deals whose seller never transacted", n,
        n / n_deals,
        "Funnel analysis scoped to matched sellers; gap reported as a limitation")

    # 11. Negative or zero prices
    n = con.execute(
        "SELECT count(*) FROM raw_order_items WHERE price <= 0"
    ).fetchone()[0]
    add("order_items", "price <= 0", n, float("nan"),
        "None found" if n == 0 else "Excluded from value metrics")

    df = pd.DataFrame(checks)
    df.to_csv(C.TABLE_DIR / "data_quality_audit.csv", index=False)
    return df


def main() -> None:
    con = connect()
    load_raw(con)
    build_geo_lookup(con)
    dedupe_reviews(con)
    audit = quality_audit(con)
    print("\nData quality audit")
    print(audit.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
