"""Central configuration: paths, thresholds and shared constants.

Every module imports from here so that the pipeline can be relocated by
changing a single file (or the MI_DATA_DIR environment variable).
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = Path(os.environ.get("MI_DATA_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

DB_PATH = PROCESSED_DIR / "marketplace.duckdb"
FACTS_PATH = PROCESSED_DIR / "order_facts.parquet"

for _d in (PROCESSED_DIR, OUTPUT_DIR, TABLE_DIR, REPORT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Raw table registry: logical name -> filename
# --------------------------------------------------------------------------
RAW_TABLES = {
    "orders": "orders_dataset.csv",
    "order_items": "order_items_dataset.csv",
    "order_payments": "order_payments_dataset.csv",
    "order_reviews": "order_reviews_dataset.csv",
    "customers": "customers_dataset.csv",
    "sellers": "sellers_dataset.csv",
    "products": "products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "geolocation": "geolocation_dataset.csv",
    "mql": "marketing_qualified_leads_dataset.csv",
    "closed_deals": "closed_deals_dataset.csv",
}

# --------------------------------------------------------------------------
# Analysis constants
# --------------------------------------------------------------------------

# A review of 1 or 2 stars is treated as a "detractor" / bad review.
BAD_REVIEW_MAX = 2

# Minimum order volume before a seller is judged on its average score.
# Below this, a single unhappy customer swings the mean too far.
SELLER_MIN_ORDERS = 30

# Minimum sample size before a category or state is shown in league tables.
SEGMENT_MIN_ORDERS = 500

# Seller scorecard quadrant thresholds.
SELLER_BAD_SCORE = 3.5      # average review below this = quality problem
SELLER_LATE_RATE = 0.10     # late rate above this = delivery problem

# Buckets used for the "damage curve" of lateness (days late vs review score).
DELAY_BUCKETS = [-10_000, -15, -10, -5, -2, 0, 2, 5, 10, 10_000]
DELAY_LABELS = [
    "15+ d early", "10-15 d early", "5-10 d early", "2-5 d early",
    "0-2 d early", "0-2 d late", "2-5 d late", "5-10 d late", "10+ d late",
]

# The reporting window. Data outside this range is sparse and distorts trends.
ANALYSIS_START = "2017-01-01"
ANALYSIS_END = "2018-08-31"

# Brand palette used across every figure and the dashboard.
PALETTE = {
    "primary": "#1f3a68",
    "accent": "#e8833a",
    "good": "#2e8b6f",
    "bad": "#c1462f",
    "warn": "#d9a441",
    "neutral": "#8a93a5",
    "bg": "#ffffff",
    "grid": "#e6e9ef",
}

DIVERGING = ["#c1462f", "#e08a5a", "#f0d9a8", "#8fbfa8", "#2e8b6f"]
