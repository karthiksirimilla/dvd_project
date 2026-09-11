# Marketplace Insights Dashboard

An analytics pipeline plus a Streamlit dashboard, built on an e-commerce
marketplace dataset: orders, sellers, reviews, delivery logistics, and the
marketing funnel that feeds new sellers into the platform. It ingests the
raw CSVs, builds a clean order-level fact table, computes the headline
health metrics, trains a small "will this order get a bad review" model,
mines the review text for complaint themes, and serves all of it through
an interactive dashboard.

Nothing here calls out to the internet once the data is processed. The
dashboard just reads local parquet/CSV/JSON files, so it opens in about a
second and works fine offline.

## What's in the dashboard

Six tabs, each answering a different question:

- **Marketplace health**: the headline KPIs, average review score,
  bad-review rate, late-delivery rate, and how those move over time.
- **Delivery engine**: where orders lose time (handling vs. transit
  vs. promise padding) and how lateness drags review scores down.
- **Categories and regions**: which product categories and states
  are the biggest risk, weighted by volume.
- **Seller scorecard**: a quadrant view (quality problem vs.
  fulfilment problem) with a suggested action per seller: invest,
  coach, monitor, or remove.
- **What if**: a small simulator for testing how a change (e.g.
  tightening the delivery promise) would move the detractor rate.
- **About this data**: the source tables, a glossary of every term
  used elsewhere, and the data-quality audit.

## The notebooks

`notebooks/` holds the written analysis, one per week of the project.
Read them in order. They are built as a single argument, and each one
picks up where the last left off:

1. **`01_exploration_and_cleaning`**: the schema and the grain of each
   table, the data-quality audit and the two judgement calls that
   mattered, the population every later number is computed on, and the
   Satisfaction-at-Risk metric the rest of the project runs on.
2. **`02_exploratory_and_explanatory`**: the four questions answered.
   Whether growth is costing satisfaction, the shape of the damage from
   lateness, whose delay it actually is, and where the risk concentrates.
   Ends with an independent check of the delivery story against what
   customers wrote in Portuguese.
3. **`03_model_scenarios_recommendations`**: a detractor-risk model
   (and an honest account of what it can't do), a verdict for every
   seller, a price on each intervention, and the recommendations.

## Running it

You'll need the raw CSVs, which are not in the repository. Download the
Brazilian e-commerce dataset and its marketing funnel tables, then put all
eleven files in `data/raw/`:

```bash
mkdir -p data/raw       # copy the eleven CSVs in here

python3 -m venv .venv
source .venv/bin/activate
make setup   # installs everything in requirements.txt
make all     # ingest -> features -> analysis -> model -> text -> figures -> notebooks
make dash    # launches the dashboard at http://localhost:8501
```

`make all` is the whole pipeline end to end. It rebuilds
`data/processed/`, `outputs/tables/` and `reports/figures/`, then
re-executes the three notebooks under `notebooks/` against the result.
The dashboard only reads what `make all` produces, so re-run it
whenever the raw data changes.

The notebooks are **written by hand and only executed, never
regenerated**. The narrative in them is the point, not a by-product.
`make all` refills their code outputs and leaves every word of the
markdown alone, which keeps the numbers quoted in the prose honest
without putting the writing at risk. See `src/run_notebooks.py`.

`make test` sanity-checks the pipeline (22 tests covering the
ingest/feature/analysis logic). `make clean` wipes everything generated
and starts fresh.

## Layout

```
src/            the pipeline itself, one module per stage
  ingest.py       raw CSVs -> duckdb + a data-quality audit
  features.py     builds the order-level fact table
  analysis.py     KPIs, aggregates, Satisfaction-at-Risk scoring
  model.py        detractor-risk model (checkout-time vs. post-hoc)
  textmining.py   complaint themes from review text
  export_figures.py  static PNG/HTML charts for the report
  run_notebooks.py   re-runs the hand-written notebooks in place
  simulate.py     backs the "What if" tab
  viz.py          shared chart styling - every figure goes through here
  config.py       paths, thresholds, the brand palette - change once, applies everywhere
app/streamlit_app.py   the dashboard itself
.streamlit/config.toml the Streamlit theme, kept in sync with config.py's palette
tests/                 pytest suite for the pipeline
reports/figures/       exported charts, PNG and interactive HTML
reports/               the technical report source
```

## Notes

- The dashboard is intentionally read-only against pre-computed
  outputs. It doesn't touch duckdb or re-run the model live, which is
  why it stays fast even on the full ~100k order dataset.
- `src/config.py` is the one file to edit if you want to relocate the
  data directory (`MI_DATA_DIR` env var), change the "bad review"
  threshold, or tweak the seller scorecard cutoffs. `PALETTE["bg"]`
  there is the chart background: every figure exports on an opaque
  white canvas rather than a transparent one, so a chart pulled out of
  `reports/figures/` still reads correctly on a dark background (a
  dark-mode editor, a dark slide) instead of dropping its gridlines
  and text into an invisible void.
- The Streamlit theme in `.streamlit/config.toml` mirrors that same
  palette, so sliders, tabs, checkboxes and buttons pick up the brand's
  navy-and-amber instead of Streamlit's default red.
