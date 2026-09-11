# Marketplace Insights

A study of a Brazilian e-commerce marketplace: whether growth is costing
customer satisfaction, where the damage comes from, and what each available
fix would cost.

The headline result is that growth and customer experience are not opposed
across the board. They are opposed in a few specific places, and most of
those cost far less to fix than the brief assumes. Late delivery is the
dominant driver of bad reviews (4.30 stars on time against 2.57 late), the
damage is a cliff rather than a slope, and most of the delay belongs to the
carrier rather than the seller.

## What's in the repository

```
dashboard/                                 pipeline, notebooks and the app
marketplace_analysis.ipynb                 the analysis as one standalone notebook
marketplace_analysis_with_outputs.ipynb    the same analysis in a first-person voice
requirements.txt                           runtime dependencies for the hosted app
.streamlit/config.toml                     theme for the hosted app
```

The raw data is not included. The CSVs come to about 120 MB and the DuckDB
build is larger than GitHub allows, so both are excluded and rebuilt locally.

## Getting the data

Download the Brazilian e-commerce dataset and its marketing funnel tables,
then put all eleven CSVs in `dashboard/data/raw/`:

```
customers_dataset.csv                   order_reviews_dataset.csv
geolocation_dataset.csv                 orders_dataset.csv
order_items_dataset.csv                 product_category_name_translation.csv
order_payments_dataset.csv              products_dataset.csv
sellers_dataset.csv                     closed_deals_dataset.csv
                                        marketing_qualified_leads_dataset.csv
```

## Running it

```bash
cd dashboard
python3 -m venv .venv && source .venv/bin/activate
make setup     # install dependencies
make all       # build every table, figure and notebook from the raw CSVs
make dash      # open the dashboard at localhost:8501
```

`dashboard/README.md` covers the pipeline stages and the layout in more
detail.

## The two root notebooks

Both cover the same analysis and produce the same charts. They differ in
voice: `marketplace_analysis.ipynb` is written in a neutral register, and
`marketplace_analysis_with_outputs.ipynb` is written in the first person.
Each is self-contained and already carries its outputs, so either can be
read start to finish without running anything.

The three notebooks under `dashboard/notebooks/` are the staged version of
the same study, one per week of the project. They are the more current of
the two sets, and `make all` re-executes them so their numbers stay tied to
the pipeline. Where a figure differs, treat the pipeline output as correct.
