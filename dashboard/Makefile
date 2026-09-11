# Marketplace Insights - reproducible build
#
# `make all` takes the raw CSVs in data/raw and rebuilds every table, figure
# and dashboard input from scratch, then re-runs the notebooks against the
# result. The notebooks themselves are hand-written and are only executed,
# never regenerated - see src/run_notebooks.py.

PY := python3

.PHONY: all setup ingest features analysis model text figures notebooks test dash clean help

help:
	@echo "make setup      Install dependencies"
	@echo "make all        Run the full pipeline end to end"
	@echo "make dash       Launch the interactive dashboard"
	@echo "make test       Run the test suite"
	@echo "make clean      Remove generated artefacts"

setup:
	$(PY) -m pip install -r requirements.txt

all: ingest features analysis model text figures notebooks
	@echo ""
	@echo "Build complete. Launch the dashboard with:  make dash"

ingest:
	@echo "==> 1/7 Ingest and quality audit"
	$(PY) -m src.ingest

features:
	@echo "==> 2/7 Feature engineering"
	$(PY) -m src.features

analysis:
	@echo "==> 3/7 Aggregates and Satisfaction-at-Risk"
	$(PY) -m src.analysis

model:
	@echo "==> 4/7 Detractor risk model"
	$(PY) -m src.model

text:
	@echo "==> 5/7 Review text mining"
	$(PY) -m src.textmining

figures:
	@echo "==> 6/7 Figure export"
	$(PY) -m src.export_figures

notebooks:
	@echo "==> 7/7 Notebook execution"
	$(PY) -m src.run_notebooks

test:
	$(PY) -m pytest tests -q

dash:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/processed/* outputs/tables/* reports/figures/* __pycache__ \
	       src/__pycache__ .pytest_cache
