"""Re-run the roadmap notebooks against the current analysis layer.

The notebooks under notebooks/ are written by hand - the prose is the point,
not a by-product - so nothing here generates them. This only executes them in
place: code cells get fresh outputs, markdown is left exactly as written.

That keeps the numbers quoted in the narrative honest. Re-run the pipeline and
every figure and table in the notebooks is recomputed from the same data the
dashboard reads, so the two can never quietly disagree.

Run:  python -m src.run_notebooks
"""

from __future__ import annotations

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src import config as C

NB_DIR = C.ROOT / "notebooks"

NOTEBOOKS = [
    "01_exploration_and_cleaning.ipynb",
    "02_exploratory_and_explanatory.ipynb",
    "03_model_scenarios_recommendations.ipynb",
]


def run(name: str) -> None:
    path = NB_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. These notebooks are hand-written and live in "
            f"version control; they are not generated, so a missing file has to "
            f"be restored rather than rebuilt."
        )

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        raise RuntimeError(
            f"{name} failed to execute. The notebook is left untouched on disk; "
            f"fix the analysis code or the offending cell and re-run."
        ) from exc

    nbformat.write(nb, path)

    figures = sum(
        1
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if "image/png" in out.get("data", {})
    )
    words = sum(
        len(cell.source.split()) for cell in nb.cells if cell.cell_type == "markdown"
    )
    print(f"  {path.relative_to(C.ROOT)}  "
          f"({len(nb.cells)} cells, {figures} figures, {words:,} words of narrative)")


def main() -> None:
    for name in NOTEBOOKS:
        run(name)


if __name__ == "__main__":
    main()
