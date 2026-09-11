"""Stage 7 - Figure export.

Writes every figure twice: a PNG for the report and slides, and a
self-contained HTML for anyone who wants to hover over the numbers.

Run:  python -m src.export_figures
"""

from __future__ import annotations

import os
import shutil

# Static PNG export goes through Kaleido, which drives a headless Chrome.
# Point it at a system Chrome if one is present; otherwise Kaleido will look
# for its own download. HTML export never needs this.
for _candidate in ("/opt/google/chrome/chrome", "/usr/bin/chromium",
                   "/usr/bin/google-chrome"):
    if os.path.exists(_candidate):
        os.environ.setdefault("BROWSER_PATH", _candidate)
        break
else:  # pragma: no cover - environment dependent
    _found = shutil.which("chromium") or shutil.which("google-chrome")
    if _found:
        os.environ.setdefault("BROWSER_PATH", _found)

import pandas as pd

from src import analysis as A
from src import config as C
from src import simulate as S
from src import viz as V


def export(fig, name: str, width: int = 1100, height: int | None = None) -> None:
    """Write one figure as PNG and HTML.

    The HTML always succeeds. PNG needs a Chrome binary, so a missing browser
    degrades to a warning rather than killing the run: the dashboard and the
    interactive figures do not depend on it.
    """
    h = height or fig.layout.height or 450
    fig.write_html(C.FIGURE_DIR / f"{name}.html", include_plotlyjs="cdn",
                   full_html=True)
    try:
        fig.write_image(C.FIGURE_DIR / f"{name}.png", width=width, height=h,
                        scale=2)
        print(f"  {name}  (png + html)")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  {name}  (html only - PNG export unavailable: "
              f"{type(exc).__name__})")


def main() -> None:
    base = A.analysis_base(A.load_facts())
    baseline = float(base.is_bad_review.mean())
    monthly = A.monthly_trend(base)

    print("Exporting figures to", C.FIGURE_DIR)
    export(V.fig_growth_vs_satisfaction(monthly), "01_growth_vs_satisfaction")
    export(V.fig_late_and_detractors(monthly), "02_late_and_detractors")
    export(V.fig_delay_damage(A.delay_curve(base)), "03_delay_damage_curve")
    export(V.fig_journey(A.journey_stages(base)), "04_journey_stages")
    export(V.fig_distance(A.basket_effects(base)), "05_distance_effect")
    export(V.fig_promise_gap(A.promise_analysis(base)), "06_promise_padding")
    export(V.fig_category_risk(A.category_view(base, baseline)),
           "07_category_risk")
    state = A.state_view(base, baseline)
    export(V.fig_state_map(state), "08_state_map")
    export(V.fig_state_bars(state), "08b_state_performance")
    export(V.fig_lane_risk(A.lane_view(base, baseline)), "09_lane_risk")

    sc = A.seller_scorecard(base, baseline)
    export(V.fig_seller_quadrant(sc), "10_seller_quadrant")
    export(V.fig_action_treemap(sc), "11_seller_actions")

    curve = S.fit_delay_curve(base)
    export(V.fig_sim_curve(curve), "12_damage_curve_model")

    rows = []
    for name, preset in S.preset_scenarios().items():
        r = S.run(base, preset, curve)
        rows.append({
            "scenario": name,
            "detractors_avoided": r["detractors_avoided"],
            "detractors_avoided_pct": r["detractors_avoided_pct"],
            "orders_lost": r["orders_lost"],
            "gmv_lost": r["gmv_lost"],
            "gmv_lost_pct": r["gmv_lost_pct"],
            "late_rate_after": r["after"]["late_rate"],
            "net_value": r["net_value"],
            "notes": preset.notes,
        })
    scen = pd.DataFrame(rows)
    scen.to_csv(C.TABLE_DIR / "scenarios.csv", index=False)
    export(V.fig_scenario_compare(scen), "13_scenario_comparison")

    imp = pd.read_csv(C.TABLE_DIR / "model_importance.csv")
    export(V.fig_importance(imp, "at_risk"), "14_model_drivers")
    export(V.fig_calibration(pd.read_csv(C.TABLE_DIR / "model_calibration.csv")),
           "15_model_calibration")

    print(f"\nDone. {len(list(C.FIGURE_DIR.glob('*.png')))} PNG files written.")


if __name__ == "__main__":
    main()
