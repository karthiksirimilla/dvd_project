"""Marketplace Health Console - interactive dashboard.

Run from the project root:
    streamlit run app/streamlit_app.py

Reads only data/processed/order_facts.parquet and outputs/tables/*, so it
starts in about a second and never re-derives a number the analysis layer
already produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import analysis as A  # noqa: E402
from src import config as C  # noqa: E402
from src import simulate as S  # noqa: E402
from src import viz as V  # noqa: E402

st.set_page_config(
    page_title="Marketplace Health Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling. Slate structure, one amber signal colour, generous whitespace.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
      html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
      .block-container { padding-top: 2.2rem; max-width: 1500px; }
      h1, h2, h3 { color: #1f3a68; letter-spacing: -0.01em; }
      .lede { color: #5c667c; font-size: 1.02rem; line-height: 1.55;
              max-width: 70ch; }
      .metric-row { display: flex; gap: 14px; flex-wrap: wrap; }
      .kpi { flex: 1 1 170px; border: 1px solid #e6e9ef; border-left: 3px solid #1f3a68;
             padding: 14px 16px; background: #fbfcfe; }
      .kpi .v { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem;
                color: #1f3a68; line-height: 1.1; }
      .kpi .l { font-size: 0.78rem; color: #6b7488; margin-top: 4px; }
      .kpi.alert { border-left-color: #c1462f; }
      .kpi.alert .v { color: #c1462f; }
      .kpi.good { border-left-color: #2e8b6f; }
      .kpi.good .v { color: #2e8b6f; }
      .takeaway { border-left: 3px solid #e8833a; background: #fdf7f1;
                  padding: 12px 16px; margin: 10px 0 18px 0; color: #45403a; }
      .stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding: 8px 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading order history")
def load_base() -> pd.DataFrame:
    return A.analysis_base(A.load_facts())


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(C.TABLE_DIR / f"{name}.csv")


@st.cache_data
def load_kpis() -> dict:
    with open(C.TABLE_DIR / "kpis.json") as fh:
        return json.load(fh)


@st.cache_data
def build_curve(_base: pd.DataFrame) -> pd.DataFrame:
    return S.fit_delay_curve(_base)


try:
    base_all = load_base()
except FileNotFoundError:
    st.error(
        "No processed data found. Run `make all` (or `python -m src.ingest && "
        "python -m src.features && python -m src.analysis`) first."
    )
    st.stop()

kpis = load_kpis()


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filter the marketplace")
    st.caption("Every tab responds to these. Charts recompute from the order "
               "history, not from a cached summary.")

    months = sorted(base_all.purchase_month.unique())
    m_from, m_to = st.select_slider(
        "Period",
        options=months,
        value=(months[0], months[-1]),
        format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"),
    )

    states = ["All"] + sorted(base_all.customer_state.dropna().unique())
    state_sel = st.multiselect("Customer state", states, default=["All"])

    cats = ["All"] + sorted(base_all.main_category.dropna().unique())
    cat_sel = st.multiselect("Product category", cats, default=["All"])

    only_late = st.checkbox("Late deliveries only", value=False)

    st.divider()
    st.caption(
        f"Source: {kpis['orders_total']:,} orders, of which "
        f"{kpis['orders_analysed']:,} were delivered and reviewed. "
        "Satisfaction figures use that reviewed population."
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.purchase_month >= m_from) & (df.purchase_month <= m_to)]
    if state_sel and "All" not in state_sel:
        d = d[d.customer_state.isin(state_sel)]
    if cat_sel and "All" not in cat_sel:
        d = d[d.main_category.isin(cat_sel)]
    if only_late:
        d = d[d.is_late.astype(bool)]
    return d


base = apply_filters(base_all)

if base.empty:
    st.warning("No orders match these filters. Widen the period or clear a "
               "category selection.")
    st.stop()

baseline_bad = float(base.is_bad_review.mean())


def kpi_card(value: str, label: str, tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="v">{value}</div>' \
           f'<div class="l">{label}</div></div>'


def kpi_row(cards: list[str]) -> None:
    st.markdown(f'<div class="metric-row">{"".join(cards)}</div>',
                unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.markdown(f'<div class="takeaway">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Marketplace Health Console")
st.markdown(
    '<p class="lede">Growth and customer experience pull against each other. '
    'This console shows where the marketplace is buying growth with goodwill, '
    'and what each available intervention would cost.</p>',
    unsafe_allow_html=True,
)

tab_health, tab_delivery, tab_segments, tab_sellers, tab_sim, tab_about = st.tabs(
    ["Marketplace health", "Delivery engine", "Categories and regions",
     "Seller scorecard", "What if", "About this data"]
)

# --------------------------------------------------------------------------
# Tab 1 - health
# --------------------------------------------------------------------------
with tab_health:
    late_rate = base.is_late.mean()
    bad_rate = base.is_bad_review.mean()
    kpi_row([
        kpi_card(f"{len(base):,}", "Orders delivered and reviewed"),
        kpi_card(f"R$ {base.item_total.sum():,.0f}", "Gross merchandise value"),
        kpi_card(f"{base.review_score.mean():.2f}", "Average review score"),
        kpi_card(f"{bad_rate:.1%}", "One and two star reviews",
                 "alert" if bad_rate > 0.12 else "good"),
        kpi_card(f"{late_rate:.1%}", "Delivered after the promised date",
                 "alert" if late_rate > 0.08 else "good"),
        kpi_card(f"{base.delivery_days.mean():.1f} d", "Average time to deliver"),
    ])

    st.subheader("Growth is not the problem. Keeping it is.")
    monthly = A.monthly_trend(base)
    st.plotly_chart(V.fig_growth_vs_satisfaction(monthly),
                    width="stretch")
    takeaway(
        f"GMV grew steadily while the average review score stayed near "
        f"{base.review_score.mean():.2f}. The marketplace is not yet trading "
        f"satisfaction for growth at an aggregate level, which is exactly why "
        f"the risk is easy to miss: it is concentrated in specific lanes, "
        f"categories and sellers rather than spread across the business."
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Late deliveries and detractors move together**")
        st.plotly_chart(V.fig_late_and_detractors(monthly),
                        width="stretch")
    with c2:
        st.markdown("**Why one bad review costs so much**")
        st.markdown(
            f"""
Only **{kpis['repeat_customer_rate']:.1%}** of customers ever place a second
order. In a marketplace with almost no repeat purchase, a detractor is not a
customer to win back. They are a customer already gone.

That reframes satisfaction from a soft metric into an acquisition cost. Every
one-star review means the money spent acquiring that customer bought exactly
one order.

- On-time orders average **{kpis['score_on_time']:.2f}** stars
- Late orders average **{kpis['score_late']:.2f}** stars
- **{kpis['detractor_share_from_lateness']:.0%}** of all detractors are
  attributable to lateness alone
            """
        )

# --------------------------------------------------------------------------
# Tab 2 - delivery
# --------------------------------------------------------------------------
with tab_delivery:
    st.subheader("Lateness does not hurt gradually. It falls off a cliff.")
    st.plotly_chart(V.fig_delay_damage(A.delay_curve(base)),
                    width="stretch")
    takeaway(
        "Arriving two days early and fifteen days early score the same, so "
        "there is no satisfaction return on being very early. Past the "
        "promised date the score collapses. Being slightly late is survivable; "
        "being a week late is not. The target is the tail, not the average."
    )

    st.subheader("The delay is in transit, not with the seller")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(V.fig_journey(A.journey_stages(base)),
                        width="stretch")
    with c2:
        st.markdown(
            f"""
Sellers hand parcels to the carrier in about
**{base.handling_days.mean():.1f} days**. The carrier then takes about
**{base.transit_days.mean():.1f} days**.

Correlation with the review score is
**{base.review_score.corr(base.handling_days):.2f}** for seller handling and
**{base.review_score.corr(base.transit_days):.2f}** for carrier transit.

The instinct to police slow sellers targets the smaller half of the problem.
            """
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Distance drives time, and time drives the score**")
        st.plotly_chart(V.fig_distance(A.basket_effects(base)),
                        width="stretch")
    with c4:
        st.markdown("**The promise carries a lot of unused slack**")
        st.plotly_chart(V.fig_promise_gap(A.promise_analysis(base)),
                        width="stretch")
    takeaway(
        f"The median order is promised in "
        f"{base.promised_days.median():.0f} days and arrives in "
        f"{base.delivery_days.median():.0f}. That padding looks like waste, "
        f"but the What if tab shows it is the single cheapest thing protecting "
        f"the current on-time rate."
    )

# --------------------------------------------------------------------------
# Tab 3 - segments
# --------------------------------------------------------------------------
with tab_segments:
    st.subheader("Where the goodwill is leaking")
    st.caption(
        "Satisfaction-at-Risk counts the one and two star reviews a segment "
        "produces beyond what its size would predict, then prices them at the "
        "segment's average order value."
    )

    cat = A.category_view(base, baseline_bad)
    state = A.state_view(base, baseline_bad)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Categories: volume against satisfaction**")
        st.plotly_chart(V.fig_category_risk(cat), width="stretch")
    with c2:
        st.markdown("**Regions: score, size and speed**")
        st.plotly_chart(V.fig_state_map(state), width="stretch")

    st.markdown("**The shipping lanes doing the most damage**")
    lane = A.lane_view(base, baseline_bad)
    if lane.empty:
        st.info("No lane has enough orders under the current filters.")
    else:
        st.plotly_chart(V.fig_lane_risk(lane), width="stretch")

    st.markdown("**Structural effects worth knowing**")
    eff = A.basket_effects(base)
    dim = st.selectbox("Dimension", sorted(eff.dimension.unique()),
                       index=list(sorted(eff.dimension.unique())).index(
                           "Sellers in order"))
    show = eff[eff.dimension == dim][
        ["level", "orders", "avg_score", "bad_rate", "late_rate",
         "avg_delivery_days"]
    ]
    st.dataframe(
        show.style.format({"orders": "{:,.0f}", "avg_score": "{:.2f}",
                           "bad_rate": "{:.1%}", "late_rate": "{:.1%}",
                           "avg_delivery_days": "{:.1f}"}),
        width="stretch", hide_index=True,
        column_config={
            "level": st.column_config.Column("Group"),
            "orders": st.column_config.Column("Orders"),
            "avg_score": st.column_config.Column("Avg. score"),
            "bad_rate": st.column_config.Column("Bad rate"),
            "late_rate": st.column_config.Column("Late rate"),
            "avg_delivery_days": st.column_config.Column("Avg. delivery (days)"),
        },
    )
    if dim == "Sellers in order":
        takeaway(
            f"Orders filled by one seller average "
            f"{base[~base.is_multi_seller].review_score.mean():.2f} stars. "
            f"Orders split across two or more sellers average "
            f"{base[base.is_multi_seller].review_score.mean():.2f}. Split "
            f"shipments are only "
            f"{base.is_multi_seller.mean():.1%} of orders, which makes this "
            f"the cheapest fix available: the customer waits for the slowest "
            f"parcel and reviews the whole order on it."
        )

    with st.expander("Category league table"):
        st.dataframe(
            cat[["main_category", "orders", "avg_score", "bad_rate",
                 "late_rate", "gmv", "excess_detractors", "gmv_exposure"]]
            .style.format({"orders": "{:,.0f}", "avg_score": "{:.2f}",
                           "bad_rate": "{:.1%}", "late_rate": "{:.1%}",
                           "gmv": "R$ {:,.0f}",
                           "excess_detractors": "{:,.0f}",
                           "gmv_exposure": "R$ {:,.0f}"}),
            width="stretch", hide_index=True,
            column_config={
                "main_category": st.column_config.Column("Category"),
                "orders": st.column_config.Column("Orders"),
                "avg_score": st.column_config.Column("Avg. score"),
                "bad_rate": st.column_config.Column("Bad rate"),
                "late_rate": st.column_config.Column("Late rate"),
                "gmv": st.column_config.Column("GMV"),
                "excess_detractors": st.column_config.Column(
                    "Excess detractors",
                    help="1-2 star reviews beyond what the category's size "
                         "would predict, at the marketplace's average rate."),
                "gmv_exposure": st.column_config.Column(
                    "GMV exposure",
                    help="Excess detractors priced at the category's "
                         "average order value."),
            },
        )

# --------------------------------------------------------------------------
# Tab 4 - sellers
# --------------------------------------------------------------------------
with tab_sellers:
    st.subheader("Every seller gets a verdict, not just a score")
    sc = A.seller_scorecard(base, baseline_bad)
    rated = sc[sc.orders >= C.SELLER_MIN_ORDERS]

    counts = sc.action.value_counts()
    gmv_share = sc.groupby("action").gmv.sum() / sc.gmv.sum()
    kpi_row([
        kpi_card(f"{counts.get('Invest', 0):,}", "Invest: healthy and growing",
                 "good"),
        kpi_card(f"{counts.get('Coach: fulfilment', 0):,}",
                 "Coach on fulfilment"),
        kpi_card(f"{counts.get('Coach: product quality', 0):,}",
                 "Coach on product quality"),
        kpi_card(f"{counts.get('Remove', 0):,}", "Remove from the catalogue",
                 "alert"),
        kpi_card(f"{gmv_share.get('Remove', 0):.2%}",
                 "Share of GMV held by removals", "alert"),
    ])

    st.plotly_chart(V.fig_seller_quadrant(sc), width="stretch")
    st.plotly_chart(V.fig_action_treemap(sc), width="stretch")

    takeaway(
        f"Of {len(rated):,} sellers with {C.SELLER_MIN_ORDERS}+ orders, only "
        f"{counts.get('Remove', 0):,} fail on both quality and delivery, and "
        f"they hold {gmv_share.get('Remove', 0):.2%} of GMV. Cleaning up the "
        f"catalogue is not the growth sacrifice leadership assumes it is."
    )

    st.markdown("**Look up a seller**")
    action_filter = st.multiselect(
        "Show", sorted(sc.action.unique()),
        default=[a for a in sc.action.unique() if a != "Monitor (low volume)"])
    view = sc[sc.action.isin(action_filter)][
        ["seller_id", "state", "top_category", "orders", "gmv", "avg_score",
         "bad_rate", "late_rate", "avg_handling_days", "excess_detractors",
         "action"]
    ].sort_values("gmv", ascending=False)
    st.dataframe(
        view.style.format({"orders": "{:,.0f}", "gmv": "R$ {:,.0f}",
                           "avg_score": "{:.2f}", "bad_rate": "{:.1%}",
                           "late_rate": "{:.1%}",
                           "avg_handling_days": "{:.1f}",
                           "excess_detractors": "{:,.1f}"}),
        width="stretch", hide_index=True, height=420,
        column_config={
            "seller_id": st.column_config.Column("Seller ID"),
            "state": st.column_config.Column("State"),
            "top_category": st.column_config.Column("Top category"),
            "orders": st.column_config.Column("Orders"),
            "gmv": st.column_config.Column("GMV"),
            "avg_score": st.column_config.Column("Avg. score"),
            "bad_rate": st.column_config.Column("Bad rate"),
            "late_rate": st.column_config.Column("Late rate"),
            "avg_handling_days": st.column_config.Column("Avg. handling (days)"),
            "excess_detractors": st.column_config.Column(
                "Excess detractors",
                help="1-2 star reviews beyond what this seller's volume "
                     "would predict, at the marketplace's average rate."),
            "action": st.column_config.Column("Recommended action"),
        },
    )

# --------------------------------------------------------------------------
# Tab 5 - simulator
# --------------------------------------------------------------------------
with tab_sim:
    st.subheader("Trade growth against experience, before committing")
    st.caption(
        "Each lever is replayed against the real order history. The detractor "
        "count comes from the observed relationship between lateness and "
        "review score, shown at the bottom of this tab."
    )

    curve = build_curve(base)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Logistics investment**")
        cut = st.slider("Cut carrier transit time by", 0, 50, 25, 5,
                        format="%d%%")
        target = st.radio(
            "Applied to",
            ["worst_lanes", "cross_state", "all"],
            format_func={"worst_lanes": "Worst lanes only",
                         "cross_state": "All cross-state shipments",
                         "all": "The whole network"}.get,
        )
    with c2:
        st.markdown("**Delivery promise**")
        trim = st.slider("Shorten the quoted ETA by", 0, 10, 0, 1,
                         format="%d days")
        st.caption("Tightening the promise wins trust at checkout but moves "
                   "the deadline the customer judges you against.")
    with c3:
        st.markdown("**Catalogue quality**")
        do_remove = st.checkbox("Remove underperforming sellers", value=False)
        thresh = st.slider("Remove sellers averaging below", 2.0, 4.5, 3.5, 0.1,
                           disabled=not do_remove)

    scenario = S.Scenario(
        transit_cut_pct=cut,
        transit_target=target,
        promise_trim_days=trim,
        remove_below_score=thresh if do_remove else None,
    )
    r = S.run(base, scenario, curve)

    avoided = r["detractors_avoided"]
    kpi_row([
        kpi_card(f"{avoided:,.0f}", "One and two star reviews avoided",
                 "good" if avoided > 0 else "alert"),
        kpi_card(f"{r['after']['late_rate']:.1%}", "Late rate after the change",
                 "good" if r["late_rate_delta"] < 0 else "alert"),
        kpi_card(f"R$ {r['gmv_lost']:,.0f}", "GMV given up",
                 "alert" if r["gmv_lost"] > 0 else ""),
        kpi_card(f"{r['sellers_removed']:,}", "Sellers removed"),
        kpi_card(f"R$ {r['net_value']:,.0f}", "Net value",
                 "good" if r["net_value"] > 0 else "alert"),
    ])

    if trim > 0 and avoided < 0:
        st.warning(
            f"Shortening the promise by {trim} days pushes the late rate from "
            f"{r['baseline']['late_rate']:.1%} to "
            f"{r['after']['late_rate']:.1%} and creates "
            f"{-avoided:,.0f} additional detractors. The padding in the "
            f"current ETA is doing real work. Buy the speed first, then spend "
            f"it on a shorter promise."
        )

    st.markdown("**Compare the standard programmes**")
    rows = []
    for name, sc_preset in S.preset_scenarios().items():
        res = S.run(base, sc_preset, curve)
        rows.append({
            "scenario": name,
            "detractors_avoided": res["detractors_avoided"],
            "orders_lost": res["orders_lost"],
            "gmv_lost": res["gmv_lost"],
            "late_rate_after": res["after"]["late_rate"],
            "net_value": res["net_value"],
            "what it does": sc_preset.notes,
        })
    comp = pd.DataFrame(rows)
    st.plotly_chart(V.fig_scenario_compare(comp), width="stretch")
    st.dataframe(
        comp.style.format({"detractors_avoided": "{:,.0f}",
                           "orders_lost": "{:,.0f}",
                           "gmv_lost": "R$ {:,.0f}",
                           "late_rate_after": "{:.1%}",
                           "net_value": "R$ {:,.0f}"}),
        width="stretch", hide_index=True,
        column_config={
            "scenario": st.column_config.Column("Scenario"),
            "detractors_avoided": st.column_config.Column("Detractors avoided"),
            "orders_lost": st.column_config.Column("Orders lost"),
            "gmv_lost": st.column_config.Column("GMV lost"),
            "late_rate_after": st.column_config.Column("Late rate after"),
            "net_value": st.column_config.Column("Net value"),
            "what it does": st.column_config.Column("What it does"),
        },
    )

    st.markdown("**The curve every number above rests on**")
    st.plotly_chart(V.fig_sim_curve(curve), width="stretch")
    st.caption(
        "Assumptions: lateness is treated as causal; a removed seller's orders "
        "are treated as lost rather than recaptured, so the GMV cost shown is "
        "a worst case; a shorter promise is assumed not to change demand, so "
        "any conversion benefit is excluded."
    )

# --------------------------------------------------------------------------
# Tab 6 - about the data
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("What this dashboard is built on")
    st.markdown(
        f"""An online marketplace's order history: {kpis['orders_total']:,}
orders placed between September 2016 and October 2018, spanning eleven
tables - the order lifecycle, what was in each order, how it was paid,
what the customer said about it, and the marketing funnel that brought
sellers onto the platform in the first place.

Everything on the other five tabs is computed from
**{kpis['orders_analysed']:,} of those orders** - the ones that were both
delivered and reviewed. Orders that never completed, or completed but got
no review, are shown in the funnel view but left out of every score,
rate and dollar figure, because there is nothing to average for them.
The window is also trimmed to **{pd.Timestamp(C.ANALYSIS_START).strftime('%b %Y')}
through {pd.Timestamp(C.ANALYSIS_END).strftime('%b %Y')}** - the months on
either side have too few orders for a monthly trend to mean anything."""
    )

    st.markdown("**The eleven source tables**")
    sources = pd.DataFrame([
        {"table": "orders", "file": "orders_dataset.csv",
         "what it holds": "One row per order: every lifecycle timestamp "
                          "(purchase, payment, carrier handoff, delivery) "
                          "and the order status."},
        {"table": "order_items", "file": "order_items_dataset.csv",
         "what it holds": "One row per item in an order - an order can "
                          "hold several, possibly from different sellers. "
                          "Price, freight, product and seller."},
        {"table": "order_payments", "file": "order_payments_dataset.csv",
         "what it holds": "How an order was paid: method, installments, "
                          "value."},
        {"table": "order_reviews", "file": "order_reviews_dataset.csv",
         "what it holds": "The 1-5 star score and optional written "
                          "comment a customer left for an order."},
        {"table": "customers", "file": "customers_dataset.csv",
         "what it holds": "Delivery location for an order. customer_id "
                          "is minted fresh per order; "
                          "customer_unique_id is the actual person."},
        {"table": "sellers", "file": "sellers_dataset.csv",
         "what it holds": "Which state a seller ships from."},
        {"table": "products", "file": "products_dataset.csv",
         "what it holds": "Category, weight and dimensions for a "
                          "product."},
        {"table": "category_translation",
         "file": "product_category_name_translation.csv",
         "what it holds": "Maps the Portuguese product category name "
                          "to English."},
        {"table": "geolocation", "file": "geolocation_dataset.csv",
         "what it holds": "Sampled latitude/longitude per zip-code "
                          "prefix - used to place states on the map."},
        {"table": "mql",
         "file": "marketing_qualified_leads_dataset.csv",
         "what it holds": "Sellers who entered the marketing funnel "
                          "before ever listing a product."},
        {"table": "closed_deals", "file": "closed_deals_dataset.csv",
         "what it holds": "Which of those leads actually converted "
                          "into a seller who went on to transact."},
    ])
    st.dataframe(
        sources, width="stretch", hide_index=True, height=422,
        column_config={
            "table": st.column_config.Column("Table", width="small"),
            "file": st.column_config.Column("Source file", width="medium"),
            "what it holds": st.column_config.Column("What it holds",
                                                      width="large"),
        },
    )

    st.markdown("**Terms used throughout**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
- **GMV** - gross merchandise value, the sum of item prices (freight
  excluded).
- **Late** - delivered after the promised date. Late rate is the share
  of orders that qualify.
- **Detractor / bad review** - a 1 or 2 star review, out of 5.
- **Satisfaction-at-Risk / excess detractors** - how many extra 1-2 star
  reviews a segment produces beyond what its size alone would predict,
  at the marketplace's overall bad-rate.
- **GMV exposure** - excess detractors priced at the segment's average
  order value. A dollar estimate of the goodwill at stake.
"""
        )
    with c2:
        st.markdown(
            f"""
- **Handling days** - from payment to the carrier collecting the
  parcel: the seller's part of the journey.
- **Transit days** - from carrier pickup to the customer receiving it:
  the carrier's part.
- **Promise padding** - the gap between the quoted delivery date and
  when the order actually arrives.
- **Seller verdict** - Invest (score ≥ {C.SELLER_BAD_SCORE}, late rate
  ≤ {C.SELLER_LATE_RATE:.0%}), Coach: fulfilment (good score, too
  slow), Coach: product quality (on time, poorly rated), Remove (fails
  both), Monitor (fewer than {C.SELLER_MIN_ORDERS} orders - too little
  history to judge).
"""
        )

    st.markdown("**Known gaps and how they're handled**")
    st.caption(
        "Every quirk the ingest step found, paired with the decision made "
        "about it - so a number on the other tabs can be traced back to a "
        "judgement call instead of taken on faith."
    )
    audit = load_table("data_quality_audit")
    # Pre-format rather than lean on Styler's na_rep: st.dataframe renders
    # a Styler's own values, not its na_rep substitution, so a genuinely
    # missing share was showing up as the raw word "None" instead of a
    # placeholder.
    audit = audit.assign(
        rows=audit.rows.map("{:,.0f}".format),
        share=audit.share.map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a"),
    )
    st.dataframe(
        audit,
        width="stretch", hide_index=True,
        column_config={
            "area": st.column_config.Column("Area"),
            "finding": st.column_config.Column("Finding", width="large"),
            "rows": st.column_config.Column("Rows"),
            "share": st.column_config.Column("Share"),
            "treatment": st.column_config.Column("How it's handled",
                                                  width="large"),
        },
    )
