"""Shared figure library.

Every chart in the report and the dashboard is built here, so the two can
never drift apart visually or numerically.

Design intent: this is an operations instrument panel, not a marketing page.
Deep slate for structure, a single amber signal colour for "attention here",
red and green reserved exclusively for bad and good outcomes so that colour
always means the same thing. Nothing is coloured for decoration.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config as C

FONT = "IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif"


def style(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    # `title` must be omitted outright when there is none - passing
    # title=None while still setting title_font leaves Plotly with a title
    # object that has a font but no text, and it renders the JS literal
    # "undefined" instead of just staying blank.
    layout: dict = dict(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, size=13, color="#2b3245"),
        margin=dict(l=60, r=30, t=60 if title else 30, b=50),
        plot_bgcolor=C.PALETTE["bg"],
        paper_bgcolor=C.PALETTE["bg"],
        # title=dict(text="") - plotly express titles the legend with the raw
        # dataframe column name ("action", "stage") by default; every other
        # chart here has no legend title, so drop it for consistency.
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    title=dict(text="")),
        hoverlabel=dict(font_family=FONT),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=17, color=C.PALETTE["primary"]))
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=C.PALETTE["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=C.PALETTE["grid"], zeroline=False)
    return fig


# --------------------------------------------------------------------------
# 1. Marketplace health
# --------------------------------------------------------------------------
def fig_score_distribution(base: pd.DataFrame) -> go.Figure:
    """The shape of the outcome variable, which decides how to model it.

    Red for the detractor scores, green for the promoter scores, grey for
    the three-star middle - the same colour meanings used everywhere else.
    The point of the chart is the hollow centre: averaging a distribution
    this bimodal lands on a value almost no review actually takes.
    """
    dist = base.review_score.value_counts().sort_index()
    share = dist / dist.sum()
    colors = [C.PALETTE["bad"], C.PALETTE["bad"], C.PALETTE["neutral"],
              C.PALETTE["good"], C.PALETTE["good"]]
    fig = go.Figure(go.Bar(
        x=dist.index.astype(int), y=dist.values,
        marker_color=[colors[int(s) - 1] for s in dist.index],
        text=[f"{v:.0%}" for v in share], textposition="outside",
        hovertemplate="%{x} star<br>%{y:,} orders<extra></extra>",
    ))
    fig.update_xaxes(title_text="Review score", dtick=1)
    fig.update_yaxes(title_text="Orders")
    return style(fig, 360)


def fig_growth_vs_satisfaction(monthly: pd.DataFrame) -> go.Figure:
    """The core tension in two aligned panels: GMV climbing, satisfaction flat.

    Not a dual-axis chart. Two y-scales on one plot let the alignment
    between reais and star-rating be picked arbitrarily, which can invent
    a correlation that isn't really there. Stacking GMV and score as two
    panels sharing an x-axis tells the same story without that risk.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(
        go.Bar(x=monthly.purchase_month, y=monthly.gmv, name="GMV",
               marker_color=C.PALETTE["primary"], showlegend=False,
               hovertemplate="%{x|%b %Y}<br>GMV R$%{y:,.0f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=monthly.purchase_month, y=monthly.avg_score,
                   name="Average review score", mode="lines+markers",
                   showlegend=False,
                   line=dict(color=C.PALETTE["accent"], width=3),
                   hovertemplate="%{x|%b %Y}<br>%{y:.2f} stars<extra></extra>"),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="GMV (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Average review score", range=[3.5, 4.6],
                     row=2, col=1)
    return style(fig, 460)


def fig_late_and_detractors(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly.purchase_month, y=monthly.late_rate * 100, name="Late deliveries",
        mode="lines+markers", line=dict(color=C.PALETTE["bad"], width=3)))
    fig.add_trace(go.Scatter(
        x=monthly.purchase_month, y=monthly.bad_rate * 100, name="1-2 star reviews",
        mode="lines+markers", line=dict(color=C.PALETTE["warn"], width=3,
                                        dash="dot")))
    fig.update_yaxes(title_text="% of orders", ticksuffix="%")
    return style(fig, 360)


# --------------------------------------------------------------------------
# 2. The delivery engine
# --------------------------------------------------------------------------
def fig_delay_damage(curve: pd.DataFrame) -> go.Figure:
    """Review score against lateness. The cliff is the point of the chart.

    One metric, not two: the bad-rate line used to ride a second y-axis
    alongside the score bars, but it moves in lockstep with the score by
    construction and the bars are already value-labelled, so the second
    axis bought a dual-scale chart nothing but risk of a misread.
    """
    colors = [C.PALETTE["good"] if "early" in b else C.PALETTE["bad"]
              for b in curve.delay_bucket]
    fig = go.Figure(go.Bar(
        x=curve.delay_bucket, y=curve.avg_score, marker_color=colors,
        text=curve.avg_score.round(2), textposition="outside",
        hovertemplate="%{x}<br>%{y:.2f} stars<br>"
                      "%{customdata:,} orders<extra></extra>",
        customdata=curve.orders,
    ))
    fig.update_yaxes(title_text="Average review score", range=[0, 5.4])
    return style(fig, 400)


def fig_journey(stages: pd.DataFrame, label_min_share: float = 0.06) -> go.Figure:
    """Where the days go, for on-time versus late orders.

    Payment approval is around half a day against a carrier transit of
    twenty-five, so its segment is far too narrow to hold a number. Labelling
    it anyway clips the text into an unreadable sliver, so a segment is only
    labelled once it is a reasonable share of its row; the rest stays on
    hover, where there is room for it.
    """
    d = stages.copy()
    share = d.days / d.groupby("group").days.transform("sum")
    d["label"] = d.days.round(1).astype(str).where(share >= label_min_share, "")

    fig = px.bar(
        d, x="days", y="group", color="stage", orientation="h",
        color_discrete_map={"Payment approval": C.PALETTE["neutral"],
                            "Seller handling": C.PALETTE["accent"],
                            "Carrier transit": C.PALETTE["primary"]},
        text="label",
        hover_data={"days": ":.2f", "label": False},
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle",
                      cliponaxis=False)
    fig.update_xaxes(title_text="Average days")
    fig.update_yaxes(title_text="")
    return style(fig, 300)


def fig_promise_gap(promise: pd.DataFrame, top: int = 15) -> go.Figure:
    """Promised versus actual delivery time by state."""
    d = promise.nlargest(top, "orders").sort_values("median_actual")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d.customer_state, x=d.median_actual, orientation="h",
                         name="Actual (median)", marker_color=C.PALETTE["primary"]))
    fig.add_trace(go.Bar(y=d.customer_state,
                         x=d.median_promised - d.median_actual, orientation="h",
                         name="Unused promise padding",
                         marker_color=C.PALETTE["grid"]))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Days from purchase")
    return style(fig, 460)


def fig_distance(basket: pd.DataFrame) -> go.Figure:
    """Two aligned panels rather than a dual-axis chart - see
    fig_growth_vs_satisfaction for why."""
    d = basket[basket.dimension == "Distance band"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Bar(x=d.level, y=d.avg_delivery_days, name="Delivery days",
                         marker_color=C.PALETTE["primary"], showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=d.level, y=d.avg_score, name="Average score",
                             mode="lines+markers", showlegend=False,
                             line=dict(color=C.PALETTE["accent"], width=3)),
                  row=2, col=1)
    fig.update_yaxes(title_text="Avg. delivery days", row=1, col=1)
    fig.update_yaxes(title_text="Avg. score", range=[3.6, 4.5], row=2, col=1)
    return style(fig, 440)


# --------------------------------------------------------------------------
# 3. Segments
# --------------------------------------------------------------------------
def fig_state_map(state: pd.DataFrame) -> go.Figure:
    """Satisfaction by customer state, positioned on Brazil.

    Uses point geography rather than a choropleth so the figure needs no
    external boundary file and stays reproducible offline.
    """
    centroids = {
        "AC": (-9.0, -70.5), "AL": (-9.6, -36.6), "AM": (-4.1, -63.0),
        "AP": (1.4, -51.8), "BA": (-12.6, -41.7), "CE": (-5.2, -39.3),
        "DF": (-15.8, -47.9), "ES": (-19.6, -40.3), "GO": (-16.0, -49.6),
        "MA": (-5.0, -45.3), "MG": (-18.5, -44.5), "MS": (-20.5, -54.5),
        "MT": (-13.0, -55.9), "PA": (-4.3, -52.5), "PB": (-7.2, -36.7),
        "PE": (-8.4, -37.9), "PI": (-7.7, -42.7), "PR": (-24.6, -51.5),
        "RJ": (-22.3, -42.7), "RN": (-5.8, -36.5), "RO": (-10.9, -63.0),
        "RR": (2.1, -61.4), "RS": (-30.0, -53.2), "SC": (-27.3, -50.5),
        "SE": (-10.6, -37.4), "SP": (-22.2, -48.7), "TO": (-10.2, -48.3),
    }
    d = state.copy()
    d["lat"] = d.customer_state.map(lambda s: centroids.get(s, (None, None))[0])
    d["lng"] = d.customer_state.map(lambda s: centroids.get(s, (None, None))[1])
    d = d.dropna(subset=["lat", "lng"])

    fig = px.scatter_geo(
        d, lat="lat", lon="lng", size="orders", color="avg_score",
        hover_name="customer_state",
        color_continuous_scale=C.DIVERGING, range_color=[3.7, 4.4],
        size_max=48,
        hover_data={"orders": ":,", "late_rate": ":.1%",
                    "avg_delivery_days": ":.1f", "lat": False, "lng": False},
    )
    fig.update_geos(scope="south america", showcountries=True,
                    countrycolor="#c9cfdb", showland=True, landcolor="#f6f7fa",
                    lataxis_range=[-34, 6], lonaxis_range=[-75, -33])
    fig.update_layout(coloraxis_colorbar_title="Avg score")
    return style(fig, 520)


def fig_state_bars(state: pd.DataFrame, top: int = 16) -> go.Figure:
    """Print-safe companion to the map.

    The interactive map needs boundary data fetched at render time, so the
    report and slides use this instead: same story, no network dependency.
    """
    d = state.nlargest(top, "orders").sort_values("avg_score")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(
        go.Bar(x=d.customer_state, y=d.avg_score, showlegend=False,
               marker=dict(color=d.late_rate, colorscale=[[0, C.PALETTE["good"]],
                                                          [0.5, C.PALETTE["warn"]],
                                                          [1, C.PALETTE["bad"]]],
                           colorbar=dict(title="Late rate", tickformat=".0%",
                                        len=0.45, y=1, yanchor="top")),
               hovertemplate="%{x}<br>%{y:.2f} stars<extra></extra>"),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=d.customer_state, y=d.avg_delivery_days, showlegend=False,
                   name="Average delivery days", mode="lines+markers",
                   line=dict(color=C.PALETTE["primary"], width=2)),
        row=2, col=1)
    fig.update_yaxes(title_text="Average review score", range=[3.5, 4.5],
                     row=1, col=1)
    fig.update_yaxes(title_text="Days to deliver", row=2, col=1)
    return style(fig, 460)


def fig_category_risk(category: pd.DataFrame, top: int = 18, labelled: int = 7) -> go.Figure:
    """Volume against satisfaction, sized by the goodwill at stake.

    The upper-right is a healthy franchise. The lower-right is the danger
    zone: large categories quietly generating detractors.

    Only the biggest bubbles get a permanent text label - with 18
    categories bunched in a narrow score band, labelling all of them
    collides into an unreadable pile of overlapping text. Everything else
    is still there on hover.
    """
    d = category.nlargest(top, "orders")
    labels = d.nlargest(labelled, "gmv_exposure").sort_values("orders")["main_category"]
    d = d.assign(label=d.main_category.where(d.main_category.isin(set(labels)), ""))
    fig = px.scatter(
        d, x="orders", y="avg_score", size="gmv_exposure", color="late_rate",
        hover_name="main_category", size_max=55,
        color_continuous_scale=["#2e8b6f", "#d9a441", "#c1462f"],
        labels={"orders": "Orders", "avg_score": "Average review score",
                "late_rate": "Late rate"},
        text="label",
    )
    # The labelled categories cluster tightly on both axes, so a single
    # fixed textposition stacks their names on top of one another.
    # Alternating above/below by rank (ordered left to right) keeps
    # neighbours apart instead.
    above_below = {cat: ("top center" if i % 2 == 0 else "bottom center")
                   for i, cat in enumerate(labels)}
    fig.update_traces(
        textposition=[above_below.get(c, "top center") for c in d.main_category],
        textfont=dict(size=10, color="#5c667c"))
    fig.add_hline(y=category.avg_score.mean(), line_dash="dot",
                  line_color=C.PALETTE["neutral"],
                  annotation_text="category average")
    return style(fig, 520)


def fig_lane_risk(lane: pd.DataFrame, top: int = 15) -> go.Figure:
    d = lane.nlargest(top, "gmv_exposure").sort_values("gmv_exposure")
    fig = go.Figure(go.Bar(
        y=d.lane, x=d.excess_detractors, orientation="h",
        marker_color=[C.PALETTE["bad"] if v > 0 else C.PALETTE["good"]
                      for v in d.excess_detractors],
        hovertemplate="%{y}<br>%{x:.0f} excess 1-2 star reviews<br>"
                      "late rate %{customdata:.1%}<extra></extra>",
        customdata=d.late_rate,
    ))
    fig.update_xaxes(title_text="Excess 1-2 star reviews versus a typical lane")
    return style(fig, 480)


# --------------------------------------------------------------------------
# 4. Sellers
# --------------------------------------------------------------------------
def fig_seller_quadrant(sc: pd.DataFrame, min_orders: int = 30) -> go.Figure:
    """The operational chart: what to do with each seller.

    One panel per verdict instead of one scatter carrying four categorical
    colours. Past three series, an all-pairs form like a scatter stops
    being legible - and with 600+ rated sellers, a single shared panel
    collapsed into a solid mass of overlapping bubbles that told the
    reader nothing beyond "there are a lot of sellers." Faceting keeps
    every seller visible on identical axes, so panels stay comparable.
    """
    d = sc[sc.orders >= min_orders].copy()
    order = ["Remove", "Coach: product quality", "Coach: fulfilment", "Invest"]
    colors = {"Invest": C.PALETTE["good"],
              "Coach: fulfilment": C.PALETTE["warn"],
              "Coach: product quality": C.PALETTE["accent"],
              "Remove": C.PALETTE["bad"]}
    counts = d.action.value_counts()
    max_orders = d.orders.max()

    fig = make_subplots(
        rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.03,
        subplot_titles=[f"{a} ({counts.get(a, 0):,})" for a in order],
    )
    for i, action in enumerate(order, start=1):
        sub = d[d.action == action]
        fig.add_trace(
            go.Scatter(
                x=sub.gmv, y=sub.avg_score, mode="markers", showlegend=False,
                marker=dict(
                    size=sub.orders, sizemode="area",
                    sizeref=2.0 * max_orders / (34.0 ** 2), sizemin=3,
                    color=colors[action], opacity=0.6,
                    line=dict(width=1, color=C.PALETTE["bg"]),
                ),
                text=sub.seller_id,
                customdata=sub[["orders", "late_rate", "top_category"]],
                hovertemplate="%{text}<br>R$ %{x:,.0f} GMV, %{y:.2f} stars"
                              "<br>%{customdata[0]:,} orders, "
                              "%{customdata[1]:.1%} late<br>"
                              "%{customdata[2]}<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.add_hline(y=C.SELLER_BAD_SCORE, line_dash="dash",
                      line_color=C.PALETTE["neutral"], row=1, col=i)
        fig.update_xaxes(type="log", title_text="GMV (log)", row=1, col=i)
    fig.update_xaxes(matches="x")
    fig.update_yaxes(title_text="Average review score", row=1, col=1)
    fig.update_annotations(font_size=13)
    return style(fig, 460)


def fig_action_treemap(sc: pd.DataFrame) -> go.Figure:
    d = sc.groupby("action", as_index=False).agg(
        gmv=("gmv", "sum"), sellers=("seller_id", "size"))
    fig = px.treemap(
        d, path=["action"], values="gmv", color="action",
        color_discrete_map={"Invest": C.PALETTE["good"],
                            "Coach: fulfilment": C.PALETTE["warn"],
                            "Coach: product quality": C.PALETTE["accent"],
                            "Remove": C.PALETTE["bad"],
                            "Monitor (low volume)": C.PALETTE["neutral"],
                            "(?)": "#ffffff"},
        custom_data=["sellers"],
    )
    fig.update_traces(
        texttemplate="%{label}<br>R$%{value:,.0f}<br>%{customdata[0]:,} sellers",
        hovertemplate="%{label}<br>R$%{value:,.0f}<extra></extra>")
    return style(fig, 380)


# --------------------------------------------------------------------------
# 5. Model and simulation
# --------------------------------------------------------------------------
def fig_importance(imp: pd.DataFrame, model_prefix: str, top: int = 12) -> go.Figure:
    d = imp[imp.model.str.startswith(model_prefix)].nlargest(top, "importance")
    d = d.sort_values("importance")
    fig = go.Figure(go.Bar(y=d.feature, x=d.importance, orientation="h",
                           error_x=dict(array=d["std"], color=C.PALETTE["neutral"]),
                           marker_color=C.PALETTE["primary"]))
    fig.update_xaxes(title_text="Drop in ROC AUC when the feature is shuffled")
    return style(fig, 420)


def fig_calibration(cal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cal.decile + 1, y=cal.actual * 100,
                         name="Actual detractor rate",
                         marker_color=C.PALETTE["primary"]))
    fig.add_trace(go.Scatter(x=cal.decile + 1, y=cal.predicted * 100,
                             name="Predicted", mode="lines+markers",
                             line=dict(color=C.PALETTE["accent"], width=3)))
    fig.update_xaxes(title_text="Predicted risk decile (10 = riskiest)")
    fig.update_yaxes(title_text="1-2 star rate", ticksuffix="%")
    return style(fig, 380)


def fig_scenario_compare(results: pd.DataFrame) -> go.Figure:
    """Goodwill saved against revenue given up, for each scenario."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=results.scenario, y=results.detractors_avoided,
                         name="1-2 star reviews avoided",
                         marker_color=C.PALETTE["good"]))
    fig.add_trace(go.Bar(x=results.scenario, y=-results.orders_lost,
                         name="Orders given up",
                         marker_color=C.PALETTE["bad"]))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="Orders")
    return style(fig, 400)


def fig_sim_curve(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=curve.centre, y=curve.rate * 100, mode="lines+markers",
        line=dict(color=C.PALETTE["bad"], width=3),
        marker=dict(size=6)))
    fig.add_vline(x=0, line_dash="dash", line_color=C.PALETTE["neutral"],
                  annotation_text="promised date")
    fig.update_xaxes(title_text="Days late (negative = early)")
    fig.update_yaxes(title_text="Probability of a 1-2 star review",
                     ticksuffix="%")
    return style(fig, 380)
