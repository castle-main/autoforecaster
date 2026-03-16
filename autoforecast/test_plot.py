"""Interactive scatter plot: model probability vs market price for live testing."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from .types import PROJECT_ROOT


def generate_test_plot(
    results_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Generate an interactive HTML scatter plot from test results.

    X-axis: market price, Y-axis: model probability (toggle raw/calibrated).
    Hover shows question details, click opens exchange URL.
    """
    results_path = results_path or PROJECT_ROOT / "logs" / "test_results.json"
    output_path = output_path or PROJECT_ROOT / "logs" / "test_calibration.html"

    with open(results_path) as f:
        results = json.load(f)

    # Separate by source
    sources = {}
    for r in results:
        src = r["source"]
        sources.setdefault(src, []).append(r)

    # Color mapping
    colors = {"polymarket": "#6366f1", "kalshi": "#f59e0b"}

    fig = go.Figure()

    # For each source, add raw and calibrated traces
    for source, items in sources.items():
        color = colors.get(source, "#888888")
        market = [r["market_price"] for r in items]
        raw = [r["raw_probability"] for r in items]
        calibrated = [r["calibrated_probability"] for r in items]
        titles = [r["title"] for r in items]
        urls = [r["url"] for r in items]
        hover = [
            f"<b>{r['title'][:80]}</b><br>"
            f"Market: {r['market_price']:.1%}<br>"
            f"Raw: {r['raw_probability']:.1%}<br>"
            f"Calibrated: {r['calibrated_probability']:.1%}<br>"
            f"Source: {r['source']}"
            if r["calibrated_probability"] is not None
            else f"<b>{r['title'][:80]}</b><br>"
            f"Market: {r['market_price']:.1%}<br>"
            f"Raw: {r['raw_probability']:.1%}<br>"
            f"Source: {r['source']}"
            for r in items
        ]

        # Raw probability trace (visible by default)
        fig.add_trace(go.Scatter(
            x=market,
            y=raw,
            mode="markers",
            name=f"{source} (raw)",
            marker=dict(color=color, size=8, opacity=0.7),
            text=titles,
            customdata=urls,
            hovertext=hover,
            hoverinfo="text",
            visible=True,
        ))

        # Calibrated probability trace (hidden by default)
        cal_y = [c if c is not None else r for c, r in zip(calibrated, raw)]
        fig.add_trace(go.Scatter(
            x=market,
            y=cal_y,
            mode="markers",
            name=f"{source} (calibrated)",
            marker=dict(color=color, size=8, opacity=0.7, symbol="diamond"),
            text=titles,
            customdata=urls,
            hovertext=hover,
            hoverinfo="text",
            visible=False,
        ))

    # y=x reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", width=1, dash="dot"),
        name="Perfect calibration",
        hoverinfo="skip",
        showlegend=False,
    ))

    n_sources = len(sources)

    # Toggle buttons: raw vs calibrated
    # Raw: show odd-indexed traces (0, 2, ...) per source pair, hide even
    # Calibrated: show even-indexed (1, 3, ...), hide odd
    raw_visible = []
    cal_visible = []
    for _ in sources:
        raw_visible.extend([True, False])   # raw visible, calibrated hidden
        cal_visible.extend([False, True])    # raw hidden, calibrated visible
    # Reference line always visible
    raw_visible.append(True)
    cal_visible.append(True)

    fig.update_layout(
        title="Model vs Market — Live Testing",
        xaxis_title="Market Price",
        yaxis_title="Model Probability",
        xaxis=dict(range=[-0.02, 1.02], dtick=0.1),
        yaxis=dict(range=[-0.02, 1.02], dtick=0.1),
        width=900,
        height=700,
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5,
                xanchor="center",
                y=1.12,
                buttons=[
                    dict(
                        label="Raw Probability",
                        method="update",
                        args=[{"visible": raw_visible}],
                    ),
                    dict(
                        label="Calibrated Probability",
                        method="update",
                        args=[{"visible": cal_visible}],
                    ),
                ],
            )
        ],
    )

    # JavaScript to open URL on click — runs inside the plotly div script
    # {plot_id} is replaced by plotly with the actual div id
    click_js = """
    var plot = document.getElementById('{plot_id}');
    plot.on('plotly_click', function(data) {
        var point = data.points[0];
        if (point && point.customdata) {
            window.open(point.customdata, '_blank');
        }
    });
    """

    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        post_script=click_js,
    )

    return output_path
