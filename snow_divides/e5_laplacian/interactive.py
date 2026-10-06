from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray


def embedding_3d_html(
    output: Path,
    vectors: NDArray[np.float64],
    fiedler_labels: NDArray[np.int32],
    cluster_labels: NDArray[np.int32],
    selected: NDArray[np.int64],
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
) -> None:
    row, column = np.divmod(selected, longitude.size)
    coordinates = np.column_stack((latitude[row], longitude[column], selected, cluster_labels + 1))
    figure = go.Figure()
    trace_count = 0
    domains = (
        (0, "Fiedler q₂ < 0", "#31688e"),
        (1, "Fiedler q₂ ≈ 0", "#888888"),
        (2, "Fiedler q₂ > 0", "#d95f02"),
    )
    for label, name, color in domains:
        members = fiedler_labels == label
        if not np.any(members):
            continue
        figure.add_trace(
            go.Scatter3d(
                x=vectors[members, 1],
                y=vectors[members, 2],
                z=vectors[members, 3],
                mode="markers",
                name=name,
                customdata=coordinates[members],
                marker={"size": 2.5, "color": color, "opacity": 0.7},
                hovertemplate=(
                    "lat %{customdata[0]:.2f}°N, lon %{customdata[1]:.2f}°E"
                    "<br>original grid index %{customdata[2]:.0f}"
                    "<br>cluster %{customdata[3]:.0f}"
                    "<br>q₂ %{x:.5f}, q₃ %{y:.5f}, q₄ %{z:.5f}<extra></extra>"
                ),
            )
        )
        trace_count += 1
    fiedler_trace_count = trace_count
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    for label in np.unique(cluster_labels):
        members = cluster_labels == label
        figure.add_trace(
            go.Scatter3d(
                x=vectors[members, 1],
                y=vectors[members, 2],
                z=vectors[members, 3],
                mode="markers",
                name=f"Cluster {label + 1}",
                visible=False,
                customdata=coordinates[members],
                marker={"size": 2.5, "color": colors[int(label) % len(colors)], "opacity": 0.7},
                hovertemplate=(
                    "lat %{customdata[0]:.2f}°N, lon %{customdata[1]:.2f}°E"
                    "<br>original grid index %{customdata[2]:.0f}"
                    "<br>cluster %{customdata[3]:.0f}"
                    "<br>q₂ %{x:.5f}, q₃ %{y:.5f}, q₄ %{z:.5f}<extra></extra>"
                ),
            )
        )
        trace_count += 1
    total_traces = trace_count
    figure.update_layout(
        title="Laplacian coordinates q₂, q₃, q₄; colors show graph partitions",
        scene={
            "xaxis_title": "q₂ (not longitude)",
            "yaxis_title": "q₃ (not latitude)",
            "zaxis_title": "q₄ (not snow depth)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
        height=780,
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "x": 0.02,
                "y": 1.04,
                "buttons": [
                    {
                        "label": "Fiedler domains (q₂ sign)",
                        "method": "update",
                        "args": [
                            {
                                "visible": [True] * fiedler_trace_count
                                + [False] * (total_traces - fiedler_trace_count)
                            }
                        ],
                    },
                    {
                        "label": "Spectral clusters",
                        "method": "update",
                        "args": [
                            {
                                "visible": [False] * fiedler_trace_count
                                + [True] * (total_traces - fiedler_trace_count)
                            }
                        ],
                    },
                ],
            }
        ],
    )
    figure.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
