from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def connectivity_scan(
    output: Path,
    k_values: NDArray[np.int32],
    components: NDArray[np.int32],
    fiedler_values: NDArray[np.float64],
) -> None:
    """Show component collapse and the connected-graph Fiedler trajectory across k."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(k_values, components, "o-", color="#31688e")
    axes[0].set(
        xlabel="Feature neighbors k",
        ylabel="Connected components",
        title="Connectivity of union kNN graphs",
    )
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    valid = np.isfinite(fiedler_values)
    axes[1].plot(k_values[valid], fiedler_values[valid], "o-", color="#b35c1e")
    axes[1].set(
        xlabel="Feature neighbors k",
        ylabel="Normalized Laplacian λ₂",
        title="Fiedler value after graph becomes connected",
    )
    axes[1].grid(alpha=0.25)
    fig.suptitle("E-3/E-5 k scan from the all-pairs snow-feature distance matrix")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
