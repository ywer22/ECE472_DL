import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import structlog


from .config import PlottingSettings
from .data import Data
from .model import NNXMLPModel
from sklearn.inspection import DecisionBoundaryDisplay

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_fit(
    model: NNXMLPModel,
    data: Data,
    settings: PlottingSettings,
):
    """Plots synthetic spiral data with decision boundary."""
    log.info("Spiral Data with decision boundary.")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    labels = data.y.flatten()
    # plot data points
    ax.scatter(data.x[:, 0], data.x[:, 1], c=labels, cmap="summer", edgecolor="k")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MLP Decision Boundary on Spiral Dataset")

    # mesh grid for decision boundary, add padding
    x_min, x_max = data.x[:, 0].min(), data.x[:, 0].max()
    y_min, y_max = data.x[:, 1].min(), data.x[:, 1].max()
    pad_x = (x_max - x_min) * 0.15
    pad_y = (y_max - y_min) * 0.15
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    grid_res = 1000
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res)
    )

    # Plot decision boundary
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    probs = model(grid)
    probs = probs.ravel().reshape(xx.shape)
    probs = (probs > 0.5).astype(int)
    display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=probs)
    display.plot(ax=ax, cmap="RdYlGn", alpha=0.35)

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "spiral.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
