import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import structlog

from .config import PlottingSettings
from .data import Data
from .model import Basis_Model_param, BasisModel

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def compare_linear_models(a: Basis_Model_param, b: Basis_Model_param):
    """Prints a comparison of two linear models."""
    log.info("Comparing models", true=a, estimated=b)
    print("w,    w_hat")
    for w_a, w_b in zip(a.w, b.w):
        print(f"{w_a:0.2f}, {w_b:0.2f}")

    print(f"{a.b:0.2f}, {b.b:0.2f}")


def plot_fit(
    model: BasisModel,
    data: Data,
    settings: PlottingSettings,
):
    """Plots the linear fit and saves it to a file."""
    log.info("Plotting fit")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("Linear fit")
    ax.set_xlabel("x")
    ax.set_ylim(-np.amax(data.y) * 1.5, np.amax(data.y) * 1.5)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 100)
    xs = xs[:, np.newaxis]
    ax.plot(xs, np.squeeze(model(jnp.asarray(xs))), "-", label="Estimated function")
    ax.plot(np.squeeze(data.x), data.y, "o")
    ax.plot(xs, np.sin(2 * np.pi * xs), "--", label="Sine wave")

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw01_fit.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))


def base_fit(
    model: BasisModel,
    data: Data,
    settings: PlottingSettings,
    num_basis_ftn: int,
):
    """Plots the basis ftn and saves it to a file."""
    log.info("Plotting fit")
    fig2, ax2 = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax2.set_title("Base for fit")
    ax2.set_xlabel("x")
    ax2.set_ylim(0, np.amax(data.y))
    h = ax2.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    x = np.linspace(-1, 1, 100)

    params = model.model_params
    for i in range(num_basis_ftn):
        ax2.plot(x, jnp.exp(-((x - params.mu[i]) ** 2) / (params.sigma[i] ** 2)))

    plt.tight_layout()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw01_basis.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
