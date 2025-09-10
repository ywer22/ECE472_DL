import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import Basis_Model_param, BasisModel
from .plotting import compare_linear_models, plot_fit, base_fit
from .training import train


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    # Generate 50 data points given parameters
    data_generating_model = Basis_Model_param(
        mu=np_rng.integers(low=0, high=5, size=(settings.data.num_features)),
        sigma=np_rng.integers(low=0, high=5, size=(settings.data.num_features)),
        w=np_rng.integers(low=0, high=5, size=(settings.data.num_features)),
        b=0,
    )
    log.debug("Data generating model", model=data_generating_model)

    data = Data(
        model=data_generating_model,
        rng=np_rng,
        num_features=settings.data.num_features,
        num_basis_ftn=settings.data.num_basis_ftn,
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
    )

    model = BasisModel(
        rngs=nnx.Rngs(params=model_key), num_basis_ftn=settings.data.num_basis_ftn
    )
    log.debug("Initial model", model=model.model)

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train(model, optimizer, data, settings.training, np_rng)

    log.debug("Trained model", model=model.model)

    compare_linear_models(data.model, model.model)

    if settings.data.num_features == 1:
        plot_fit(model, data, settings.plotting)
        base_fit(
            model, data, settings.plotting, num_basis_ftn=settings.data.num_basis_ftn
        )

    else:
        log.info("Skipping plotting for multi-feature models.")
