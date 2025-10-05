import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import NNXMLPModel
from .plotting import plot_fit
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

    data = Data(
        model=None,
        rng=np_rng,
        num_features=settings.data.num_features,
        sigma_noise=settings.data.sigma_noise,
        num_samples=settings.data.num_samples,
    )
    log.debug("Generating data points", model=data)

    # create the MLP model using NNX
    rngs = nnx.Rngs(settings.random_seed)
    model = NNXMLPModel(
        rngs=rngs,
        input_dim=int(settings.model.input_dim),
        output_dim=int(settings.model.output_dim),
        hidden_layer_width=int(settings.model.hidden_layer_width),
        num_hidden_layers=int(settings.model.num_hidden_layers),
        hidden_activation=jax.nn.relu,
        output_activation=jax.nn.sigmoid,
    )
    log.debug("Initial model created", model=model)

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train(model, optimizer, data, settings.training, np_rng)
    log.info("finish training")

    # plot
    plot_fit(model, data, settings.plotting)
