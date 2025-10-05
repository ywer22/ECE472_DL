import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .data import Data
from .model import Classifier_mnist
from .config import load_settings
from .training import train_mnist
from .logging import configure_logging


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

    # Load data
    data = Data(
        model=None,
        rng=np_rng,
        batch_size=int(settings.data.batch_size),
        val_split=float(settings.data.val_split),
    )
    log.debug("Data loaded")

    # Create model and optimizer
    model_rngs = nnx.Rngs(params=model_key)

    model = Classifier_mnist(
        input_depth=int(settings.model.input_depth),
        layer_depths=list(settings.model.layer_depths),
        layer_kernel_sizes=list(
            settings.model.layer_kernel_sizes
        ),  # I casted and change config.py for this to work but I don't think this is good practice, is there a better way?
        num_classes=int(settings.model.num_classes),
        dropout=float(settings.model.dropout),
        l2_reg=float(settings.model.l2_reg),
        rngs=model_rngs,
    )
    log.debug("Model created")

    # Initialize optimizer
    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    # Train the model
    train_mnist(model, optimizer, data, settings.training, np_rng)
    log.info("Training completed")
