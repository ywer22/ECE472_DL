import jax

import numpy as np

# import optax
# from flax import nnx
import structlog

from .data import Data_CIFAR10, Data_CIFAR100

# from .model import
from .config import load_settings

# from .training import
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

    data = Data_CIFAR10(
        model=None,
        rng=np_rng,
        batch_size=int(settings.data.batch_size),
        val_split=float(settings.data.val_split),
    )
    log.debug("Generating CIFAR-10 data points", model=data)

    data2 = Data_CIFAR100(
        model=None,
        rng=np_rng,
        batch_size=int(settings.data.batch_size),
        val_split=float(settings.data.val_split),
    )
    log.debug("Generating CIFAR-100 data points", model=data2)
