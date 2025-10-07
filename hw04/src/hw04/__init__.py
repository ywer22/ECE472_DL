import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path
import structlog

from .data import Data_CIFAR
from .model import Classifier, Data_Augmentation
from .config import load_settings
from .logging import configure_logging
from .training import train, evaluate_model


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key, aug_key = jax.random.split(key, 3)
    np_rng = np.random.default_rng(np.array(data_key))

    # Data Augmentation
    data_aug = Data_Augmentation()

    # Create datasets
    cifar10_data = Data_CIFAR(
        model=None,
        rng=np_rng,
        batch_size=int(settings.data.batch_size),
        val_split=float(settings.data.val_split),
        use_CIFAR10=True,
    )

    cifar10_data.set_data_augmentation(data_aug)

    log.info(
        "CIFAR-10 dataset loaded",
        train_samples=cifar10_data.x_train.shape[0],
        val_samples=cifar10_data.x_val.shape[0],
        test_samples=cifar10_data.x_test.shape[0],
    )

    # Create models for datasets
    model_rngs = nnx.Rngs(params=model_key)

    # Model for CIFAR-10
    model_cifar10 = Classifier(
        num_classes=10,
        base_planes=32,
        block_counts=(3, 4, 6, 3),
        num_groups=8,
        expansion=4,
        rngs=model_rngs,
    )

    # Initialize models by calling them with sample data
    sample_batch = cifar10_data.x_train[:2].astype(np.float32) / 255.0
    sample_batch = jax.numpy.array(sample_batch)
    _ = model_cifar10(sample_batch, training=False)

    # Initialize optimizers
    optimizer_cifar10 = nnx.Optimizer(
        model_cifar10, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )
    log.info("Optimizers initialized")

    # Train the model
    train(
        model=model_cifar10,
        optimizer=optimizer_cifar10,
        data=cifar10_data,
        settings=settings.training,
        np_rng=np_rng,
    )

    # Create checkpoint directory
    ckpt_dir = Path("/tmp/my-checkpoints-5/")
    ckpt_dir.mkdir(exist_ok=True)

    # Split the model state
    dynamic_context, state = nnx.split(model_cifar10)

    # Save checkpoint
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "state", state)
    checkpointer.wait_until_finished()
    log.info("Checkpoint saved", directory=str(ckpt_dir))


def test() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key, aug_key = jax.random.split(key, 3)
    np_rng = np.random.default_rng(np.array(data_key))

    # Create datasets
    log.info("Creating dataset...")
    cifar10_data = Data_CIFAR(
        model=None,
        rng=np_rng,
        batch_size=int(settings.data.batch_size),
        val_split=float(settings.data.val_split),
        use_CIFAR10=True,
    )
    log.info(
        "CIFAR-10 dataset loaded",
        train_samples=cifar10_data.x_train.shape[0],
        val_samples=cifar10_data.x_val.shape[0],
        test_samples=cifar10_data.x_test.shape[0],
    )

    # Create models for both datasets
    model_rngs = nnx.Rngs(params=model_key)

    # Model for CIFAR-10
    model_cifar10 = Classifier(
        num_classes=10,
        base_planes=16,
        block_counts=(3, 4, 6, 3),
        num_groups=8,
        expansion=4,
        rngs=model_rngs,
    )

    # Initialize model
    sample_batch = cifar10_data.x_test[:2].astype(np.float32) / 255.0
    sample_batch = jax.numpy.array(sample_batch)
    _ = model_cifar10(sample_batch, training=False)

    # Load checkpoint
    ckpt_dir = Path("/tmp/my-checkpoints-5/")
    if (ckpt_dir / "state").exists():
        checkpointer = ocp.StandardCheckpointer()

        # Split current model state
        dynamic_context, state = nnx.split(model_cifar10)

        # Restore trained weights
        restored_state = checkpointer.restore(ckpt_dir / "state", state)

        # Merge restored state back into model
        model_cifar10 = nnx.merge(dynamic_context, restored_state)
        log.info("Checkpoint loaded", directory=str(ckpt_dir))

        # Evaluate on test set
        test_accuracy = evaluate_model(
            model=model_cifar10,
            data=cifar10_data,
            batch_size=settings.training.batch_size,
            dataset_type="test",
        )
        log.info("Test accuracy", accuracy=test_accuracy)
    else:
        log.error("No checkpoint found", directory=str(ckpt_dir))
