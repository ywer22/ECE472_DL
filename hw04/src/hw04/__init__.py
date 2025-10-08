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
    data_aug = Data_Augmentation(pad_size=4)

    # Create datasets, bool true to use CIFAR10
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
    )

    # Model for CIFAR-10
    model_rngs = nnx.Rngs(params=model_key)
    model_cifar10 = Classifier(
        num_classes=10,
        base_planes=settings.model.base_planes,
        block_counts=tuple(settings.model.block_counts),
        num_groups=settings.model.num_groups,
        l2reg=settings.model.l2reg,
        kernel_size=tuple(settings.model.kernel_size),
        strides=settings.model.strides,
        rngs=model_rngs,
    )

    # Initialize model param with sample batch (?)
    sample_batch = cifar10_data.x_train[:2].astype(np.float32) / 255.0
    sample_batch = jax.numpy.array(sample_batch)
    _ = model_cifar10(sample_batch, training=False)
    logits = model_cifar10(sample_batch, training=False)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: {logits.min():.3f} to {logits.max():.3f}")

    schedule = optax.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=settings.training.num_iters,
        alpha=0.01,
    )

    # Initialize optimizers
    optimizer_cifar10 = nnx.Optimizer(
        model_cifar10,
        optax.sgd(learning_rate=schedule, momentum=settings.training.momentum),
        wrt=nnx.Param,
    )
    log.info("Optimizers initialized")

    # Train the model
    train(
        model=model_cifar10,
        optimizer=optimizer_cifar10,
        data=cifar10_data,
        settings=settings.training,
        np_rng=np_rng,
        aug_key=aug_key,
    )

    # Create checkpoint directory
    ckpt_dir = Path("/tmp/my-checkpoints-cifar10-6/")
    ckpt_dir.mkdir(exist_ok=True)

    # Split and Save checkpoint
    dynamic_context, state = nnx.split(model_cifar10)
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

    # Create models
    model_rngs = nnx.Rngs(params=model_key)
    model_cifar10 = Classifier(
        num_classes=10,
        base_planes=settings.model.base_planes,
        block_counts=tuple(settings.model.block_counts),
        num_groups=settings.model.num_groups,
        l2reg=settings.model.l2reg,
        kernel_size=tuple(settings.model.kernel_size),
        strides=settings.model.strides,
        rngs=model_rngs,
    )

    # Initialize model params (??)
    sample_batch = cifar10_data.x_test[:2].astype(np.float32) / 255.0
    sample_batch = jax.numpy.array(sample_batch)
    _ = model_cifar10(sample_batch, training=False)

    # Load checkpoint
    ckpt_dir = Path("/tmp/my-checkpoints-cifar10-6/")
    if (ckpt_dir / "state").exists():
        checkpointer = ocp.StandardCheckpointer()
        dynamic_context, state = nnx.split(model_cifar10)
        restored_state = checkpointer.restore(ckpt_dir / "state", state)
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
