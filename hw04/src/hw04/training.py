import jax.numpy as jnp
import jax
from flax import nnx
import numpy as np
import optax
import structlog

from .config import TrainingSettings
from .data import Data_CIFAR
from .model import Classifier
from tqdm import trange

log = structlog.get_logger()


@nnx.jit
def train_step(
    model: Classifier, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
):
    """Performs a single training step."""

    def loss_fn(model: Classifier):
        logits = model(x, training=True)
        ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        l2_loss = model.l2_loss()
        total_loss = ce_loss + l2_loss

        return total_loss, (ce_loss, l2_loss)

    (total_loss, (ce_loss, l2_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model
    )
    optimizer.update(model, grads)
    return total_loss, ce_loss, l2_loss


def compute_accuracy(
    model: Classifier,
    data: Data_CIFAR,
    batch_size: int,
    validation: bool = True,
) -> float:
    """Calculate classification accuracy."""
    if validation:
        x_np, y_np = data.get_val_data()
    else:
        x_np, y_np = data.get_test_data()

    total_correct = 0
    total_samples = 0

    for i in range(0, len(y_np), batch_size):
        batch_end = min(i + batch_size, len(y_np))
        x_batch = jnp.asarray(x_np[i:batch_end])
        y_batch = jnp.asarray(y_np[i:batch_end])

        logits = model(x_batch, training=False)
        pred = jnp.argmax(logits, axis=1)
        correct = jnp.sum(pred == y_batch)

        total_correct += correct
        total_samples += len(y_batch)

    accuracy = float(total_correct) / total_samples
    log.info(
        "Accuracy computed",
        correct=total_correct,
        total=total_samples,
        accuracy=accuracy,
    )
    return accuracy


def train(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data_CIFAR,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
    aug_key: jnp.ndarray = None,
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)

    for i in bar:
        if aug_key is not None:
            current_aug_key, aug_key = jax.random.split(aug_key)
            x_np, y_np = data.get_batch(
                np_rng, settings.batch_size, training=True, aug_key=current_aug_key
            )
        else:
            x_np, y_np = data.get_batch(np_rng, settings.batch_size, training=True)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        total_loss, ce_loss, l2_loss = train_step(model, optimizer, x, y)

        if i % 100 == 0:
            bar.set_description(
                f"Loss @ {i} => Total: {total_loss:.4f}, CE: {ce_loss:.4f}, L2: {l2_loss:.4f}"
            )
            log.debug(
                "Training progress",
                iteration=i,
                total_loss=float(total_loss),
                ce_loss=float(ce_loss),
                l2_loss=float(l2_loss),
            )
            bar.refresh()

    log.info("Training finished")

    # Compute final accuracy
    val_accuracy = compute_accuracy(model, data, settings.batch_size, validation=True)
    log.info("Final validation accuracy", accuracy=val_accuracy)
