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
    """Performs a single training step with gradient clipping."""

    def loss_fn(model: Classifier):
        logits = model(x, training=True)
        per_example = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        ce_loss = jnp.mean(per_example)
        l2_loss = model.l2_loss()
        total_loss = ce_loss + l2_loss
        return total_loss, (logits, per_example, ce_loss, l2_loss)

    (total_loss, (logits, per_example, ce_loss, l2_loss)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(model)

    # Apply gradient clipping to prevent explosion
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

    optimizer.update(model, grads)

    # Calculate training accuracy for monitoring
    preds = jnp.argmax(logits, axis=1)
    accuracy = jnp.mean(preds == y)

    return total_loss, ce_loss, l2_loss, accuracy


def evaluate_model(
    model: Classifier, data: Data_CIFAR, batch_size: int, dataset_type: str = "test"
) -> float:
    """Calculate classification accuracy on specified dataset."""
    if dataset_type == "test":
        x_np, y_np = data.get_test_data()
    elif dataset_type == "val":
        x_np, y_np = data.get_val_data()
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    total_correct = 0
    total_samples = 0

    for i in range(0, len(y_np), batch_size):
        batch_end = min(i + batch_size, len(y_np))
        x_batch = jnp.asarray(x_np[i:batch_end])
        y_batch = jnp.asarray(y_np[i:batch_end])

        logits = model(x_batch, training=False)
        pred = jnp.argmax(logits, axis=1)
        correct = jnp.sum(pred == y_batch)

        total_correct += int(correct)
        total_samples += len(y_batch)

    accuracy = float(total_correct) / total_samples
    log.info(
        f"{dataset_type.capitalize()} accuracy",
        correct=total_correct,
        total=total_samples,
        accuracy=f"{accuracy:.4f}",
    )
    return accuracy


def train(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data_CIFAR,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
    aug_key: jnp.ndarray = None,
):
    """Train the model with cosine decay learning rate scheduling."""
    log.info("Starting training", **settings.model_dump())

    # Create learning rate schedule
    schedule = optax.cosine_decay_schedule(
        init_value=settings.learning_rate,
        decay_steps=settings.num_iters,
    )

    bar = trange(settings.num_iters)

    for i in bar:
        current_lr = schedule(i)

        # Get training batch
        if aug_key is not None:
            current_aug_key, aug_key = jax.random.split(aug_key)
            x_np, y_np = data.get_batch(
                np_rng, settings.batch_size, training=True, aug_key=current_aug_key
            )
        else:
            x_np, y_np = data.get_batch(np_rng, settings.batch_size, training=True)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        # Training step
        total_loss, ce_loss, l2_loss, train_accuracy = train_step(
            model, optimizer, x, y
        )

        # Update progress bar only
        bar.set_description(
            f"Loss: {total_loss:.3f} (CE: {ce_loss:.3f}, L2: {l2_loss:.3f}) | "
            f"Acc: {train_accuracy:.3f} | LR: {current_lr:.5f}"
        )

    log.info("Training completed")

    # Final evaluations only
    val_accuracy = evaluate_model(model, data, settings.batch_size, "val")
    test_accuracy = evaluate_model(model, data, settings.batch_size, "test")

    log.info(
        "Final results",
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
    )

    return val_accuracy, test_accuracy
