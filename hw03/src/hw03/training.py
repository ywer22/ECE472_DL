import jax
import jax.numpy as jnp
import numpy as np
import structlog
import optax
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import Classifier_mnist as Classifier

log = structlog.get_logger()


def calculate_accuracy(
    model: Classifier, data: Data, batch_size: int, validation_set: bool = True
) -> float:
    """Calculate classification accuracy."""
    if validation_set:
        x_np, y_np = data.get_val_data()
    else:
        x_np, y_np = data.get_test_data()

    n = len(y_np)
    correct = 0
    total = 0

    for i in range(0, n, batch_size):
        x_batch = jnp.asarray(x_np[i : i + batch_size])
        y_batch = jnp.asarray(y_np[i : i + batch_size])

        logits = model(x_batch, training=False)
        predictions = jnp.argmax(logits, axis=1)

        equal = predictions == y_batch
        total += len(equal)
        correct += jnp.sum(equal)

    accuracy = float(correct / total)
    log.info(
        "Accuracy calculated",
        correct=int(correct),
        total=int(total),
        accuracy=accuracy,
        dataset="validation" if validation_set else "test",
    )

    return accuracy


def train_step(
    model: Classifier,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
    dropout_key: jax.Array,
) -> float:
    """Single training step with cross-entropy loss and L2 regularization."""

    def loss_fn(model: Classifier):
        # Create RNGs from the key for this specific call
        rngs = nnx.Rngs(dropout=dropout_key)
        logits = model(x, training=True, rngs=rngs)
        loss_ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        loss_ce = jnp.mean(loss_ce)
        l2_loss = model.l2_loss()
        return loss_ce + l2_loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Update optimizer with both model and grads (new API)
    optimizer.update(model, grads)

    return loss


# JIT compile the training step
train_step_jitted = nnx.jit(train_step)


def train_mnist(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> float:
    """Train the model and return final accuracy."""
    log.info("Starting training", **settings.model_dump())

    # Convert numpy RNG to JAX RNG
    key = jax.random.PRNGKey(np_rng.integers(0, 2**32))

    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        # Split key for this training step
        key, dropout_key = jax.random.split(key)

        # Use JIT-compiled training step
        loss = train_step_jitted(model, optimizer, x, y, dropout_key)

        if i % 100 == 0:
            bar.set_description(f"Loss @ {i} => {loss:.6f}")

    log.info("Training finished")

    # Evaluate on validation set
    val_accuracy = calculate_accuracy(
        model, data, settings.batch_size, validation_set=True
    )

    if val_accuracy > 0.955:
        test_accuracy = calculate_accuracy(
            model, data, settings.batch_size, validation_set=False
        )
        log.info("Final test result:", accuracy=test_accuracy)
        return test_accuracy
    else:
        log.info("Validation accuracy too low, skipping test set")
        return val_accuracy
