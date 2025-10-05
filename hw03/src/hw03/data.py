from dataclasses import InitVar, dataclass, field

import structlog
import numpy as np
import tensorflow as tf

log = structlog.get_logger()


@dataclass
class Data:
    """Handles loading and batching of the MNIST dataset."""

    model: object
    rng: InitVar[np.random.Generator]
    batch_size: int
    val_split: float
    x_train: np.ndarray = field(init=False)
    y_train: np.ndarray = field(init=False)
    x_val: np.ndarray = field(init=False)
    y_val: np.ndarray = field(init=False)
    x_test: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Load MNIST dataset using TensorFlow Datasets."""

        (x_train, y_train), (self.x_test, self.y_test) = (
            tf.keras.datasets.mnist.load_data()
        )

        # Normalize and reshape to (N, 28, 28), add channel dimension so (N, 28, 28, 1)
        x_train = x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0
        x_train = np.expand_dims(x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)

        # Split training data into training and validation sets
        index = np.arange(len(x_train))
        rng.shuffle(index)
        split_idx = int(len(x_train) * (1 - self.val_split))
        train_idx, val_idx = index[:split_idx], index[split_idx:]

        # Index for batching
        self.index = np.arange(len(train_idx))
        self.x_train, self.y_train = x_train[train_idx], y_train[train_idx]
        self.x_val, self.y_val = x_train[val_idx], y_train[val_idx]

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x_train[choices], self.y_train[choices].flatten()

    def get_val_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the entire validation dataset."""
        return self.x_val, self.y_val.flatten()

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the entire test dataset."""
        return self.x_test, self.y_test.flatten()
