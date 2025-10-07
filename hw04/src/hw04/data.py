from dataclasses import InitVar, dataclass, field

import structlog
import numpy as np
import jax.numpy as jnp
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100

log = structlog.get_logger()


@dataclass
class Data_CIFAR:
    """Handles loading and batching of the CIFAR-10 dataset."""

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
    use_CIFAR10: bool = True
    data_aug: object = None

    def __post_init__(self, rng: np.random.Generator):
        """Load CIFAR-10 or 100 dataset using TensorFlow Datasets."""

        if self.use_CIFAR10:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        # Ensure labels are 1D arrays
        y_train = np.squeeze(y_train).astype(np.int32)
        y_test = np.squeeze(y_test).astype(np.int32)

        # Split training data into training and validation sets
        index = np.arange(len(x_train))
        rng.shuffle(index)
        split_idx = int(len(x_train) * (1 - self.val_split))
        train_idx, val_idx = index[:split_idx], index[split_idx:]

        # Index for batching
        self.index = np.arange(len(train_idx))
        self.x_train, self.y_train = x_train[train_idx], y_train[train_idx]
        self.x_val, self.y_val = x_train[val_idx], y_train[val_idx]
        self.x_test, self.y_test = x_test, y_test

        log.info(
            "CIFAR dataset loaded",
            n_CIFAR10_train=index.shape[0],
            n_train=self.x_train.shape[0],
            n_val=self.x_val.shape[0],
            n_test=self.x_test.shape[0],
        )

    def get_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        training: bool = True,
        aug_key: jnp.ndarray = None,  # Add augmentation key
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        x_batch = self.x_train[choices].astype(np.float32) / 255.0
        y_batch = self.y_train[choices].flatten()

        # Apply data augmentation if provided
        if training and aug_key is not None and self.data_aug is not None:
            x_batch = self.data_aug(jnp.array(x_batch), key=aug_key, training=True)
            x_batch = np.array(x_batch)  # Convert back to numpy for consistency

        return x_batch, y_batch

    def get_val_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the entire validation dataset."""
        return self.x_val.astype(np.float32) / 255.0, self.y_val.flatten()

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the entire test dataset."""
        return self.x_test.astype(np.float32) / 255.0, self.y_test.flatten()

    def set_data_augmentation(self, data_aug):
        """Set the data augmentation module."""
        self.data_aug = data_aug
