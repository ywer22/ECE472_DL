from dataclasses import InitVar, dataclass, field

import numpy as np

from .model import Basis_Model_param


@dataclass
class Data:
    """Handles generation of synthetic data for linear regression."""

    model: Basis_Model_param
    rng: InitVar[np.random.Generator]
    num_features: int
    num_basis_ftn: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate synthetic data based on the model."""
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0, 1, size=(self.num_samples, self.num_features))
        clean_y = np.sin(2 * np.pi * self.x) + self.model.b  # bias=sigma gauss noise
        self.y = rng.normal(loc=clean_y, scale=self.sigma)

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
