from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class Data:
    """Handles generation of spiral synthetic data for MLP."""

    model: object
    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma_noise: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate synthetic data(spiral) based on the model.
        Uses Archimedean Spiral Formula and Desmos for formula
        r=a+b*theta and r=a-b*theta, a=0, b=0.9, theta=[0, 3.5*pi].
        Generates data in polar coordinate, then converted to cartesian.
        """
        self.index = np.arange(self.num_samples)
        a = 0
        b = 0.9
        theta = np.linspace(0, 7 * np.pi, self.num_samples)

        r1 = a + b * theta
        r2 = a - b * theta

        # Split samples into 2 arms of the spiral
        half_data = self.num_samples // 2
        clean_r = np.concatenate([r1[:half_data], r2[: (self.num_samples - half_data)]])
        clean_theta = np.concatenate(
            [theta[:half_data], theta[: (self.num_samples - half_data)]]
        )

        # cartesian and [N, 2] data points
        x_clean = clean_r * np.cos(clean_theta)
        y_clean = clean_r * np.sin(clean_theta)

        self.x = np.stack([x_clean, y_clean], axis=1)
        self.x += rng.normal(loc=0.0, scale=float(self.sigma_noise), size=self.x.shape)

        # label into group 0 and 1
        self.y = np.concatenate(
            [np.zeros(half_data), np.ones(self.num_samples - half_data)]
        )

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
