from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class Basis_Model_param:
    """Parameters for gaussian basis function."""

    mu: np.ndarray
    sigma: np.ndarray
    w: np.ndarray
    b: float


class BasisModel(nnx.Module):
    """A Flax NNX module for a basis model."""

    def __init__(self, *, rngs: nnx.Rngs, num_basis_ftn: int):
        self.num_basis_ftn = num_basis_ftn
        key = rngs.params()
        self.mu = nnx.Param(jax.random.uniform(key, (self.num_basis_ftn, 1)))
        self.sigma = nnx.Param(jax.random.normal(key, (self.num_basis_ftn, 1)))
        self.w = nnx.Param(
            jax.random.normal(key, (self.num_basis_ftn, 1)) * 0.2
        )  # initial weight is less using *0.2
        self.b = nnx.Param(jnp.zeros((1, 1)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        # compute gaussian basis ftn
        gauss_basis = jnp.exp(-((x - self.mu.value.T) ** 2) / self.sigma.value.T**2)

        return jnp.squeeze(gauss_basis @ self.w.value + self.b.value)

    @property
    def model(self) -> Basis_Model_param:
        """Returns the model parameters."""
        return Basis_Model_param(
            mu=np.array(self.mu.value).reshape([self.num_basis_ftn]),
            sigma=np.array(self.sigma.value).reshape([self.num_basis_ftn]),
            w=np.array(self.w.value).reshape([self.num_basis_ftn]),
            b=np.array(self.b.value).squeeze(),
        )

    @property
    def model_params(self) -> Basis_Model_param:
        """Return the model parameters."""
        return Basis_Model_param(
            mu=np.array(self.mu.value).reshape([self.num_basis_ftn]),
            sigma=np.array(self.sigma.value).reshape([self.num_basis_ftn]),
            w=np.array(self.w.value).reshape([self.num_basis_ftn]),
            b=np.array(self.b.value).squeeze(),
        )
