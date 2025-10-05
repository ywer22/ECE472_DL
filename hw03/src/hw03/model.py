import jax
import jax.numpy as jnp
import structlog
from flax import nnx

log = structlog.get_logger()


class Conv2d(nnx.Module):
    """A 2D convolutional layer wrapper."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int] = (3, 3),
        padding: str = "SAME",
        strides: tuple[int, int] = (1, 1),
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class Classifier_mnist(nnx.Module):
    """A simple CNN model for MNIST classification."""

    def __init__(
        self,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        num_classes: int,
        dropout: float,
        l2_reg: float,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes
        self.dropout = dropout
        self.l2_reg = l2_reg

        # Create convolutional layers
        self.conv_layers = []
        in_ch = input_depth

        for i, out_ch in enumerate(layer_depths):
            self.conv_layers.append(
                Conv2d(
                    in_features=in_ch,
                    out_features=out_ch,
                    kernel_size=layer_kernel_sizes[i],
                    rngs=rngs,
                )
            )
            in_ch = out_ch

        # Calculate dense layer input size, 28x28 to 7x7 after 2 poolings
        final_size = (
            7 * 7 * layer_depths[-1]
        )  # I should change this later to function for more flexibility

        self.dense = nnx.Linear(final_size, num_classes, rngs=rngs)

    def max_pool(
        self, x: jax.Array, window_size: int = 2, stride: int = 2
    ) -> jax.Array:
        """Max Pooling function to reduce overfitting and downsample for efficiency."""
        return jax.lax.reduce_window(
            x,
            -jnp.inf,  # Identity for max, -infinity
            jax.lax.max,
            window_dimensions=(1, window_size, window_size, 1),
            window_strides=(1, stride, stride, 1),
            padding="SAME",
        )

    def __call__(
        self, x: jax.Array, training: bool = False, rngs: nnx.Rngs = None
    ) -> jax.Array:
        """Forward pass of the classifier."""
        # Apply convolutional layers with ReLU and pooling
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = jax.nn.relu(x)
            x = self.max_pool(x, window_size=2, stride=2)

        # Flatten
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))

        # Apply dropout for training data
        if training:
            if rngs is not None:
                # Use dropout with RNGs
                x = nnx.Dropout(rate=self.dropout)(x, rngs=rngs)
            else:
                # If no RNGs provided, create a temporary one (for non-JIT cases)
                temp_rngs = nnx.Rngs(42)  # arbitrary seed
                x = nnx.Dropout(rate=self.dropout)(x, rngs=temp_rngs)

        x = self.dense(x)
        return x

    def l2_loss(self) -> jax.Array:
        """Calculate L2 regularization loss."""
        l2_loss = 0.0
        # For convolutional layers
        for conv_layer in self.conv_layers:
            if hasattr(conv_layer.conv, "kernel"):
                l2_loss += self.l2_reg * jnp.sum(conv_layer.conv.kernel**2)

        # For dense layer
        if hasattr(self.dense, "kernel"):
            l2_loss += self.l2_reg * jnp.sum(self.dense.kernel**2)

        return l2_loss
