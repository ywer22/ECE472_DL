import jax
import jax.numpy as jnp
import structlog
from flax import nnx

log = structlog.get_logger()


class Block(nnx.Module):
    """
    Compute one instance of hidden layer, linear and activation.
    maps (batch, hidden_dim) to (batch, hidden_dim)
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        hidden_layer_width: int,
        hidden_activation=jax.nn.relu,
    ):
        # create linear mapping
        self.linear = nnx.Linear(hidden_layer_width, hidden_layer_width, rngs=rngs)
        self.hidden_activation = hidden_activation

    # runs the forward pass
    def __call__(self, x: jax.Array) -> jax.Array:
        # x@w + b, x is now (batch, hidden layer width)
        h = self.linear(x)
        return self.hidden_activation(h)


class NNXMLPModel(nnx.Module):
    """
    input linear (input_dim) to (hidden_layer_width)
    Block modules (hidden_layer_width) to (hidden_layer_width)
    output linear (hidden_layer_width) to (output_dim)
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        input_dim: int,
        output_dim: int,
        hidden_layer_width: int,
        num_hidden_layers: int,
        hidden_activation=jax.nn.relu,
        output_activation=jax.nn.sigmoid,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layer_width = int(hidden_layer_width)
        self.num_hidden_layers = int(num_hidden_layers)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # input layer
        self.input_linear = nnx.Linear(
            self.input_dim, self.hidden_layer_width, rngs=rngs
        )

        """hidden layers, tried to split the RNGs by indexing rngs_split but run into 
        error as using split on RngStream object. Couldn't figure out how to use @nnx.vmap
        for my written function... So used a list and loop instead.
        """
        self.hidden_layers = []
        for i in range(self.num_hidden_layers):
            # new Rngs object for each hidden layer
            layer_rngs = nnx.Rngs(rngs.params())
            self.hidden_layers.append(
                Block(
                    rngs=layer_rngs,
                    hidden_layer_width=self.hidden_layer_width,
                    hidden_activation=hidden_activation,
                )
            )

        # output layer
        self.output_linear = nnx.Linear(
            self.hidden_layer_width, self.output_dim, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        pass x, size (batch, input_dim)
        returns (batch,) if output layer, else (batch, 1)
        """
        # input layer, (batch, hidden_layer_width)
        h = self.input_linear(x)
        h = self.hidden_activation(h)

        # hidden layers
        for layer in self.hidden_layers:
            h = layer(h)

        # (batch, output_dim)
        logits = self.output_linear(h)
        logits = self.output_activation(logits)

        if logits.shape[-1] == 1:
            return jnp.ravel(logits)
        return logits
