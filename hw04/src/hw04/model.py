import jax
import jax.numpy as jnp
import structlog
import dm_pix as pix
from flax import nnx

log = structlog.get_logger()


class Conv2d(nnx.Module):
    """A 2D convolutional layer wrapper with L2 regularization."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        stride,
        l2reg: float,
        padding: str = "SAME",
        rngs: nnx.Rngs = None,
    ):
        self.l2reg = l2reg
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=(stride, stride),
            padding=padding,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)

    def l2_loss(self):
        return self.l2reg * jnp.sum(jnp.square(self.conv.kernel))


class Data_Augmentation(nnx.Module):
    """Data Augmentation layer in charge of upscaling, rotation, flip, and added noise."""

    def __init__(self, pad_size: int = 4):
        super().__init__()
        self.pad_size = pad_size

    def __call__(
        self, x: jax.Array, key: jax.Array, training: bool = True
    ) -> jax.Array:
        # Only apply to training set
        if not training or key is None:
            return x

        # Split key for different augmentations
        keys = jax.random.split(key, 3)

        # augmentation testing data
        x = self.random_pad_and_crop(x, keys[0])
        x = self.flip_img(x, keys[1])
        x = self.gaussian_noise(x, keys[2], std=0.05)

        return x

    def random_pad_and_crop(self, img: jax.Array, key: jax.Array) -> jax.Array:
        """Random padding and cropping as shown in PDF."""
        B, H, W, C = img.shape
        pad = self.pad_size

        # Pad the image
        padded_img = jnp.pad(
            img, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode="reflect"
        )

        def crop_single(padded, crop_key):
            # Random crop position
            start_h = jax.random.randint(crop_key, (), 0, 2 * pad + 1)
            start_w = jax.random.randint(crop_key, (), 0, 2 * pad + 1)

            cropped = jax.lax.dynamic_slice(padded, (start_h, start_w, 0), (H, W, C))
            return cropped

        # Split key for each image's crop
        crop_keys = jax.random.split(key, B)
        return jax.vmap(crop_single)(padded_img, crop_keys)

    def flip_img(self, img: jax.Array, key: jax.Array) -> jax.Array:
        """Random horizontal flip using dm_pix."""
        B, H, W, C = img.shape
        flip = jax.random.bernoulli(key, 0.5, (B,))

        def flip_single(img, should_flip):
            return jax.lax.cond(
                should_flip, lambda: pix.flip_left_right(img), lambda: img
            )

        return jax.vmap(flip_single)(img, flip)

    def gaussian_noise(
        self, x: jax.Array, key: jax.Array, std: float = 0.05
    ) -> jax.Array:
        """Blur the image using dm_pix by apply gaussian noise."""
        B, H, W, C = x.shape
        noise = jax.random.normal(key, (B, H, W, C)) * std
        return jnp.clip(x + noise, 0.0, 1.0)


class GroupNorm(nnx.Module):
    """Group Normalization layer."""

    def __init__(
        self,
        num_groups: int,
        num_features: int,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.gn = nnx.GroupNorm(
            num_groups=num_groups,
            num_features=num_features,
            epsilon=eps,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.gn(x)


class ResidualBlock(nnx.Module):
    """Residual block similar to identity mapping that uses pre-activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_groups: int = 8,
        l2reg: float = 0.001,
        kernel_size: tuple[int, int] = (3, 3),
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_groups = num_groups
        self.l2reg = l2reg
        self.kernel_size = kernel_size

        # First convolution block BN -> Activation -> Conv
        self.gn1 = GroupNorm(num_groups=num_groups, num_features=in_channels, rngs=rngs)
        self.prelu1 = nnx.PReLU()
        self.conv1 = Conv2d(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            l2reg=l2reg,
            padding="SAME",
            rngs=rngs,
        )

        # Second convolution block BN -> Activation -> Conv
        self.gn2 = GroupNorm(
            num_groups=num_groups,
            num_features=out_channels,
            rngs=rngs,
        )
        self.prelu2 = nnx.PReLU()
        self.conv2 = Conv2d(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            l2reg=l2reg,
            rngs=rngs,
        )

        # Shortcut connection, apply kernel size (1, 1) to add together
        if stride != 1 or in_channels != self.out_channels:
            self.shortcut_conv = Conv2d(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                stride=stride,
                l2reg=l2reg,
                rngs=rngs,
            )
        else:
            self.shortcut_conv = None

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x

        # First Block
        out = self.gn1(x)
        out = self.prelu1(out)
        out = self.conv1(out)

        # Second block
        out = self.gn2(out)
        out = self.prelu2(out)
        out = self.conv2(out)

        # Shortcut connection
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)

        return out + residual


class Classifier(nnx.Module):
    """A ResNet model for CIFAR classification with L2 regularization."""

    def __init__(
        self,
        num_classes: int,
        base_planes: int = 32,
        block_counts: tuple = (3, 4, 6, 3),
        num_groups: int = 8,
        l2reg: float = 0.001,
        kernel_size: tuple[int, int] = (3, 3),
        strides: list[int] = None,
        rngs: nnx.Rngs = None,
    ):
        self.num_classes = num_classes
        self.base_planes = base_planes
        self.block_counts = block_counts
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.l2reg = l2reg

        # strides for each stage
        if strides is None:
            strides = [1, 2, 2, 2]
        self.strides = strides

        # Initial convolution, 3 input channels
        self.conv1 = Conv2d(
            in_features=3,
            out_features=base_planes,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            l2reg=l2reg,
            rngs=rngs,
        )

        # Go through all the Residual layers
        self.stages = nnx.List()
        # 32, 64, 128, 256
        planes_list = [base_planes, base_planes * 2, base_planes * 4, base_planes * 8]
        current_channels = base_planes

        for i, (planes, num_blocks, stride) in enumerate(
            zip(planes_list, block_counts, strides)
        ):
            stage_blocks = nnx.List()
            for j in range(num_blocks):
                # First block in stage may downsample
                block_stride = stride if j == 0 else 1

                block = ResidualBlock(
                    in_channels=current_channels,
                    out_channels=planes,
                    stride=block_stride,
                    num_groups=num_groups,
                    l2reg=l2reg,
                    kernel_size=kernel_size,
                    rngs=rngs,
                )
                stage_blocks.append(block)
                current_channels = planes

            self.stages.append(stage_blocks)

        # Final layers
        self.gn_final = GroupNorm(
            num_groups=num_groups, num_features=current_channels, rngs=rngs
        )
        self.prelu_final = nnx.PReLU()
        self.dense = nnx.Linear(
            in_features=current_channels, out_features=num_classes, rngs=rngs
        )

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        x = self.conv1(x)

        for stage in self.stages:
            for block in stage:
                x = block(x)

        x = self.gn_final(x)
        x = self.prelu_final(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Classification
        logits = self.dense(x)
        return logits

    def l2_loss(self) -> jax.Array:
        """Calculate total L2 loss for all Conv2d layers in the model."""
        total_loss = jnp.array(0.0)
        num_layers = 0

        # Initial conv
        total_loss += self.conv1.l2_loss()
        num_layers += 1

        # Residual blocks conv
        for stage in self.stages:
            for block in stage:
                total_loss += block.conv1.l2_loss()
                total_loss += block.conv2.l2_loss()
                num_layers += 2
                if block.shortcut_conv is not None:
                    total_loss += block.shortcut_conv.l2_loss()
                    num_layers += 1

        # Average across layers to prevent scaling issues
        return total_loss / max(1, num_layers)
