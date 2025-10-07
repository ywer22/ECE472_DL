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
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int] = (3, 3),
        strides: tuple[int, int] = (1, 1),
        padding: str = "SAME",
        l2reg: float = 0.001,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.l2reg = l2reg

        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)

    def l2_loss(self) -> jax.Array:
        """Calculate L2 loss for this layer's weights."""
        if self.l2reg == 0.0:
            return jnp.array(0.0)
        return self.l2reg * jnp.sum(jnp.square(self.conv.kernel))


class Data_Augmentation(nnx.Module):
    """Data Augmentation layer in charge of upscaling, rotation, and contrast."""

    def __init__(self):
        super().__init__()

    def __call__(
        self, x: jax.Array, key: jax.Array, training: bool = True
    ) -> jax.Array:
        # Only apply to training set
        if not training or key is None:
            return x

        # Split key for different augmentations
        keys = jax.random.split(key, 3)

        # augmentation testing data
        x = self.upscaling_img(x, keys[0], scale_range=(1.1, 1.3))
        x = self.img_rotation(x, keys[1], rotation_angle=3.0)
        x = self.flip_img(x, keys[2])

        return x

    def upscaling_img(
        self,
        img: jax.Array,
        key: jax.Array,
        scale_range: tuple[float, float] = (1.1, 1.3),
    ) -> jax.Array:
        """Upscaling image on training set."""
        B, H, W, C = img.shape
        scale = jax.random.uniform(
            key, (B,), minval=scale_range[0], maxval=scale_range[1]
        )

        def upsample_and_crop_single(img, scale, crop_key):
            new_H = int(H * scale)
            new_W = int(W * scale)

            # Upscale image
            upscale_img = jax.image.resize(img, (new_H, new_W, C), method="bilinear")

            # Random crop back to original size
            start_h = jax.random.randint(crop_key, (), 0, new_H - H + 1)
            start_w = jax.random.randint(crop_key, (), 0, new_W - W + 1)
            cropped = upscale_img[start_h : start_h + H, start_w : start_w + W, :]
            return cropped

        # Split key for each image's crop
        crop_keys = jax.random.split(key, B)
        return jax.vmap(upsample_and_crop_single)(img, scale, crop_keys)

    def img_rotation(
        self, img: jax.Array, key: jax.Array, rotation_angle: float = 3.0
    ) -> jax.Array:
        """Rotation of image on training set."""
        B, H, W, C = img.shape
        rotation = jax.random.uniform(
            key, (B,), minval=-rotation_angle, maxval=rotation_angle
        )

        # dm_pix.rotate(H, W, C), so vmap over batch
        def rotate_single(img, angle):
            return pix.rotate(
                image=img, angle_radians=jnp.deg2rad(angle), interpolation="bilinear"
            )

        return jax.vmap(rotate_single)(img, rotation)

    def flip_img(self, img: jax.Array, key: jax.Array) -> jax.Array:
        """Random horizontal flip using dm_pix."""
        B, H, W, C = img.shape
        flip = jax.random.bernoulli(key, 0.5, (B,))

        def flip_single(img, should_flip):
            return jax.lax.cond(
                should_flip, lambda: pix.flip_left_right(img), lambda: img
            )

        return jax.vmap(flip_single)(img, flip)


class GroupNorm(nnx.Module):
    """Group Normalization layer."""

    def __init__(
        self,
        num_groups: int,
        num_features: int,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.gn = nnx.GroupNorm(
            num_groups=num_groups,
            num_features=num_features,
            epsilon=eps,
            rngs=rngs,  # Fixed: added rngs parameter
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.gn(x)


class ResidualBlock(nnx.Module):
    """Residual block similar to identity mapping that uses pre-activation."""

    def __init__(
        self,
        planes: int,
        channels: int,
        stride: int = 1,
        num_groups: int = 8,
        expansion: int = 4,
        l2reg: float = 0.001,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.planes = planes
        self.channels = channels
        self.stride = stride
        self.num_groups = num_groups
        self.expansion = expansion
        self.l2reg = l2reg
        self.out_channels = planes * expansion

        # 1x1 reduce
        self.gn1 = GroupNorm(
            num_groups=num_groups, num_features=channels, rngs=rngs
        )  # Fixed: added rngs
        self.prelu1 = nnx.PReLU()
        self.conv1 = Conv2d(
            in_features=channels,
            out_features=planes,
            kernel_size=(1, 1),
            strides=(1, 1),
            l2reg=l2reg,
            rngs=rngs,
        )

        # 3x3 downsampling
        self.gn2 = GroupNorm(
            num_groups=num_groups, num_features=planes, rngs=rngs
        )  # Fixed: added rngs
        self.prelu2 = nnx.PReLU()
        self.conv2 = Conv2d(
            in_features=planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding="SAME",
            l2reg=l2reg,
            rngs=rngs,
        )

        # 1x1 expand
        self.gn3 = GroupNorm(
            num_groups=num_groups, num_features=planes, rngs=rngs
        )  # Fixed: added rngs
        self.prelu3 = nnx.PReLU()
        self.conv3 = Conv2d(
            in_features=planes,
            out_features=self.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            l2reg=l2reg,
            rngs=rngs,
        )

        # Shortcut connection
        if stride != 1 or channels != self.out_channels:
            self.shortcut_conv = Conv2d(
                in_features=channels,
                out_features=self.out_channels,
                kernel_size=(1, 1),
                strides=(stride, stride),
                l2reg=l2reg,
                rngs=rngs,
            )
        else:
            self.shortcut_conv = None

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x

        # 1x1 reduce
        out = self.gn1(x)
        out = self.prelu1(out)
        out = self.conv1(out)

        # 3x3 downsampling
        out = self.gn2(out)
        out = self.prelu2(out)
        out = self.conv2(out)

        # 1x1 expand
        out = self.gn3(out)
        out = self.prelu3(out)
        out = self.conv3(out)

        # Shortcut connection
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)

        return out + residual


class Classifier(nnx.Module):
    """A ResNet model for CIFAR classification with L2 regularization."""

    def __init__(
        self,
        num_classes: int,
        base_planes: int = 64,
        block_counts: tuple = (3, 4, 6, 3),
        num_groups: int = 8,
        expansion: int = 4,
        l2reg: float = 0.001,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_planes = base_planes
        self.block_counts = block_counts
        self.num_groups = num_groups
        self.expansion = expansion
        self.l2reg = l2reg

        # Initial convolution
        self.conv1 = Conv2d(
            in_features=3,  # CIFAR has 3 channels
            out_features=base_planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            l2reg=l2reg,
            rngs=rngs,
        )

        # Residual stages
        self.stages = nnx.List()
        planes_list = [base_planes, base_planes * 2, base_planes * 4, base_planes * 8]
        strides = [1, 2, 2, 2]

        current_channels = base_planes
        for i, (planes, num_blocks, stride) in enumerate(
            zip(planes_list, block_counts, strides)
        ):
            stage_blocks = nnx.List()
            for j in range(num_blocks):
                # First block in stage may downsample
                block_stride = stride if j == 0 else 1
                block = ResidualBlock(
                    planes=planes,
                    channels=current_channels,
                    stride=block_stride,
                    num_groups=num_groups,
                    expansion=expansion,
                    l2reg=l2reg,
                    rngs=rngs,
                )
                stage_blocks.append(block)
                current_channels = block.out_channels

            self.stages.append(stage_blocks)

        # Final layers
        self.gn_final = GroupNorm(
            num_groups=num_groups, num_features=current_channels, rngs=rngs
        )  # Fixed: added rngs
        self.prelu_final = nnx.PReLU()
        self.dense = nnx.Linear(
            in_features=current_channels, out_features=num_classes, rngs=rngs
        )

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        # Initial convolution
        x = self.conv1(x)

        # Residual stages
        for stage in self.stages:
            for block in stage:
                x = block(x)

        # Final layers
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

        # Initial convolution
        total_loss += self.conv1.l2_loss()

        # Residual blocks' convolutions
        for stage in self.stages:
            for block in stage:
                total_loss += block.conv1.l2_loss()
                total_loss += block.conv2.l2_loss()
                total_loss += block.conv3.l2_loss()
                if block.shortcut_conv is not None:
                    total_loss += block.shortcut_conv.l2_loss()

        return total_loss
