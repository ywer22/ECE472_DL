import jax
import jax.numpy as jnp
import structlog
import dm_pix as pix
from flax import nnx

log = structlog.get_logger()


class Conv2d(nnx.Module):
    """A 2D convolutional layer wrapper."""

    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    padding: str = "SAME"

    @nnx.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.Conv(
            in_features=x.shape[-1],
            out_features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )(x)
        return x


class Data_Augmentation(nnx.Module):
    """Data Augmentation layer in charge of upscaling, rotation, and contast."""

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
        self, img: jax.Array, key: jax.Array, rotation_angle: float = 15.0
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
        x = self.img_rotation(x, keys[1], rotation_angle=15.0)
        x = self.flip_img(x, keys[2])

        return x


class GroupNorm(nnx.Module):
    """Group Normalization layer."""

    def __init__(
        self,
        num_features: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.rngs = rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        N, H, W, C = x.shape
        assert C % self.num_groups == 0, "num_features must be divisible by num_groups"
        layer = nnx.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.eps,
            use_scale=True,
            use_bias=True,
            rngs=self.rngs,
        )

        nnx.state(layer)
