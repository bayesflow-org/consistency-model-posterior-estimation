import argparse
import datetime
import os
import pickle
import sys

import bayesflow as bf
import numpy as np
import tensorflow as tf
from bayesflow.experimental.rectifiers import RectifiedDistribution
from bayesflow.trainers import Trainer
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append("../../")
from amortizers import ConsistencyAmortizer


def grayscale_camera(theta, noise="poisson", psf_width=2.5, noise_scale=1, noise_gain=0.5):
    """Creates a noisy blurred image.

    Parameters:
    ----------
    theta       : input image to be blurred.
    noise       : noise type.
    psf_width   : width of point-spread function.
    noise_scale : scale for noise distribution.
    noise_gain  : gain for noise distribution.
    """

    image1 = noise_gain * random_noise(noise_scale * theta, mode=noise)
    image2 = gaussian_filter(image1, sigma=psf_width)
    return image2


def configurator(f):
    out = {}

    B = f["prior_draws"].shape[0]
    H = f["prior_draws"].shape[1]
    W = f["prior_draws"].shape[2]

    # Normalize image between -1 and 1
    p = (f["prior_draws"]).reshape((B, H * W)).astype(np.float32)
    p = -1.0 + (p * 2) / 255.0

    # Add blurr
    blurred = np.stack([grayscale_camera(f["sim_data"][b]) for b in range(B)]).astype(np.float32)

    # Add posterior inputs + some dequantization noise
    out["parameters"] = p.reshape((B, H * W))
    out["summary_conditions"] = blurred[..., None]
    return out


class GeneralDriftNetwork(tf.keras.Model):
    """Implements a learnable velocity field for a neural ODE. Will typically be used
    in conjunction with a ``RectifyingFlow`` instance, as proposed by [1] in the context
    of unconditional image generation.

    [1] Liu, X., Gong, C., & Liu, Q. (2022).
    Flow straight and fast: Learning to generate and transfer data with rectified flow.
    arXiv preprint arXiv:2209.03003.
    """

    def __init__(
        self,
        net,
        input_dim,
        **kwargs,
    ):
        """Creates a learnable velocity field instance to be used in the context of rectifying
        flows or neural ODEs.

        [1] Liu, X., Gong, C., & Liu, Q. (2022).
        Flow straight and fast: Learning to generate and transfer data with rectified flow.
        arXiv preprint arXiv:2209.03003.

        Parameters
        ----------
        net: tf.keras.Model
            The network to use as a drift network
        """

        super().__init__(**kwargs)

        # set for compatibility with RectifiedDistribution
        self.latent_dim = input_dim
        self.net = net

    def call(self, target_vars, latent_vars, time, condition, **kwargs):
        """Performs a linear interpolation between target and latent variables
        over time (i.e., a single ODE step during training).

        Parameters
        ----------
        target_vars : tf.Tensor of shape (batch_size, ..., num_targets)
            The variables of interest (e.g., parameters) over which we perform inference.
        latent_vars : tf.Tensor of shape (batch_size, ..., num_targets)
            The sampled random variates from the base distribution.
        time        : tf.Tensor of shape (batch_size, ..., 1)
            A vector of time indices in (0, 1)
        condition   : tf.Tensor of shape (batch_size, ..., condition_dim)
            The optional conditioning variables (e.g., as returned by a summary network)
        **kwargs    : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model`` call() method
        """

        diff = target_vars - latent_vars
        wdiff = time * target_vars + (1 - time) * latent_vars
        drift = self.drift(wdiff, time, condition, **kwargs)
        return diff, drift

    def drift(self, target_t, time, condition, **kwargs):
        """Returns the drift at target_t time given optional condition(s).

        Parameters
        ----------
        target_t    : tf.Tensor of shape (batch_size, ..., num_targets)
            The variables of interest (e.g., parameters) over which we perform inference.
        time        : tf.Tensor of shape (batch_size, ..., 1)
            A vector of time indices in (0, 1)
        condition   : tf.Tensor of shape (batch_size, ..., condition_dim)
            The optional conditioning variables (e.g., as returned by a summary network)
        **kwargs    : dict, optional, default: {}
            Optional keyword arguments passed to the drift network.
        """
        inp = [
            tf.reshape(target_t, (-1, tf.shape(target_t)[-1])),
            tf.reshape(condition, (-1, tf.shape(condition)[-1])),
            tf.reshape(time, (-1, 1)),
        ]
        net_out = self.net(inp, **kwargs)
        return tf.reshape(net_out, tf.shape(target_t))


# Source code for networks adapted from: https://keras.io/examples/generative/ddpm/
# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, tmax, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.tmax = tmax
        self.half_dim = dim // 2
        self.emb = tf.math.log(10000.0) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32) * 1000.0 / self.tmax
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(units, activation=activation_fn, kernel_initializer=kernel_init(1.0))(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply


def build_naive_model(
    img_size,
    tmax,
    cond_dim=128,
    dense_dim=1024,
    groups=8,
):
    image_input = layers.Input(shape=(img_size * img_size), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.float32, name="time_input")
    condition_input = keras.Input(shape=(cond_dim), dtype=tf.float32, name="condition_input")

    temb = TimeEmbedding(dim=64, tmax=tmax)(time_input)
    temb = TimeMLP(units=64, activation_fn="relu")(temb)
    x = layers.Concatenate(axis=-1)([image_input, condition_input, temb])

    x = layers.Dense(dense_dim, activation="relu")(x)
    z = layers.Dense(dense_dim)(x)
    x = layers.Add()([x, z])
    x = keras.activations.relu(x)
    z = layers.Dense(dense_dim)(x)
    x = layers.Add()([x, z])
    x = keras.activations.relu(x)
    z = layers.Dense(dense_dim)(x)
    x = layers.Add()([x, z])
    x = layers.Dense(img_size * img_size)(x)

    model = keras.Model([image_input, condition_input, time_input], x, name="naive")
    model.latent_dim = img_size * img_size
    model.input_dim = img_size * img_size
    model.condition_dim = cond_dim
    return model


## Consistency Network - U-Net
def build_unet_model(
    img_size,
    widths,
    has_attention,
    tmax,
    cond_dim=128,
    num_res_blocks=2,
    norm_groups=8,
    first_conv_channels=16,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
):
    image_input = layers.Input(shape=(img_size * img_size), name="image_input")
    image = layers.Reshape((img_size, img_size, 1))(image_input)
    time_input = keras.Input(shape=(), dtype=tf.float32, name="time_input")
    condition_input = keras.Input(shape=(cond_dim), dtype=tf.float32, name="condition_input")

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image)

    temb = TimeEmbedding(dim=first_conv_channels * 4, tmax=tmax)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    temb = layers.Concatenate(axis=-1)([temb, condition_input])

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))(x)
    x = layers.Flatten()(x)
    model = keras.Model([image_input, condition_input, time_input], x, name="unet")
    model.latent_dim = img_size * img_size
    model.input_dim = img_size * img_size
    model.condition_dim = cond_dim
    return model


def build_summary_network(groups=8):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                input_shape=(28, 28, 1),
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.GroupNormalization(groups=groups),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.GroupNormalization(groups=groups),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.GroupNormalization(groups=groups),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.GroupNormalization(groups=groups),
            tf.keras.layers.GlobalAveragePooling2D(),
        ]
    )


def build_trainer(checkpoint_path, args, forward_train=None):
    summary_net = build_summary_network()
    img_size = 28
    input_dim = int(img_size * img_size)
    cond_dim = 128

    tmax = args.tmax if args.method == "cmpe" else 1.0

    if args.architecture == "unet":
        norm_groups = 8  # Number of groups used in GroupNormalization layer
        first_conv_channels = 16
        channel_multiplier = [1, 2, 4]
        widths = [first_conv_channels * mult for mult in channel_multiplier]
        has_attention = [False, False, True]
        num_res_blocks = 2  # Number of residual blocks
        consistency_net = build_unet_model(
            img_size=img_size,
            widths=widths,
            tmax=tmax,
            has_attention=has_attention,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
            activation_fn=keras.activations.swish,
        )
    elif args.architecture == "naive":
        consistency_net = build_naive_model(
            img_size=img_size,
            cond_dim=cond_dim,
            dense_dim=args.dense_dim,
            tmax=tmax,
        )

    batch_size = args.batch_size
    num_steps = args.num_steps
    initial_learning_rate = 5e-4
    if forward_train is not None:
        num_batches = np.ceil(forward_train["prior_draws"].shape[0] / batch_size)
        num_epochs = int(np.ceil(num_steps / num_batches))
        num_steps = num_epochs * num_batches
    else:
        num_epochs = 0
        num_steps = 0

    if args.method == "cmpe":
        sigma2 = args.sigma2
        epsilon = args.epsilon
        s0 = args.s0
        s1 = args.s1

        amortized_posterior = ConsistencyAmortizer(
            consistency_net=consistency_net,
            num_steps=num_steps,
            summary_net=summary_net,
            sigma2=sigma2,
            eps=epsilon,
            T_max=tmax,
            s0=s0,
            s1=s1,
        )

    elif args.method == "fmpe":
        s0 = None
        s1 = None
        drift_net = GeneralDriftNetwork(consistency_net, input_dim)

        amortized_posterior = RectifiedDistribution(
            drift_net,
            summary_net,
        )
    else:
        raise ValueError(f"Method '{args.method}' not supported.")

    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(checkpoint_path, "args.pickle"), "wb") as f:
        pickle.dump(args, f)

    trainer = Trainer(amortized_posterior, configurator=configurator, checkpoint_path=checkpoint_path)

    if forward_train is not None:
        # Optimizer
        if args.lr_adapt == "cosine":
            lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, num_steps)
        elif args.lr_adapt == "none":
            lr = args.initial_learning_rate
        else:
            raise ValueError(f"Invalid value for learning rate adaptation: '{args.lr_adapt}'")

        if args.optimizer.lower() == "adamw":
            optimizer = tf.keras.optimizers.AdamW(lr)
        else:
            optimizer = type(tf.keras.optimizers.get(args.optimizer))(lr)

        return trainer, optimizer, num_epochs, batch_size

    return trainer


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except (ValueError, RuntimeError):
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=40000)
    parser.add_argument("--num-training", type=int, default=60000)
    parser.add_argument("--tmax", type=float, default=200)
    parser.add_argument("--lr-adapt", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--s0", type=int, default=10)
    parser.add_argument("--s1", type=int, default=50)
    parser.add_argument("--sigma2", type=float, default=0.25)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--method", type=str, default="cmpe", choices=["cmpe", "fmpe"])
    parser.add_argument("--architecture", type=str, default="unet", choices=["unet", "naive"])
    # dense-dim for naive architecture
    parser.add_argument("--dense-dim", type=int, default=1024)

    args = parser.parse_args()
    # Load and split data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images[: args.num_training]
    train_labels = train_labels[: args.num_training]

    forward_train = {"prior_draws": train_images, "sim_data": train_images}

    # split into validation and test set
    num_val = 500
    perm = np.random.default_rng(seed=42).permutation(test_images.shape[0])

    forward_val = {
        "prior_draws": test_images[perm[:num_val]],
        "sim_data": test_images[perm[:num_val]],
    }

    forward_test = {
        "prior_draws": test_images[perm[num_val:]],
        "sim_data": test_images[perm[num_val:]],
    }

    val_labels = test_labels[perm[:num_val]]
    test_labels = test_labels[perm[num_val:]]
    checkpoint_path = os.path.join(
        "checkpoints",
        f"{args.method}-{args.architecture}-{args.num_training}-{datetime.datetime.today():%y-%m-%d-%H%M%S}",
    )

    trainer, optimizer, num_epochs, batch_size = build_trainer(checkpoint_path, args, forward_train=forward_train)

    print(f"Training for {num_epochs} epochs...")

    h = trainer.train_offline(
        forward_train, optimizer=optimizer, epochs=num_epochs, batch_size=batch_size, validation_sims=forward_val
    )
