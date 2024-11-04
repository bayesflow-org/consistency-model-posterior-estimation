import bayesflow.default_settings as defaults
import numpy as np
import tensorflow as tf
from bayesflow.amortizers import AmortizedPosterior

# Training helpers
from bayesflow.exceptions import SummaryStatsError
from bayesflow.losses import mmd_summary_space
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Input, Model


@tf.function
def discretize_time(eps, T_max, num_steps, rho=7.0):
    """Function for obtaining the discretized time according to
    https://arxiv.org/pdf/2310.14189.pdf, Section 2, bottom of page 2.

    Parameters:
    -----------
    T_max   : int
        Maximal time (corresponds to $\sigma_{max}$)
    eps     : float
        Minimal time (correspond to $\sigma_{min}$)
    N       : int
        Number of discretization steps
    rho     : number
        Control parameter
    """
    N = tf.cast(num_steps, tf.float32) + 1.0
    i = tf.range(1, N + 1, dtype=tf.float32)
    one_over_rho = 1.0 / rho
    discretized_time = (
        eps**one_over_rho + (i - 1.0) / (N - 1.0) * (T_max**one_over_rho - eps**one_over_rho)
    ) ** rho
    return discretized_time


class ConfigurableHiddenBlock(tf.keras.Model):
    def __init__(
        self, num_units, activation="relu", residual_connection=True, dropout_rate=0.0, kernel_regularization=0.0
    ):
        super().__init__()

        self.act_func = tf.keras.activations.get(activation)
        self.residual_connection = residual_connection
        self.dense = tf.keras.layers.Dense(
            num_units, activation=None, kernel_regularizer=regularizers.l2(kernel_regularization)
        )
        self.dropout_rate = dropout_rate

    @tf.function
    def call(self, inputs, training=False, mask=None):
        x = self.dense(inputs)
        x = tf.nn.dropout(x, self.dropout_rate)

        if self.residual_connection:
            x += inputs
        return self.act_func(x)


# Source code for networks adapted from: https://keras.io/examples/generative/ddpm/
# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")

# class TimeEmbedding(tf.keras.layers.Layer):
#     def __init__(self, dim, tmax, **kwargs):
#         super().__init__(**kwargs)
#         self.dim = dim
#         self.tmax = tmax
#         self.half_dim = dim // 2
#         self.emb = tf.math.log(10000.0) / (self.half_dim - 1)
#         self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

#     @tf.function
#     def call(self, inputs):
#         inputs = tf.cast(inputs, dtype=tf.float32) * 1000.0 / self.tmax
#         emb = inputs[:, None] * self.emb[None, :]
#         emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
#         return emb

class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, tmax, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.tmax = tmax
        self.half_dim = dim // 2
        self.freqs = tf.exp(
        -tf.math.log(self.tmax) * tf.range(0, self.half_dim, dtype=tf.float32) / self.half_dim
    )

    @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs = inputs[:, tf.newaxis] * self.freqs[tf.newaxis]
        embedding = tf.concat([tf.cos(inputs), tf.sin(inputs)], axis=-1)
        if self.dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

def TimeMLP(units, activation_fn=tf.keras.activations.swish):
    def apply(inputs):
        temb = tf.keras.layers.Dense(units, activation=activation_fn, kernel_initializer=kernel_init(1.0))(inputs)
        temb = tf.keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

class ConfigurableMLP(tf.keras.Model):
    """Implements a configurable MLP with optional residual connections and dropout."""

    def __init__(
        self,
        input_dim,
        condition_dim,
        hidden_dim=512,
        num_hidden=2,
        activation="relu",
        residual_connections=True,
        dropout_rate=0.0,
        kernel_regularization=0.0,
    ):
        """
        Creates an instance of a flexible MLP with optional residual connections
        and dropout.

        Parameters:
        -----------
        input_dim : int
            The input dimensionality
        condition_dim  : int
            The dimensionality of the condition
        hidden_dim: int, optional, default: 512
            The dimensionality of the hidden layers
        num_hidden: int, optional, default: 2
            The number of hidden layers (minimum 1)
        eps       : float, optional, default: 0.002
            The minimum time
        activation: string, optional, default: 'relu'
            The activation function of the dense layers
        T_max     : float, optional, default: 0.20
            End time of the diffusion
        N         : int, optional, default: s1
            Discretization level during inference
        residual_connections: bool, optional, default: True
            Use residual connections in the MLP
        dropout_rate        : float, optional, default: 0.0
            Dropout rate for the hidden layers in the MLP
        kernel_regularization: float, optional, default: 0.0
            L2 regularization factor for the kernel weights
        """
        # super(ConfigurableMLP, self).__init__()
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = input_dim  # only for compatibility with bayesflow.amortizers.AmortizedPosterior

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(kernel_regularization)
                ),
            ]
        )
        for _ in range(num_hidden):
            self.model.add(
                ConfigurableHiddenBlock(
                    hidden_dim,
                    activation=activation,
                    residual_connection=residual_connections,
                    dropout_rate=dropout_rate,
                    kernel_regularization=kernel_regularization,
                )
            )
        self.model.add(tf.keras.layers.Dense(input_dim))

    @tf.function
    def call(self, inputs, training=False, mask=None):
        return self.model(tf.concat(inputs, axis=-1), training=training)



def build_mlp(input_dim,
        condition_dim,
        hidden_dim=512,
        use_time_embedding=False,
        T_max=200.0,
        num_hidden=2,
        activation="relu",
        residual_connections=True,
        dropout_rate=0.0,
        kernel_regularization=0.0,):
    
    use_time_embedding = use_time_embedding


    x_input = Input(shape=(input_dim), name="x_input")
    
    time_input = Input(shape=(), dtype=tf.float32, name="time_input")
    condition_input = Input(shape=(condition_dim), dtype=tf.float32, name="condition_input")

    t = time_input
    if use_time_embedding:
        t = TimeEmbedding(dim=32, tmax=T_max)(time_input)
        t = TimeMLP(units=32, activation_fn="relu")(t)
    else:
        t = t[..., tf.newaxis]
    
    x = layers.Concatenate(axis=-1)([x_input, condition_input, t])

    x = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(kernel_regularization)
            ),
        ]
    )(x)

    for _ in range(num_hidden):
        x = ConfigurableHiddenBlock(
                hidden_dim,
                activation=activation,
                residual_connection=residual_connections,
                dropout_rate=dropout_rate,
                kernel_regularization=kernel_regularization,
            )(x)
    x = tf.keras.layers.Dense(input_dim)(x)

    model = Model([x_input, condition_input, time_input], x, name="MLP")
    model.input_dim = input_dim
    model.condition_dim = condition_dim
    model.latent_dim = input_dim  # only for compatibility with bayesflow.amortizers.AmortizedPosterior

    return model


class ConsistencyAmortizer(AmortizedPosterior):
    """Implements a consistency model according to https://arxiv.org/abs/2303.01469"""

    def __init__(
        self,
        consistency_net,
        num_steps,
        summary_net=None,
        loss_fun=None,
        summary_loss_fun=None,
        sigma2=1.0,
        eps=0.001,
        T_max=200.0,
        s0=10,
        s1=50,
        **kwargs,
    ):
        """
        Creates an instance of a consistency model (CM) to be used
        for standalone consistency training (CT).

        Parameters:
        -----------
        consistency_net         : tf.keras.Model
            A neural network for the consistency model
        input_dim : int
            The input dimensionality
        condition_dim  : int
            The dimensionality of the condition (or summary net output)
        num_steps: int
            The total number of training steps
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network to compress non-vector data structures.
        loss_fun          : callable or None, optional, default: None
            TODO: Currently unused, remove or implement, add documentation
        summary_loss_fun  : callable, str, or None, optional, default: None
            The loss function which accepts the outputs of the summary network. If ``None``, no loss is provided
            and the summary space will not be shaped according to a known distribution (see [2]).
            If ``summary_loss_fun='MMD'``, the default loss from [2] will be used.
        sigma2      : np.ndarray of shape (input_dim, 1), or float, optional, default: 1.0
            Controls the shape of the skip-function
        eps         : float, optional, default: 0.001
            The minimum time
        T_max       : flat, optional, default: 200.0
            The end time of the diffusion
        s0          : int, optional, default: 10
            Initial discretization steps
        s1          : int, optional, default: 50
            Final discretization steps
        **kwargs          : dict, optional, default: {}
            Additional keyword arguments passed to the ``__init__`` method of a ``tf.keras.Model`` instance via AmortizedPosterior.

        Important
        ----------
        - If no ``summary_net`` is provided, then the output dictionary of your generative model should not contain
        any ``summary_conditions``, i.e., ``summary_conditions`` should be set to ``None``, otherwise these will be ignored.
        """

        super().__init__(consistency_net, **kwargs)

        self.input_dim = consistency_net.input_dim
        self.condition_dim = consistency_net.condition_dim

        self.student = consistency_net
        self.student.build(
            input_shape=(
                None,
                self.input_dim + self.condition_dim + 1,
            )
        )

        self.summary_net = summary_net
        if loss_fun is not None:
            raise NotImplementedError("Only the default pseudo-huber loss is currently supported.")
        # self.loss_fun = self._determine_loss(loss_fun)
        self.summary_loss = self._determine_summary_loss(summary_loss_fun)

        self.sigma2 = tf.Variable(sigma2)
        self.sigma = tf.Variable(tf.math.sqrt(sigma2))
        self.eps = eps
        self.T_max = T_max
        # Choose coefficient according to https://arxiv.org/pdf/2310.14189.pdf, Section 3.3
        self.c_huber = 0.00054 * tf.sqrt(tf.cast(self.input_dim, tf.float32))
        self.c_huber2 = tf.square(self.c_huber)

        self.num_steps = tf.cast(num_steps, tf.float32)
        self.s0 = tf.cast(s0, tf.float32)
        self.s1 = tf.cast(s1, tf.float32)

        self.current_step = tf.Variable(0, trainable=False, dtype=tf.float32)

    @tf.function
    def call(self, input_dict, z, t, return_summary=False, **kwargs):
        """Performs a forward pass through the summary and consistency network given an input dictionary.

        Parameters
        ----------
        input_dict     : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``targets``            - the latent model parameters over which a condition density is learned
            ``summary_conditions`` - the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        z              : tf.Tensor of shape (batch_size, input_dim)
            The noise vector
        t              : tf.Tensor of shape (batch_size, 1)
            Vector of time samples in [eps, T]
        return_summary : bool, optional, default: False
            A flag which determines whether the learnable data summaries (representations) are returned or not.
        **kwargs       : dict, optional, default: {}
            Additional keyword arguments passed to the networks
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        net_out or (net_out, summary_out)
        """
        # Concatenate conditions, if given
        summary_out, full_cond = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            **kwargs,
        )
        # Extract target variables
        target_vars = input_dict[defaults.DEFAULT_KEYS["parameters"]]

        # Compute output
        inp = target_vars + t * z
        net_out = self.consistency_function(inp, full_cond, t, **kwargs)

        # Return summary outputs or not, depending on parameter
        if return_summary:
            return net_out, summary_out
        return net_out

    @tf.function
    def consistency_function(self, x, c, t, **kwargs):
        """Compute consistency function.

        Parameters
        ----------
        x : tf.Tensor of shape (batch_size, input_dim)
            Input vector
        c : tf.Tensor of shape (batch_size, condition_dim)
            The conditioning vector
        t : tf.Tensor of shape (batch_size, 1)
            Vector of time samples in [eps, T]
        """
        F = self.student([x, c, t], **kwargs)

        # Compute skip and out parts (vectorized, since self.sigma2 is of shape (1, input_dim)
        # Thus, we can do a cross product with the time vector which is (batch_size, 1) for
        # a resulting shape of cskip and cout of (batch_size, input_dim)
        cskip = self.sigma2 / ((t - self.eps) ** 2 + self.sigma2)
        cout = self.sigma * (t - self.eps) / (tf.math.sqrt(self.sigma2 + t**2))

        out = cskip * x + cout * F
        return out

    def compute_loss(self, input_dict, **kwargs):
        """Computes the loss of the posterior amortizer given an input dictionary, which will
        typically be the output of a Bayesian ``GenerativeModel`` instance.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``targets``            - the latent variables over which a condition density is learned
            ``summary_conditions`` - the conditioning variables that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        z          : tf.Tensor of shape (batch_size, input_dim)
            The noise vector
        t1         : tf.Tensor of shape (batch_size, 1)
            Vector of time samples in [eps, T]
        t2         : tf.Tensor of shape (batch_size, 1)
            Vector of time samples in [eps, T]
        TODO: add documentation for c, t1, t2
        **kwargs   : dict, optional, default: {}
            Additional keyword arguments passed to the networks
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        total_loss : tf.Tensor of shape (1,) - the total computed loss given input variables
        """
        self.current_step.assign_add(1.0)

        # Extract target variables and generate noise
        theta = input_dict.get(defaults.DEFAULT_KEYS["parameters"])
        z = tf.random.normal(tf.shape(theta))

        N_current = self._schedule_discretization(self.current_step, self.num_steps, s0=self.s0, s1=self.s1)
        discretized_time = discretize_time(self.eps, self.T_max, N_current)

        # Randomly sample t_n and t_[n+1] and reshape to (batch_size, 1)
        # adapted noise schedule from https://arxiv.org/pdf/2310.14189.pdf,
        # Section 3.5
        P_mean = -1.1
        P_std = 2.0
        log_p = tf.math.log(
            tf.math.erf((tf.math.log(discretized_time[1:]) - P_mean) / (tf.sqrt(2.0) * P_std))
            - tf.math.erf((tf.math.log(discretized_time[:-1]) - P_mean) / (tf.sqrt(2.0) * P_std))
        )
        times = tf.random.categorical([log_p], tf.shape(theta)[0])[0]
        t1 = tf.gather(discretized_time, times)[..., None]
        t2 = tf.gather(discretized_time, times + 1)[..., None]

        # Teacher is just the student without gradient tracing
        teacher_out = tf.stop_gradient(self(input_dict, z, t1, return_summary=False, **kwargs))
        student_out, sum_out = self(input_dict, z, t2, return_summary=True, **kwargs)
        # weighting function, see https://arxiv.org/pdf/2310.14189.pdf, Section 3.1
        lam = 1 / (t2 - t1)
        # Pseudo-huber loss, see https://arxiv.org/pdf/2310.14189.pdf, Section 3.3
        loss = tf.reduce_mean(lam * (tf.sqrt(tf.square(teacher_out - student_out) + self.c_huber2) - self.c_huber))

        # Case summary loss should be computed
        if self.summary_loss is not None:
            sum_loss = self.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Compute and return total loss
        total_loss = tf.reduce_mean(loss) + sum_loss
        return total_loss

    def sample(self, input_dict, n_samples, n_steps=10, to_numpy=True, step_size=1e-3, **kwargs):
        """Generates random draws from the approximate posterior given a dictionary with conditonal variables
        using the multistep sampling algorithm (Algorithm 1).

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        n_steps     : int
            The number of sampling steps
        TODO: This does not seem to work in some cases
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_data_sets, n_samples, n_params)
            The sampled parameters from the approximate posterior of each data set
        """

        # Compute condition (direct, summary, or both)
        _, conditions = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs,
        )
        n_data_sets, condition_dim = tf.shape(conditions)

        assert condition_dim == self.condition_dim

        post_samples = np.empty(shape=(n_data_sets, n_samples, self.input_dim), dtype=np.float32)
        n_data_sets, condition_dim = conditions.shape

        for i in range(n_data_sets):
            c = conditions[i, None]
            c_rep = tf.concat([c] * n_samples, axis=0)
            discretized_time = tf.reverse(discretize_time(self.eps, self.T_max, n_steps), axis=[-1])
            z_init = tf.random.normal((n_samples, self.input_dim), stddev=self.T_max)
            T = discretized_time[0] + tf.zeros((n_samples, 1))
            samples = self.consistency_function(z_init, c_rep, T)
            for n in range(1, n_steps):
                z = tf.random.normal((n_samples, self.input_dim))
                x_n = samples + tf.math.sqrt(discretized_time[n] ** 2 - self.eps**2) * z
                samples = self.consistency_function(x_n, c_rep, discretized_time[n] + tf.zeros((n_samples, 1)))
            post_samples[i] = samples

        # Remove trailing first dimension in the single data case
        if n_data_sets == 1:
            post_samples = tf.squeeze(post_samples, axis=0)

        # Return numpy version of tensor or tensor itself
        if to_numpy:
            return post_samples.numpy()
        return post_samples

    def _compute_summary_condition(self, summary_conditions, direct_conditions, **kwargs):
        """Determines how to concatenate the provided conditions."""

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sum_condition = self.summary_net(summary_conditions, **kwargs)
        else:
            sum_condition = None

        # Concatenate learnable summaries with fixed summaries
        if sum_condition is not None and direct_conditions is not None:
            full_cond = tf.concat([sum_condition, direct_conditions], axis=-1)
        elif sum_condition is not None:
            full_cond = sum_condition
        elif direct_conditions is not None:
            full_cond = direct_conditions
        else:
            raise SummaryStatsError("Could not concatenarte or determine conditioning inputs...")
        return sum_condition, full_cond

    def _determine_summary_loss(self, loss_fun):
        """Determines which summary loss to use if default `None` argument provided, otherwise return identity."""

        # If callable, return provided loss
        if loss_fun is None or callable(loss_fun):
            return loss_fun

        # If string, check for MMD or mmd
        elif isinstance(loss_fun, str):
            if loss_fun.lower() == "mmd":
                return mmd_summary_space
            else:
                raise NotImplementedError("For now, only 'mmd' is supported as a string argument for summary_loss_fun!")
        # Throw if loss type unexpected
        else:
            raise NotImplementedError(
                "Could not infer summary_loss_fun, argument should be of type (None, callable, or str)!"
            )

    def _determine_loss(self, loss_fun):
        """Determines which summary loss to use if default ``None`` argument provided, otherwise return identity."""

        if loss_fun is None:
            return tf.keras.losses.log_cosh
        return loss_fun

    @classmethod
    def _schedule_discretization(cls, k, K, s0=2.0, s1=100.0):
        """Schedule function for adjusting the discretization level `N` during the course
        of training. Implements the function N(k) from https://arxiv.org/abs/2310.14189,
        Section 3.4.

        Parameters:
        -----------
        k   : int
            Current iteration index.
        K   : int
            Final iteration index (len(dataset) * num_epochs)
        s0  : int, optional, default: 2
            The initial discretization steps
        s1  : int, optional, default: 100
            The final discretization steps
        """
        K_ = tf.floor(K / (tf.math.log(s1 / s0) / tf.math.log(2.0) + 1.0))
        out = tf.minimum(s0 * tf.pow(2.0, tf.floor(k / K_)), s1) + 1.0
        return tf.cast(out, tf.int32)


class DriftNetwork(tf.keras.Model):
    """Implements a learnable velocity field for a neural ODE. Will typically be used
    in conjunction with a ``RectifyingFlow`` instance, as proposed by [1] in the context
    of unconditional image generation.

    [1] Liu, X., Gong, C., & Liu, Q. (2022).
    Flow straight and fast: Learning to generate and transfer data with rectified flow.
    arXiv preprint arXiv:2209.03003.
    """

    def __init__(
        self,
        input_dim,
        cond_dim,
        hidden_dim=512,
        num_hidden=2,
        activation="relu",
        residual_connections=True,
        dropout_rate=0.0,
        kernel_regularization=0.0,
        **kwargs,
    ):
        """Creates a learnable velocity field instance to be used in the context of rectifying
        flows or neural ODEs.

        [1] Liu, X., Gong, C., & Liu, Q. (2022).
        Flow straight and fast: Learning to generate and transfer data with rectified flow.
        arXiv preprint arXiv:2209.03003.

        Parameters
        ----------
        input_dim : int
            The input dimensionality
        cond_dim  : int
            The dimensionality of the condition
        hidden_dim: int, optional, default: 512
            The dimensionality of the hidden layers
        num_hidden: int, optional, default: 2
            The number of hidden layers (minimum 1)
        eps       : float, optional, default: 0.002
            The minimum time
        activation: string, optional, default: 'relu'
            The activation function of the dense layers
        residual_connections: bool, optional, default: True
            Use residual connections in the MLP
        dropout_rate        : float, optional, default: 0.0
            Dropout rate for the hidden layers in the MLP
        kernel_regularization: float, optional, default: 0.0
            L2 regularization factor for the kernel weights
        """

        super().__init__(**kwargs)

        # set for compatibility with RectifiedDistribution
        self.latent_dim = input_dim
        self.net = ConfigurableMLP(
            input_dim=input_dim,
            condition_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
            residual_connections=residual_connections,
            dropout_rate=dropout_rate,
            kernel_regularization=kernel_regularization,
        )
        self.net.build(input_shape=())

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

        if condition is not None:
            inp = [target_t, condition, time]
        else:
            inp = [target_t, time]
        return self.net(inp, **kwargs)
