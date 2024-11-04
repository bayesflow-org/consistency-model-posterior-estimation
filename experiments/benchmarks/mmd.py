import tensorflow as tf

MMD_BANDWIDTH_LIST = [0.01, 0.05, 0.1, 0.5]

def maximum_mean_discrepancy(source_samples, target_samples, kernel="gaussian", mmd_weight=1.0, minimum=0.0):
    """Computes the MMD given a particular choice of kernel.

    For details, consult Gretton et al. (2012):
    https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    Parameters
    ----------
    source_samples : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution.
    target_samples : tf.Tensor of shape  (M, num_features)
        An array of `M` random draws from the "target" distribution.
    kernel         : str in ('gaussian', 'inverse_multiquadratic'), optional, default: 'gaussian'
        The kernel to use for computing the distance between pairs of random draws.
    mmd_weight     : float, optional, default: 1.0
        The weight of the MMD value.
    minimum        : float, optional, default: 0.0
        The lower bound of the MMD value.

    Returns
    -------
    loss_value : tf.Tensor
        A scalar Maximum Mean Discrepancy, shape (,)
    """

    # Determine kernel, fall back to Gaussian if unknown string passed
    if kernel == "gaussian":
        kernel_fun = gaussian_kernel_matrix
    elif kernel == "inverse_multiquadratic":
        kernel_fun = inverse_multiquadratic_kernel_matrix
    else:
        kernel_fun = gaussian_kernel_matrix

    # Compute and return MMD value
    loss_value = mmd_kernel(source_samples, target_samples, kernel=kernel_fun)
    loss_value = mmd_weight * tf.maximum(minimum, loss_value)
    return loss_value


def inverse_multiquadratic_kernel_matrix(x, y, sigmas=None):
    """Computes an inverse multiquadratic RBF between the samples of x and y.

    We create a sum of multiple IM-RBF kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x       :  tf.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  tf.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    sigmas  : list(float), optional, default: None
        List which denotes the widths of each of the gaussians in the kernel.
        If `sigmas is None`, a default range will be used, contained in `bayesflow.default_settings.MMD_BANDWIDTH_LIST`

    Returns
    -------
    kernel  : tf.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
    """

    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    dist = tf.expand_dims(tf.reduce_sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=-1)
    sigmas = tf.expand_dims(sigmas, 0)
    return tf.reduce_sum(sigmas / (dist + sigmas), axis=-1)


def gaussian_kernel_matrix(x, y, sigmas=None):
    """Computes a Gaussian radial basis functions (RBFs) between the samples of x and y.

    We create a sum of multiple Gaussian kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x       :  tf.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  tf.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    sigmas  : list(float), optional, default: None
        List which denotes the widths of each of the gaussians in the kernel.
        If `sigmas is None`, a default range will be used, contained in ``bayesflow.default_settings.MMD_BANDWIDTH_LIST``

    Returns
    -------
    kernel  : tf.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
    """

    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    norm = lambda v: tf.reduce_sum(tf.square(v), 1)
    beta = 1.0 / (2.0 * (tf.expand_dims(sigmas, 1)))
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    kernel = tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    return kernel

def mmd_kernel(x, y, kernel):
    """Computes the estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between random draws from
    the distributions `x ~ P` and `y ~ Q`.

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution `x ~ P`.
    y      : tf.Tensor of shape (M, num_features)
        An array of `M` random draws from the "target" distribution `y ~ Q`.
    kernel : callable
        A function which computes the distance between pairs of samples.

    Returns
    -------
    loss   : tf.Tensor of shape (,)
        The statistically biased squared maximum mean discrepancy (MMD) value.
    """

    loss = tf.reduce_mean(kernel(x, x))
    loss += tf.reduce_mean(kernel(y, y))
    loss -= 2 * tf.reduce_mean(kernel(x, y))
    return loss

def mmd_kernel_unbiased(x, y, kernel):
    """Computes the unbiased estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions `x ~ P` and `y ~ Q`.

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution `x ~ P`.
    y      : tf.Tensor of shape (M, num_features)
        An array of `M` random draws from the "target" distribution `y ~ Q`.
    kernel : callable
        A function which computes the distance between pairs of random draws from `x` and `y`.

    Returns
    -------
    loss   : tf.Tensor of shape (,)
        The statistically unbiaserd squared maximum mean discrepancy (MMD) value.
    """

    m, n = x.shape[0], y.shape[0]
    loss = (1.0 / (m * (m + 1))) * tf.reduce_sum(kernel(x, x))
    loss += (1.0 / (n * (n + 1))) * tf.reduce_sum(kernel(y, y))
    loss -= (2.0 / (m * n)) * tf.reduce_sum(kernel(x, y))
    return loss