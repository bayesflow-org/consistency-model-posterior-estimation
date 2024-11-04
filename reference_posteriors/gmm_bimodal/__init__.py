import tensorflow as tf
import tensorflow_probability as tfp
import logging

class GMM(tfp.distributions.MixtureSameFamily):
    def __init__(self, theta):
        logging.getLogger().setLevel(logging.ERROR)
        mixture_weights_dist = tfp.distributions.Categorical(probs=[0.5, 0.5])
        components_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.stack([theta, -1.0 * theta], axis=1), scale_diag=[[0.5, 0.5]]
        )

        super().__init__(mixture_distribution=mixture_weights_dist, components_distribution=components_dist)


class GMMPrior:
    def __init__(self, prior_location=[0, 0], prior_scale_diag=[1, 1]):
        self.dist = tfp.distributions.MultivariateNormalDiag(loc=prior_location, scale_diag=prior_scale_diag)

    def __call__(self, batch_size=None):
        theta = self.dist.sample([batch_size]) if batch_size else self.dist.sample()
        return theta


class GMMSimulator:
    def __init__(self, dist, n_obs=10):
        self.dist = dist
        self.n_obs = n_obs

    def __call__(self, theta, n_obs=None):
        if n_obs is None:
            n_obs = self.n_obs
        x = self.dist(theta).sample([n_obs])
        x = tf.transpose(x, perm=[1, 0, 2])
        return x
