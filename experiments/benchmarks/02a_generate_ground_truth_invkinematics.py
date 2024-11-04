import pickle
import sys

import numpy as np
import tensorflow as tf

sys.path.append("../../")

import os

from tqdm import trange

num_posterior_samples = 4000

if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("./data")

    # inverse kinematics
    lens = np.array([0.5, 0.5, 1.0]).astype(np.float32)

    @tf.function
    def sample_prior(N):
        return tf.random.normal((N, 4), dtype=tf.float32) * tf.constant([0.25, 0.5, 0.5, 0.5], dtype=tf.float32)

    @tf.function
    def tf_segment_points(p_, length, angle):
        p = p_ + tf.stack([length * tf.cos(angle), length * tf.sin(angle)], axis=-1)
        return p_, p

    @tf.function
    def tf_forward_process(x, lens):
        start = tf.stack([tf.zeros((x.shape[0])), x[:, 0]], axis=1)
        _, x1 = tf_segment_points(start, lens[0], x[:, 1])
        _, x2 = tf_segment_points(x1, lens[1], x[:, 1] + x[:, 2])
        _, y = tf_segment_points(x2, lens[2], x[:, 1] + x[:, 2] + x[:, 3])
        return y

    test_data = pickle.load(open("./data/invkinematics_test_data.pkl", "rb"))

    y_test_showcase = np.array([[0.0, 1.5]])
    y_test_data = test_data["sim_data"]
    y_combined = np.append(y_test_data, y_test_showcase, axis=0)

    # switch axis in data space to match simulator
    y_test = tf.constant(y_combined[:, ::-1], dtype=tf.float32)
    theta_test = test_data["prior_draws"]

    n_sim, n_params = theta_test.shape
    n_sim += 1  # add showcase example

    reference_samples = np.empty((n_sim, num_posterior_samples, n_params))

    # ABC sampling
    epsilon = 0.002
    # For epsilon=0.002, training took 21706 for a batch_size of 100 000 000 (on a GPU).
    # For running on a CPU, a smaller batch size is advisable, e.g. 1 000 000.
    max_iter = 10_000_000  # maximum number of batches (program will stop earlier when all samples are produced)
    batch_size = 100_000_000  # number of simulations to produce at the same time

    sample_ind = np.zeros((n_sim,), dtype=np.int32)
    for i in trange(max_iter, desc="Batch number"):
        x = sample_prior(batch_size)
        y = tf_forward_process(x, lens)
        for j in range(n_sim):
            if sample_ind[j] == num_posterior_samples:
                continue
            mask = tf.norm(y - y_test[j], axis=-1) < tf.constant(epsilon, dtype=tf.float32)
            n_sampled = tf.math.count_nonzero(mask)
            if n_sampled > 0:
                n_missing = num_posterior_samples - sample_ind[j]
                n_new = min(n_missing, n_sampled)
                end_ind = sample_ind[j] + n_new
                reference_samples[j, sample_ind[j] : end_ind] = x[mask][:n_new]
                sample_ind[j] = end_ind
        # save
        pickle.dump(
            reference_samples.astype(np.float32)[:-1],
            open(f"./data/invkinematics_reference_posterior_samples_custom_abc_{epsilon}.pkl", "wb"),
        )
        pickle.dump(
            sample_ind[:-1],
            open(f"./data/invkinematics_reference_posterior_samples_ind_custom_abc_{epsilon}.pkl", "wb"),
        )
        pickle.dump(
            reference_samples.astype(np.float32)[-1],
            open(f"./data/invkinematics_showcase_ref_custom_abc_{epsilon}.pkl", "wb"),
        )
        pickle.dump(sample_ind[-1], open(f"./data/invkinematics_showcase_ref_ind_custom_abc_{epsilon}.pkl", "wb"))
        if sample_ind.sum() >= n_sim * num_posterior_samples:
            break
        print(
            f"\n{np.argmin(sample_ind)}: {sample_ind.min()}, {sample_ind.sum()}/{n_sim * num_posterior_samples}\n",
            end="\r",
        )
