import pickle
import sys

import numpy as np

sys.path.append("../../")

import logging
import os

import cmdstanpy
from cmdstanpy import CmdStanModel
from tqdm import tqdm

from reference_posteriors.two_moons.two_moons_lueckmann_numpy import analytic_posterior_numpy

num_posterior_samples = 4000

if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("./data")

    # # gmm
    test_data = pickle.load(open("./data/gmm_test_data.pkl", "rb"))

    y_test = test_data["sim_data"].numpy()
    theta_test = test_data["prior_draws"]

    logger = logging.getLogger("cmdstanpy")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    iter_warmup = 2000

    n_sim, n_obs, data_dim = y_test.shape
    _, param_dim = theta_test.shape

    iter_sampling = num_posterior_samples // 2

    posterior_samples = np.zeros((n_sim, num_posterior_samples, param_dim))

    # Check if cmdstan is installed and set correct path
    try:
        path = cmdstanpy.cmdstan_path()
    except ValueError:
        # cmdstan not at default location
        try:
            # try installation location in the Apptainer image
            cmdstanpy.set_cmdstan_path("/opt/cmpe-cmdstan/cmdstan-2.33.1")
        except ValueError:
            print("cmdstan is not installed. Please install it to a location where cmdstanpy can find it.")
            exit(1)

    for i in tqdm(range(n_sim), desc="HMC running ..."):
        stan_data = {"n_obs": n_obs, "data_dim": data_dim, "x": y_test[i]}
        model = CmdStanModel(stan_file="../../reference_posteriors/gmm_bimodal/gmm.stan")
        fit = model.sample(
            data=stan_data,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            chains=1,
            inits={"theta": theta_test[i].tolist()},
            show_progress=False,
        )
        posterior_samples_chain = fit.stan_variable("theta")
        posterior_samples[i] = np.concatenate([posterior_samples_chain, -1.0 * posterior_samples_chain], axis=0)

    pickle.dump(posterior_samples.astype(np.float32), open("./data/gmm_reference_posterior_samples.pkl", "wb"))

    # two moons

    test_data = pickle.load(open("./data/twomoons_test_data.pkl", "rb"))

    y_test = test_data["sim_data"]
    theta_test = test_data["prior_draws"]

    n_sim, n_params = theta_test.shape

    analytic_samples = np.empty((n_sim, num_posterior_samples, n_params))

    for i in range(n_sim):
        analytic_samples[i] = analytic_posterior_numpy(
            y_test[i], num_posterior_samples, rng=np.random.default_rng(seed=1234)
        )

    pickle.dump(analytic_samples.astype(np.float32), open("./data/twomoons_reference_posterior_samples.pkl", "wb"))

    print("Done for twomoons and gmm. Use 02a_generate_ground_truth_invkinematics.py for inverse kinematics")
