import pickle
import sys
sys.path.append('../../')
import os

import bayesflow as bf
from bayesflow.benchmarks import Benchmark
import numpy as np
from reference_posteriors.gmm_bimodal import GMM, GMMPrior, GMMSimulator


if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')

    # GMM
    prior = GMMPrior(prior_location=[0, 0])
    simulator = GMMSimulator(GMM)

    generative_model = bf.simulation.GenerativeModel(
        prior=prior, simulator=simulator, prior_is_batched=True, simulator_is_batched=True
    )

    gmm_train_data = generative_model(8192)
    gmm_validation_data = generative_model(512)
    gmm_test_data = generative_model(100)

    with open('./data/gmm_train_data.pkl', 'wb') as f:
        pickle.dump(gmm_train_data, f)
    with open('./data/gmm_validation_data.pkl', 'wb') as f:
        pickle.dump(gmm_validation_data, f)
    with open('./data/gmm_test_data.pkl', 'wb') as f:
        pickle.dump(gmm_test_data, f)

    # Two Moons

    tm_benchmark = Benchmark("two_moons", mode="posterior")
    tm_train_data = tm_benchmark.generative_model(8192)
    tm_validation_data = tm_benchmark.generative_model(512)
    tm_test_data = tm_benchmark.generative_model(100)

    with open('./data/twomoons_train_data.pkl', 'wb') as f:
        pickle.dump(tm_train_data, f)
    with open('./data/twomoons_validation_data.pkl', 'wb') as f:
        pickle.dump(tm_validation_data, f)
    with open('./data/twomoons_test_data.pkl', 'wb') as f:
        pickle.dump(tm_test_data, f)


    # Inverse Kinematics

    ik_benchmark = Benchmark("inverse_kinematics", mode="posterior")
    ik_train_data = ik_benchmark.generative_model(8192)
    ik_validation_data = ik_benchmark.generative_model(512)
    ik_test_data = ik_benchmark.generative_model(100)

    with open('./data/invkinematics_train_data.pkl', 'wb') as f:
        pickle.dump(ik_train_data, f)
    with open('./data/invkinematics_validation_data.pkl', 'wb') as f:
        pickle.dump(ik_validation_data, f)
    with open('./data/invkinematics_test_data.pkl', 'wb') as f:
        pickle.dump(ik_test_data, f)
