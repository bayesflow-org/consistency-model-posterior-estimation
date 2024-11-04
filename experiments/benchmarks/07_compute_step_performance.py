import os
import pickle

import numpy as np
import tensorflow as tf
from c2st import c2st
from tqdm import tqdm
from train import get_setup

if __name__ == "__main__":
    n_test_instances = 100

    # load reference posteriors from the respective files
    reference_posteriors = {
        "gmm": pickle.load(open("./data/gmm_reference_posterior_samples.pkl", "rb"))[:n_test_instances],
        "twomoons": pickle.load(open("./data/twomoons_reference_posterior_samples.pkl", "rb"))[:n_test_instances],
        "invkinematics": pickle.load(
            open("./data/invkinematics_reference_posterior_samples_custom_abc_0.002.pkl", "rb")
        )[:n_test_instances],
    }

    computations_file = "./computations/eval_dict_steps_performance_v2.0.pkl"
    tasks = ["gmm", "twomoons", "invkinematics"]
    simulation_budgets = [8192]
    estimators = ["cmpe"]
    sampling_steps = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50]

    run_idx = 0

    # load existing computations (in case the computations have to be resumed)
    if os.path.exists(computations_file):
        eval_dict = pickle.load(open(computations_file, "rb"))
    else:
        eval_dict = {}

    # load checkpoints, recalculate sigma2 from training data
    for task in tasks:
        if task not in eval_dict:
            eval_dict[task] = {}
        for budget in simulation_budgets:
            if budget not in eval_dict[task]:
                eval_dict[task][budget] = {}
            for estimator in estimators:
                if estimator not in eval_dict[task][budget]:
                    eval_dict[task][budget][estimator] = {}
                train_data_full = pickle.load(open(f"./data/{task}_train_data.pkl", "rb"))
                train_data = {
                    "sim_data": train_data_full.get("sim_data")[:budget],
                    "prior_draws": train_data_full.get("prior_draws")[:budget],
                }
                sigma2 = tf.math.reduce_variance(
                    tf.constant(train_data["prior_draws"], dtype=tf.float32), axis=0, keepdims=True
                )
                ckpt_path = f"./checkpoints/{task}_{estimator}_{budget}_run{run_idx}"
                trainer, settings = get_setup(task, estimator, sigma2, budget, ckpt_path)
                eval_dict[task][budget][estimator]["trainer"] = trainer
                eval_dict[task][budget][estimator]["settings"] = settings

    # evaluate the estimators (in this case only CMPE) on the test data
    total = len(tasks) * len(simulation_budgets) * len(estimators)
    num_posterior_samples = 4000
    with tqdm(desc="generating posterior samples", total=total) as pbar:
        for task in tasks:
            n_test_instances = reference_posteriors[task].shape[0]
            for budget in simulation_budgets:
                for estimator in estimators:
                    trainer = eval_dict[task][budget][estimator]["trainer"]
                    settings = eval_dict[task][budget][estimator]["settings"]
                    eval_data = trainer.configurator(pickle.load(open(f"./data/{task}_test_data.pkl", "rb")))
                    eval_data["parameters"] = eval_data["parameters"][:n_test_instances]
                    if "direct_conditions" in eval_data:
                        eval_data["direct_conditions"] = eval_data["direct_conditions"][:n_test_instances]
                    if "summary_conditions" in eval_data:
                        eval_data["summary_conditions"] = eval_data["summary_conditions"][:n_test_instances]
                    if estimator == "cmpe":
                        for n_steps in tqdm(sampling_steps, leave=False):
                            eval_dict[task][budget][f"cmpe{n_steps}"] = {}
                            eval_dict[task][budget][f"cmpe{n_steps}"]["posterior_samples"] = trainer.amortizer.sample(
                                eval_data, n_steps=n_steps, n_samples=num_posterior_samples, to_numpy=False
                            )

                    pbar.update(1)

    # remove trainer and settings object, so that eval_dict can be stored using pickle
    for task in tasks:
        for budget in simulation_budgets:
            for estimator in eval_dict[task][budget].keys():
                if "posterior_samples" in eval_dict[task][budget][estimator] and not (
                    type(eval_dict[task][budget][estimator]["posterior_samples"]) == np.ndarray
                ):
                    eval_dict[task][budget][estimator]["posterior_samples"] = eval_dict[task][budget][estimator][
                        "posterior_samples"
                    ].numpy()

                try:
                    del eval_dict[task][budget][estimator]["trainer"]
                    del eval_dict[task][budget][estimator]["settings"]
                except KeyError:
                    continue

    pickle.dump(eval_dict, open(computations_file, "wb"))

    # compute c2st scores
    total = len(tasks) * len(simulation_budgets) * len(estimators)
    with tqdm(desc="computing c2st scores", total=total) as pbar:
        for task in tasks:
            n_test_instances = reference_posteriors[task].shape[0]
            for budget in simulation_budgets:
                for estimator in estimators:
                    if estimator == "cmpe":
                        for n_steps in tqdm(sampling_steps, leave=False):
                            c2st_vals = []
                            for i in tqdm(range(n_test_instances)):
                                c2st_vals.append(
                                    c2st(
                                        reference_posteriors[task][i],
                                        eval_dict[task][budget][f"cmpe{n_steps}"]["posterior_samples"][i],
                                        seed=0,
                                    )
                                )
                            eval_dict[task][budget][f"cmpe{n_steps}"]["c2st"] = np.array(c2st_vals)
                            pickle.dump(eval_dict, open(computations_file, "wb"))
                    pbar.update(1)

    pickle.dump(eval_dict, open(computations_file, "wb"))
