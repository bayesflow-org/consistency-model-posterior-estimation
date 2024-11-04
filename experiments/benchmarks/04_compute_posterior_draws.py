import os
import pickle
import tensorflow as tf
import numpy as np

from train import get_setup
from tqdm import tqdm


tasks = ["gmm", "twomoons", "invkinematics"]
simulation_budgets = [512, 1024, 2048, 4096, 8192]
estimators = ['ac', 'nsf', 'cmpe', 'fmpe']

eval_estimators = list(
    set(estimators + ['cmpe10', 'cmpe30', 'fmpe10', 'fmpe30']).difference({'cmpe'})
)


eval_dict = {task: {budget: {estimator: {} for estimator in estimators} for budget in simulation_budgets} for task in tasks}

run_idx = 0

if __name__ == '__main__':
    if not os.path.exists('./computations'):
        os.mkdir('./computations')
    for task in tasks:
        for budget in simulation_budgets:
            for estimator in estimators:
                train_data_full = pickle.load(open(f'./data/{task}_train_data.pkl', 'rb'))
                train_data = {
                    'sim_data': train_data_full.get('sim_data')[:budget],
                    'prior_draws': train_data_full.get('prior_draws')[:budget],
                }
                sigma2 = tf.math.reduce_variance(tf.constant(train_data["prior_draws"], dtype=tf.float32), axis=0, keepdims=True)
                ckpt_path = f'./checkpoints/{task}_{estimator}_{budget}_run{run_idx}'
                trainer, settings = get_setup(task, estimator, sigma2, budget, ckpt_path)
                eval_dict[task][budget][estimator]['trainer'] = trainer
                eval_dict[task][budget][estimator]['settings'] = settings

    # evaluate the estimators on the test data
    total = len(tasks) * len(simulation_budgets) * len(estimators)
    num_posterior_samples = 4000
    with tqdm(total=total) as pbar:
        for task in tasks:
            for budget in simulation_budgets:
                for estimator in estimators:
                    trainer = eval_dict[task][budget][estimator]['trainer']
                    settings = eval_dict[task][budget][estimator]['settings']
                    eval_data = trainer.configurator(pickle.load(open(f'./data/{task}_test_data.pkl', 'rb')))
                    if estimator == 'cmpe':
                        eval_dict[task][budget]['cmpe10'] = {}
                        eval_dict[task][budget]['cmpe30'] = {}
                        eval_dict[task][budget]['cmpe10']['posterior_samples'] = trainer.amortizer.sample(eval_data, n_steps=10, n_samples=num_posterior_samples, to_numpy=False)
                        eval_dict[task][budget]['cmpe30']['posterior_samples'] = trainer.amortizer.sample(eval_data, n_steps=30, n_samples=num_posterior_samples, to_numpy=False)
                    elif estimator == 'fmpe':
                        eval_dict[task][budget]['fmpe10'] = {}
                        eval_dict[task][budget]['fmpe30'] = {}
                        eval_dict[task][budget]['fmpe']['posterior_samples'] = trainer.amortizer.sample(eval_data, n_samples=num_posterior_samples, to_numpy=False)
                        eval_dict[task][budget]['fmpe10']['posterior_samples'] = trainer.amortizer.sample(eval_data, step_size=1.0/10.0, n_samples=num_posterior_samples, to_numpy=False)
                        eval_dict[task][budget]['fmpe30']['posterior_samples'] = trainer.amortizer.sample(eval_data, step_size=1.0/30.0, n_samples=num_posterior_samples, to_numpy=False)
                    else:
                        eval_dict[task][budget][estimator]['posterior_samples'] = trainer.amortizer.sample(eval_data, n_samples=num_posterior_samples, to_numpy=False)
                    pbar.update(1)

    for task in tasks:
        for budget in simulation_budgets:
            for estimator in eval_estimators:
                if not (type(eval_dict[task][budget][estimator]['posterior_samples']) == np.ndarray):
                    eval_dict[task][budget][estimator]['posterior_samples'] = eval_dict[task][budget][estimator]['posterior_samples'].numpy()

    for task in tasks:
        for budget in simulation_budgets:
            for estimator in estimators:
                try:
                    del eval_dict[task][budget][estimator]['trainer']
                    del eval_dict[task][budget][estimator]['settings']
                except KeyError:
                    continue


    # save
    with open('./computations/eval_dict.pkl', 'wb') as f:
        pickle.dump(eval_dict, f)
