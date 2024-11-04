import os
import numpy as np
import pickle
from tqdm import tqdm
from c2st import c2st
from train import get_setup
import tensorflow as tf


if __name__ == '__main__':
    if not os.path.exists('./computations'):
        os.mkdir('./computations')
    tasks = ['gmm', 'twomoons', 'invkinematics']
    budget = 1024
    estimators = ['ac', 'nsf', 'cmpe', 'fmpe']

    sampling_steps = [2, 5, 7, 8, 10, 20, 30, 40, 50, 100]

    reference_posteriors = {
        'gmm': pickle.load(open('./data/gmm_reference_posterior_samples.pkl', 'rb')),
        'twomoons': pickle.load(open('./data/twomoons_reference_posterior_samples.pkl', 'rb')),
        'invkinematics': pickle.load(open('./data/invkinematics_reference_posterior_samples_custom_abc_0.002.pkl', 'rb')),
    }

    # set up a nested dictionary (task -> simulation budget -> estimator) with comprehensions so that it's easy do adjust
    # the number of simulations and estimators
    trainer_dict = {task: {estimator: {} for estimator in estimators} for task in tasks}

    run_idx = 0

    for task in tasks:
        for estimator in estimators:
            train_data_full = pickle.load(open(f'./data/{task}_train_data.pkl', 'rb'))
            train_data = {
                'sim_data': train_data_full.get('sim_data')[:budget],
                'prior_draws': train_data_full.get('prior_draws')[:budget],
            }
            sigma2 = tf.math.reduce_variance(tf.constant(train_data["prior_draws"], dtype=tf.float32), axis=0, keepdims=True)
            ckpt_path = f'./checkpoints/{task}_{estimator}_{budget}_run{run_idx}'
            trainer, settings = get_setup(task, estimator, sigma2, budget, ckpt_path)
            trainer_dict[task][estimator]['trainer'] = trainer
            trainer_dict[task][estimator]['settings'] = settings

    import time

    def sample_timed(trainer, num_runs=3, **kwargs):
        t_min = np.inf

        for _ in range(num_runs):
            tic = time.time()
            samples = trainer.amortizer.sample(**kwargs)
            toc = time.time()
            t = toc - tic
            if t < t_min:
                t_min = t
                samples_t_min = samples

        return samples_t_min, t_min

    # evaluate the estimators on the test data
    eval_dict = {task: {estimator: {} for estimator in estimators} for task in tasks}
    total = len(tasks) * len(estimators)
    num_posterior_samples = 4000
    num_runs_timed = 5

    with tqdm(desc='timed posterior sampling', total=total) as pbar:
        for task in tasks:
            for estimator in estimators:
                trainer = trainer_dict[task][estimator]['trainer']
                settings = trainer_dict[task][estimator]['settings']
                eval_data = trainer.configurator(pickle.load(open(f'./data/{task}_test_data.pkl', 'rb')))
                if estimator == 'cmpe':
                    for step in sampling_steps:
                        eval_dict[task][f'cmpe{step}'] = {}
                        samples, t = sample_timed(trainer, num_runs=num_runs_timed, input_dict=eval_data, n_steps=step, n_samples=num_posterior_samples, to_numpy=False)
                        eval_dict[task][f'cmpe{step}']['posterior_samples'] = samples if type(samples) == np.ndarray else samples.numpy()
                        eval_dict[task][f'cmpe{step}']['time'] = t
                elif estimator == 'fmpe':
                    eval_dict[task][f'fmpe'] = {}
                    samples, t = sample_timed(trainer, num_runs=num_runs_timed, input_dict=eval_data, n_samples=num_posterior_samples, to_numpy=False)
                    eval_dict[task][f'fmpe']['posterior_samples'] = samples if type(samples) == np.ndarray else samples.numpy()
                    eval_dict[task][f'fmpe']['time'] = t
                    for step in sampling_steps:
                        eval_dict[task][f'fmpe{step}'] = {}
                        samples, t = sample_timed(trainer, num_runs=num_runs_timed, input_dict=eval_data, step_size=1.0/step, n_samples=num_posterior_samples, to_numpy=False)
                        eval_dict[task][f'fmpe{step}']['posterior_samples'] = samples if type(samples) == np.ndarray else samples.numpy()
                        eval_dict[task][f'fmpe{step}']['time'] = t
                else:
                    samples, t = sample_timed(trainer, num_runs=num_runs_timed, input_dict=eval_data, n_samples=num_posterior_samples, to_numpy=False)
                    eval_dict[task][estimator]['posterior_samples'] = samples if type(samples) == np.ndarray else samples.numpy()
                    eval_dict[task][estimator]['time'] = t
                pbar.update(1)

    pickle.dump(eval_dict, open('./computations/eval_dict_speed_performance_ik.pkl', 'wb'))

    reference_posteriors = {
        'gmm': pickle.load(open('./data/gmm_reference_posterior_samples.pkl', 'rb')),
        'twomoons': pickle.load(open('./data/twomoons_reference_posterior_samples.pkl', 'rb')),
        'invkinematics': pickle.load(open('./data/invkinematics_reference_posterior_samples_custom_abc_0.002.pkl', 'rb')),
    }

    for task in tasks:
        try:
            del eval_dict[task]['cmpe']
        except:
            continue
    
    total = len(tasks) * len(eval_dict['twomoons'].keys())
    with tqdm(desc='c2st computation', total=total) as pbar:
        for task in tasks:
            n_test_instances = reference_posteriors[task].shape[0]
            for estimator in eval_dict[task].keys():
                c2st_scores = np.array([c2st(
                    reference_posteriors[task][i],
                    eval_dict[task][estimator]['posterior_samples'][i], seed=0) for i in range(n_test_instances)])
                eval_dict[task][estimator]['c2st'] = c2st_scores
                pbar.update(1)

    for task in tasks:
        try:
            del eval_dict[task]['cmpe']
        except:
            continue
        
    #save
    pickle.dump(eval_dict, open('./computations/eval_dict_speed_performance_ik.pkl', 'wb'))
