import numpy as np
import pickle
from tqdm import tqdm
from c2st import c2st
import os

if __name__ == '__main__':

    if not os.path.exists('./computations'):
        os.mkdir('./computations')

    tasks = ['gmm', 'twomoons', 'invkinematics']
    simulation_budgets = [512, 1024, 2048, 4096, 8192]
    estimators = ['ac', 'nsf', 'cmpe', 'fmpe']

    eval_estimators = list(
        set(estimators + ['cmpe10', 'cmpe30', 'fmpe10', 'fmpe30']).difference({'cmpe'})
    )

    eval_dict = pickle.load(open(f'./computations/eval_dict.pkl', 'rb'))

    reference_posteriors = {
        'gmm': pickle.load(open('./data/gmm_reference_posterior_samples.pkl', 'rb')),
        'twomoons': pickle.load(open('./data/twomoons_reference_posterior_samples.pkl', 'rb')),
        'invkinematics': pickle.load(open('./data/invkinematics_reference_posterior_samples_custom_abc_0.002.pkl', 'rb')),
    }

    n_test_instances = 100

    c2st_dict = {task: {budget: {estimator: {} for estimator in eval_estimators} for budget in simulation_budgets} for task in tasks}
    total = len(tasks) * len(simulation_budgets) * len(eval_estimators)

    with tqdm(total=total) as pbar:
        for task in tasks:
            for estimator in eval_estimators:
                for budget in simulation_budgets:
                    c2st_vals = []
                    for i in tqdm(range(n_test_instances)):
                        c2st_vals.append(c2st(
                            reference_posteriors[task][i],
                            eval_dict[task][budget][estimator]['posterior_samples'][i], seed=0))
                    c2st_dict[task][budget][estimator] = np.array(c2st_vals)
                    pbar.update(1)

                    with open(f'./computations/c2st_dict_budget_performance.pkl', 'wb') as f:
                        pickle.dump(c2st_dict, f)
