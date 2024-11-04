import pickle
import sys

import numpy as np

from bayesflow.benchmarks import Benchmark
sys.path.append('../../')

import bayesflow as bf
import os
import tensorflow as tf

from bayesflow.experimental.rectifiers import RectifiedDistribution

from amortizers import ConfigurableMLP, ConsistencyAmortizer, DriftNetwork


tasks = ["gmm", "twomoons", "invkinematics"]
simulation_budgets = [512, 1024, 2048, 4096, 8192]
estimators = ['ac', 'nsf', 'fmpe', 'cmpe']
num_runs = 1

def configurator(forward_dict):
    return {
        'summary_conditions': forward_dict['sim_data'],
        'parameters': forward_dict['prior_draws']
    }

def get_setup(task_name, estimator, sigma2=1.0, simulation_budget=1000, checkpoint_path=""):
    if task_name == 'gmm':
        INPUT_DIM = 2
        CONDITION_DIM = 2
        HIDDEN_DIM = 256
        ACTIVATION = 'relu'
        RESIDUAL_CONNECTIONS = True
        DROPOUT_RATE = 0.10
        EPSILON=0.001
        T_MAX=1.0
        S0 = 10.0
        S1 = 1280.0
        NUM_HIDDEN = 3

        NUM_EPOCHS = 2000
        BATCH_SIZE = 64
        INITIAL_LR = 1e-4
        INITIAL_LR_FM = 5e-5
        KERNEL_REGULARIZATION = 1e-4


        if estimator == 'ac':
            trainer = bf.trainers.Trainer(
                amortizer=bf.amortizers.AmortizedPosterior(
                    inference_net = bf.networks.InvertibleNetwork(num_params=INPUT_DIM, num_coupling_layers=4, coupling_design="affine"),
                    summary_net = bf.networks.SetTransformer(input_dim=2, summary_dim=2)),
                    checkpoint_path=checkpoint_path,
                    configurator=configurator,
                    skip_checks=True
            )
            settings = {'epochs': 200, 'batch_size': 32}
        elif estimator == 'nsf':
            trainer = bf.trainers.Trainer(
                amortizer=bf.amortizers.AmortizedPosterior(
                    inference_net = bf.networks.InvertibleNetwork(num_params=INPUT_DIM, num_coupling_layers=4, coupling_design="spline"),
                    summary_net = bf.networks.SetTransformer(input_dim=2, summary_dim=2)),
                    checkpoint_path=checkpoint_path,
                    configurator=configurator,
                    skip_checks=True
)
            settings = {'epochs': 200, 'batch_size': 32}
        elif estimator == 'fmpe':
            trainer = bf.trainers.Trainer(
                    amortizer=RectifiedDistribution(
                        DriftNetwork(
                            input_dim=INPUT_DIM,
                            cond_dim=CONDITION_DIM,
                            hidden_dim=HIDDEN_DIM,
                            activation=ACTIVATION,
                            residual_connections=RESIDUAL_CONNECTIONS,
                            dropout_rate=DROPOUT_RATE,
                            kernel_regularization=KERNEL_REGULARIZATION
                        ),
                        summary_net = bf.networks.SetTransformer(input_dim=2, summary_dim=2)
                    ),
                    checkpoint_path=checkpoint_path,
                    configurator=configurator,
                    default_lr=INITIAL_LR_FM,
                    skip_checks=True
                )
            settings = {'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE}
        elif estimator == 'cmpe':
            inference_net = ConfigurableMLP(
                input_dim=INPUT_DIM,
                condition_dim=CONDITION_DIM,
                hidden_dim=HIDDEN_DIM,
                activation=ACTIVATION,
                residual_connections=RESIDUAL_CONNECTIONS,
                dropout_rate=DROPOUT_RATE,
                kernel_regularization=KERNEL_REGULARIZATION,
                num_hidden=NUM_HIDDEN
            )
            cm_amortizer = ConsistencyAmortizer(
                consistency_net=inference_net,
                num_steps=NUM_EPOCHS * np.ceil(simulation_budget / BATCH_SIZE),
                summary_net=bf.networks.SetTransformer(input_dim=2, summary_dim=2),
                eps=EPSILON,
                T_max=T_MAX,
                s0 = S0,
                s1 = S1,
                sigma2=sigma2
            )
            trainer= bf.trainers.Trainer(
                amortizer=cm_amortizer,
                configurator=configurator,
                default_lr=INITIAL_LR,
                checkpoint_path=checkpoint_path,
                skip_checks=True
            )
            settings = {'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE}
        else:
            raise ValueError(f'Estimator {estimator} not recognized for task {task_name}...')
    elif task_name == 'twomoons':
        INPUT_DIM = 2
        CONDITION_DIM = 2
        HIDDEN_DIM = 256
        ACTIVATION = 'relu'
        RESIDUAL_CONNECTIONS = True
        DROPOUT_RATE = 0.05
        EPSILON=0.001
        T_MAX=10.0
        S0 = 10.0
        S1 = 50.0
        NUM_HIDDEN = 2

        NUM_EPOCHS = 5000
        BATCH_SIZE = 64
        INITIAL_LR = 5e-4
        KERNEL_REGULARIZATION = 1e-5

        if estimator == 'ac':
            trainer = bf.trainers.Trainer(
                amortizer=bf.amortizers.AmortizedPosterior(
                inference_net = bf.networks.InvertibleNetwork(num_params=INPUT_DIM, 
                                                              num_coupling_layers=6, 
                                                              coupling_design="affine", 
                                                              coupling_settings={'dense_args' : dict(units=128), 'kernel_regularizer': tf.keras.regularizers.l2(1e-4)}),
                ), 
                checkpoint_path=checkpoint_path,
                configurator=Benchmark('two_moons', mode='posterior').configurator,
                skip_checks=True
            )
            settings = {'epochs': 200, 'batch_size': 32}
        elif estimator == 'nsf':
            trainer = bf.trainers.Trainer(
                amortizer=bf.amortizers.AmortizedPosterior(
                inference_net = bf.networks.InvertibleNetwork(num_params=INPUT_DIM, 
                                                              num_coupling_layers=6, 
                                                              coupling_design="spline", 
                                                              coupling_settings={'dense_args' : dict(units=128), 'kernel_regularizer': tf.keras.regularizers.l2(1e-4)}),
                ), 
                checkpoint_path=checkpoint_path,
                configurator=Benchmark('two_moons', mode='posterior').configurator,
                skip_checks=True
            )
            settings = {'epochs': 200, 'batch_size': 32}
        elif estimator == 'fmpe':
            trainer = bf.trainers.Trainer(
                amortizer=RectifiedDistribution(
                    DriftNetwork(
                        input_dim=INPUT_DIM,
                        cond_dim=CONDITION_DIM,
                        hidden_dim=HIDDEN_DIM,
                        num_hidden=NUM_HIDDEN,
                        activation=ACTIVATION,
                        residual_connections=RESIDUAL_CONNECTIONS,
                        dropout_rate=DROPOUT_RATE,
                        kernel_regularization=KERNEL_REGULARIZATION
                    ),
                ),
                configurator=Benchmark('two_moons', mode='posterior').configurator,
                default_lr=INITIAL_LR,
                checkpoint_path=checkpoint_path,
                skip_checks=True
            )
            settings = {'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE}
        elif estimator == 'cmpe':
            inference_net = ConfigurableMLP(
                input_dim=INPUT_DIM,
                condition_dim=CONDITION_DIM,
                hidden_dim=HIDDEN_DIM,
                num_hidden=NUM_HIDDEN,
                activation=ACTIVATION,
                residual_connections=RESIDUAL_CONNECTIONS,
                dropout_rate=DROPOUT_RATE,
                kernel_regularization=KERNEL_REGULARIZATION,
            )
            cm_amortizer = ConsistencyAmortizer(
                consistency_net=inference_net,
                num_steps=NUM_EPOCHS * np.ceil(simulation_budget / BATCH_SIZE),
                eps=EPSILON,
                T_max=T_MAX,
                s0 = S0,
                s1 = S1,
                sigma2=sigma2
            )
            trainer = bf.trainers.Trainer(
                amortizer=cm_amortizer,
                configurator=Benchmark('two_moons', mode='posterior').configurator,
                default_lr=INITIAL_LR,
                checkpoint_path=checkpoint_path,
                skip_checks=True
            )
            settings = {'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE}
        else:
            raise ValueError(f'Estimator {estimator} not recognized for task {task_name}...')
    elif task_name == 'invkinematics':
        INPUT_DIM = 4
        CONDITION_DIM = 2
        HIDDEN_DIM = 256
        ACTIVATION = 'relu'
        RESIDUAL_CONNECTIONS = True
        DROPOUT_RATE = 0.05
        EPSILON=0.001
        T_MAX=10.0
        S0 = 10.0
        S1 = 50.0
        NUM_HIDDEN = 2

        NUM_EPOCHS = 2000
        BATCH_SIZE = 32
        INITIAL_LR = 5e-4
        KERNEL_REGULARIZATION = 1e-5
        if estimator == 'ac':
            trainer = bf.trainers.Trainer(
                    amortizer=bf.amortizers.AmortizedPosterior(
                    inference_net = bf.networks.InvertibleNetwork(num_params=INPUT_DIM, num_coupling_layers=6, coupling_design="affine", coupling_settings={'dense_args' : {'units': 128, 'kernel_regularizer': tf.keras.regularizers.l2(1e-4)}}),
                    ), 
                configurator = Benchmark("inverse_kinematics", mode="posterior").configurator,
                checkpoint_path=checkpoint_path,
                skip_checks=True
            )
            settings = {'epochs': 200, 'batch_size': 32}
        elif estimator == 'nsf':
            trainer = bf.trainers.Trainer(
                    amortizer=bf.amortizers.AmortizedPosterior(
                    inference_net = bf.networks.InvertibleNetwork(num_params=INPUT_DIM, num_coupling_layers=6, coupling_design="spline", coupling_settings={'dense_args' : {'units': 128, 'kernel_regularizer': tf.keras.regularizers.l2(1e-4)}}),
                    ), 
                configurator = Benchmark("inverse_kinematics", mode="posterior").configurator,
                checkpoint_path=checkpoint_path,
                skip_checks=True
            )
            settings = {'epochs': 200, 'batch_size': 32}
        elif estimator == 'fmpe':
            trainer = bf.trainers.Trainer(
                amortizer=RectifiedDistribution(
                    DriftNetwork(
                        input_dim=INPUT_DIM,
                        cond_dim=CONDITION_DIM,
                        hidden_dim=HIDDEN_DIM,
                        num_hidden=NUM_HIDDEN,
                        activation=ACTIVATION,
                        residual_connections=RESIDUAL_CONNECTIONS,
                        dropout_rate=DROPOUT_RATE,
                        kernel_regularization=KERNEL_REGULARIZATION,
                    ),
                ),
        configurator=Benchmark("inverse_kinematics", mode="posterior").configurator,
        checkpoint_path=checkpoint_path,
        skip_checks=True
    )
            settings = {'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE}
        elif estimator == 'cmpe':
            inference_net = ConfigurableMLP(
                input_dim=INPUT_DIM,
                condition_dim=CONDITION_DIM,
                hidden_dim=HIDDEN_DIM,
                num_hidden=NUM_HIDDEN,
                activation=ACTIVATION,
                residual_connections=RESIDUAL_CONNECTIONS,
                dropout_rate=DROPOUT_RATE,
                kernel_regularization=KERNEL_REGULARIZATION,
            )
            cm_amortizer = ConsistencyAmortizer(
                consistency_net=inference_net,
                num_steps=NUM_EPOCHS * np.ceil(simulation_budget / BATCH_SIZE),
                eps=EPSILON,
                T_max=T_MAX,
                s0 = S0,
                s1 = S1,
                sigma2=sigma2
            )
            trainer = bf.trainers.Trainer(
                amortizer=cm_amortizer,
                configurator=Benchmark("inverse_kinematics", mode="posterior").configurator,
                default_lr=INITIAL_LR,
                checkpoint_path=checkpoint_path,
                skip_checks=True
            )
            settings = {'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE}
        else:
            raise ValueError(f'Estimator {estimator} not recognized for task {task_name}...')
    else:
        raise ValueError(f'Task {task_name} not recognized...')
    
    return trainer, settings



def train(estimator, simulation_budget, task_name, run_idx):
    print(f'Starting training of {estimator} network with a budget of {simulation_budget} simulations for task {task_name}...')
    ckpt_path = f'./checkpoints/{task_name}_{estimator}_{simulation_budget}_run{run_idx}'

    with open(f'./data/{task_name}_train_data.pkl', 'rb') as f:
        train_data_full = pickle.load(f)
    with open(f'./data/{task_name}_validation_data.pkl', 'rb') as f:
        validation_data = pickle.load(f)

    train_data = {
        'sim_data': train_data_full.get('sim_data')[:simulation_budget],
        'prior_draws': train_data_full.get('prior_draws')[:simulation_budget],
    }

    sigma2 = tf.math.reduce_variance(tf.constant(train_data["prior_draws"], dtype=tf.float32), axis=0, keepdims=True)

    if not os.path.exists(ckpt_path):
        trainer, settings = get_setup(task_name, estimator, sigma2, simulation_budget, ckpt_path)
        _ = trainer.train_offline(train_data, 
                                  epochs=settings['epochs'], 
                                  batch_size=settings['batch_size'], 
                                  validation_sims=validation_data,
                                  save_checkpoint=False)
        trainer._save_trainer(True)
        print('Completed...')
        tf.keras.backend.clear_session()
        del trainer
    else:
        print(f'Skipping, since {ckpt_path} exists...')


if __name__ == '__main__':
    for run_idx in range(num_runs):
        for task in tasks:
            for simulation_budget in simulation_budgets:
                for estimator in estimators:
                    train(estimator, simulation_budget, task, run_idx)
