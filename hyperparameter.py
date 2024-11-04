import argparse
import os

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from bayesflow.benchmarks import Benchmark
from bayesflow.computational_utilities import maximum_mean_discrepancy
from bayesflow.experimental.rectifiers import RectifiedDistribution
from bayesflow.helper_classes import SimulationDataset
from bayesflow.trainers import Trainer
from tensorboardX import SummaryWriter

from amortizers import ConfigurableMLP, ConsistencyAmortizer, DriftNetwork
from metrics import c2st
from reference_posteriors.gmm_bimodal import GMM, GMMPrior, GMMSimulator
from reference_posteriors.two_moons.two_moons_lueckmann_numpy import analytic_posterior_numpy


def evaluate_samples(
    trainer, eval_dict, writer, analytic_samples, epoch, N, n_samples_fig=5000, n_samples_mmd=1000, method=""
):
    # Store figure for multiple sampling steps
    if N is not None:
        samples = trainer.amortizer.sample(eval_dict, n_steps=N, n_samples=max(n_samples_fig, n_samples_mmd))
    else:
        samples = trainer.amortizer.sample(eval_dict, n_samples=max(n_samples_fig, n_samples_mmd))

    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.axis("equal")

    if analytic_samples is not None:
        mmd = maximum_mean_discrepancy(samples[:n_samples_mmd], analytic_samples[:n_samples_mmd]).numpy()
        c2st_accuracy = c2st(analytic_samples, samples)
        if N is not None:
            writer.add_scalar(f"mmd_{N}", mmd, global_step=epoch)
            writer.add_scalar(f"c2st_{N}", c2st_accuracy[0].numpy(), global_step=epoch)
        else:
            writer.add_scalar("mmd", mmd, global_step=epoch)
            writer.add_scalar("c2st", c2st_accuracy[0].numpy(), global_step=epoch)

        ax.scatter(
            analytic_samples[:n_samples_fig, 0],
            analytic_samples[:n_samples_fig, 1],
            color="orange",
            alpha=0.4,
            s=3,
            label="True",
        )
        if N is not None:
            ax.set_title(f"{method}: {N} steps, c2st={c2st_accuracy[0].numpy():.3f} MMD={mmd:.3f}")
        else:
            ax.set_title(f"{method}: c2st={c2st_accuracy[0].numpy():.3f} MMD={mmd:.3f}")
    else:
        if N is not None:
            ax.set_title(f"{method}: {N} steps")
        else:
            ax.set_title(f"{method}")
    ax.scatter(
        samples[:n_samples_fig, 0],
        samples[:n_samples_fig, 1],
        color="green",
        alpha=0.4,
        s=3,
        label="Model",
    )
    ax.grid(alpha=0.2)
    ax.legend()

    if N is not None:
        writer.add_figure(f"samples_{N}", f, global_step=epoch)
    else:
        writer.add_figure("samples", f, global_step=epoch)


if __name__ == "__main__":
    # change memory growth for GPU to dynamic
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except (ValueError, RuntimeError):
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=40000)
    parser.add_argument("--num-simulations", type=int, default=10000)
    parser.add_argument("--benchmark-name", type=str, default="two_moons")
    parser.add_argument("--tmax", type=float, default=200)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--num-hidden", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--lr-adapt", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--s0", type=int, default=10)
    parser.add_argument("--s1", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--method", type=str, default="cmpe", choices=["cmpe", "fmpe"])

    args = parser.parse_args()

    input_dim, cond_dim = None, None
    summary_net = None

    if args.benchmark_name == "gmm_bimodal":
        input_dim = 2
        cond_dim = 4

        summary_net = bf.networks.DeepSet(summary_dim=cond_dim)

        prior = GMMPrior(prior_location=[0, 0])
        simulator = GMMSimulator(GMM)

        generative_model = bf.simulation.GenerativeModel(
            prior=prior, simulator=simulator, prior_is_batched=True, simulator_is_batched=True
        )

        simulations = generative_model(args.num_simulations)

        eval_params = np.array([[-1.8, -1.0]]).astype(np.float32)
        eval_data = simulator(eval_params)
        eval_dict = {
            "summary_conditions": eval_data,
        }

        analytic_samples = None
    elif args.benchmark_name == "two_moons":
        input_dim = 2
        cond_dim = 2
        benchmark = Benchmark(args.benchmark_name, mode="posterior")

        generative_model = benchmark.generative_model
        configurator = benchmark.configurator

        simulations = benchmark.generative_model(args.num_simulations)

        condition = np.array([[0.0, 0.0]]).astype(np.float32)
        analytic_samples = tf.constant(
            analytic_posterior_numpy(condition[0], 5000, rng=np.random.default_rng(seed=1234)),
            dtype=tf.float32,
        )
        eval_dict = {
            "direct_conditions": condition,
        }
    else:
        input_dim = 2
        cond_dim = 2
        benchmark = Benchmark(args.benchmark_name, mode="posterior")

        generative_model = benchmark.generative_model
        configurator = benchmark.configurator

        simulations = benchmark.generative_model(args.num_simulations)

        analytic_samples = None
        eval_dict = None

    dataset = SimulationDataset(simulations, args.batch_size)
    # Use custom training loop - gives more control over logging/plotting
    writer = SummaryWriter(".")
    # Some training hyperparams
    batch_size = args.batch_size
    num_steps = args.num_steps
    num_simulations = args.num_simulations
    initial_learning_rate = args.initial_learning_rate
    num_epochs = int(np.ceil(num_steps / len(dataset)))
    num_steps = num_epochs * len(dataset)
    print(f"Running for {num_epochs} epochs.")

    # Epoch parameters
    T_max = args.tmax
    s0 = args.s0
    s1 = args.s1 if args.s1 > 0 else s0
    sampling_steps = args.sampling_steps if args.sampling_steps > 0 else s0

    # Consistency Model
    sigma2 = tf.math.reduce_variance(tf.constant(simulations["prior_draws"], dtype=tf.float32), axis=0, keepdims=True)

    if args.method == "cmpe":
        consistency_net = ConfigurableMLP(
            input_dim,
            cond_dim,
            args.hidden_dim,
            num_hidden=args.num_hidden,
            activation=args.activation,
            residual_connections=True,
            dropout_rate=args.dropout_rate,
        )

        amortizer = ConsistencyAmortizer(
            consistency_net=consistency_net,
            num_steps=num_steps,
            summary_net=summary_net,
            sigma2=sigma2,
            eps=args.epsilon,
            T_max=args.tmax,
            s0=s0,
            s1=s1,
        )

    elif args.method == "fmpe":
        sampling_steps = None
        s0 = None
        s1 = None
        drift_net = DriftNetwork(
            input_dim,
            cond_dim,
            args.hidden_dim,
            num_hidden=args.num_hidden,
            activation=args.activation,
            residual_connections=True,
            dropout_rate=args.dropout_rate,
        )

        amortizer = RectifiedDistribution(
            drift_net,
            summary_net,
        )
    else:
        raise ValueError(f"Method '{args.method}' not supported.")

    # Create output directory to store model
    output_dir = "output/"
    os.mkdir(output_dir)

    trainer = Trainer(
        amortizer,
        generative_model=benchmark.generative_model,
        configurator=benchmark.configurator,
        checkpoint_path=output_dir,
    )
    # _ = trainer.train_offline(dataset, epochs=args.epochs, batch_size=args.batch_size)

    # Optimizer
    if args.lr_adapt == "cosine":
        lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, num_steps)
    elif args.lr_adapt == "none":
        lr = args.initial_learning_rate
    else:
        raise ValueError(f"Invalid value for learning rate adaptation: '{args.lr_adapt}'")

    if args.optimizer.lower() == "adamw":
        optimizer = tf.keras.optimizers.AdamW(lr)
    else:
        optimizer = type(tf.keras.optimizers.get(args.optimizer))(lr)

    trainer.train_offline(simulations, num_epochs, batch_size, optimizer=optimizer)

    writer.add_scalar(
        "num_params",
        np.sum([np.prod(v.get_shape()) for v in trainer.amortizer.trainable_weights]),
    )

    if eval_dict is not None:
        # store figures and MMD values
        evaluate_samples(trainer, eval_dict, writer, analytic_samples, num_epochs, s1, method=args.method)
        evaluate_samples(trainer, eval_dict, writer, analytic_samples, num_epochs, sampling_steps, method=args.method)
    else:
        print("eval_dict not provided, can't produce samples.")
