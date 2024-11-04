import time

import numpy as np
from bayesflow.computational_utilities import maximum_mean_discrepancy
from bayesflow.diagnostics import plot_recovery, plot_sbc_ecdf
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from benchmark_architectures import benchmark_cm, benchmark_fmpe, benchmark_npe
from metrics import c2st
from reference_posteriors.two_moons.two_moons_lueckmann_numpy import analytic_posterior_numpy

algorithm_modules = {"npe": benchmark_npe, "fmpe": benchmark_fmpe, "cm": benchmark_cm}


def run_benchmark(benchmark_name, approximator_name, tag="baseline", simulation_budget=10000):
    name = f"{benchmark_name}/{simulation_budget}/{tag}/{approximator_name}"
    writer = SummaryWriter(f"tensorboard/{name}")

    trainer = algorithm_modules[approximator_name].get_trainer(benchmark_name, name)

    if approximator_name == "cm":
        EPOCHS = 3000
        if simulation_budget >= 10000:
            BATCH_SIZE = 256
        else:
            BATCH_SIZE = 64
    else:
        EPOCHS = 300
        BATCH_SIZE = 32
    N_POSTERIOR_SAMPLES = 1000
    N_TEST_DATASETS = 100
    if approximator_name == "fmpe":
        N_TEST_DATASETS = 2

    writer.add_text(
        "info",
        f"EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, N_POSTERIOR_SAMPLES={N_POSTERIOR_SAMPLES}, "
        f"N_TEST_DATASETS={N_TEST_DATASETS}",
    )

    train_data = trainer.generative_model(simulation_budget)

    tic = time.time()
    h = trainer.train_offline(train_data, EPOCHS, BATCH_SIZE)
    toc = time.time()
    training_time = toc - tic
    print(f"[{approximator_name}] Training took {training_time:.8f}s.")
    writer.add_scalar(f"training_time", training_time, global_step=EPOCHS)

    # Generate test simulations: (theta, x)
    test_sims = trainer.configurator(trainer.generative_model(N_TEST_DATASETS))

    # Sample from the approximate posterior
    tic = time.time()
    samples = trainer.amortizer.sample(test_sims, n_samples=N_POSTERIOR_SAMPLES)
    toc = time.time()
    time_per_posterior_sample = (toc - tic) / (N_POSTERIOR_SAMPLES * N_TEST_DATASETS)
    print(f"[{approximator_name}] {time_per_posterior_sample:.8f}s per posterior sample.")
    writer.add_scalar(f"time_per_sample", time_per_posterior_sample, global_step=EPOCHS)

    # Parameter recovery
    true_theta = test_sims["parameters"]
    f = plot_recovery(post_samples=samples, prior_samples=true_theta)
    writer.add_figure(f"recovery", f, global_step=EPOCHS)
    plt.show()

    # simulation-based calibration
    f = plot_sbc_ecdf(post_samples=samples, prior_samples=true_theta, difference=True, stacked=True)
    writer.add_figure(f"sbc", f, global_step=EPOCHS)
    plt.show()

    # compare with reference posterior if TwoMoons (others not implemented yet)
    if benchmark_name == "two_moons":
        test_sims = {"direct_conditions": np.array([[0.0, 0.0]]).astype(np.float32)}
        samples = trainer.amortizer.sample(test_sims, n_samples=N_POSTERIOR_SAMPLES)
        analytic_samples = analytic_posterior_numpy([0, 0], N_POSTERIOR_SAMPLES, rng=np.random.default_rng(seed=1234))
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.axis("equal")
        ax.scatter(
            analytic_samples[:, 0],
            analytic_samples[:, 1],
            color="orange",
            alpha=0.4,
            s=3,
            label="True",
        )
        ax.scatter(samples[:, 0], samples[:, 1], color="green", alpha=0.4, s=3, label="Model")
        ax.set_title(f"{approximator_name}")
        ax.grid(alpha=0.2)
        ax.legend()
        # plt.show()

        writer.add_figure(f"samples", f, global_step=EPOCHS)

        mmd = maximum_mean_discrepancy(analytic_samples.astype(np.float32), samples)
        c2st_accuracy = c2st(analytic_samples, samples)

        writer.add_scalar(f"mmd", mmd.numpy(), global_step=EPOCHS)
        writer.add_scalar(f"c2st", c2st_accuracy[0].numpy(), global_step=EPOCHS)

        print(f"MMD: {mmd:.4f}, C2ST: {c2st_accuracy}")
        writer.close()
    elif benchmark_name == "gaussian_mixture":
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.axis("equal")
        ax.scatter(
            samples[0, :, 0],
            samples[0, :, 1],
            color="orange",
            alpha=0.4,
            s=3,
            label="Model",
        )
        # ax.scatter(samples[:, 0], samples[:, 1], color="green", alpha=0.4, s=3, label="Model")
        ax.set_title(f"{approximator_name}")
        ax.grid(alpha=0.2)
        ax.legend()
        # plt.show()

        writer.add_figure(f"samples", f, global_step=EPOCHS)
        writer.close()


if __name__ == "__main__":
    # benchmarks = ['gaussian_linear', 'gaussian_linear_uniform', 'two_moons', 'gaussian_mixture', 'sir']
    # algorithms = ['npe', 'fmpe', 'cm']
    #
    # for b in benchmarks:
    #     for a in algorithms:
    #         run_benchmark(b, a)

    # run_benchmark("two_moons", "npe", "base", 10000)
    # run_benchmark("two_moons", "cm", "base", 10000)
    run_benchmark("two_moons", "fmpe", "base", 10000)

    # run_benchmark("gaussian_mixture", "npe", "base", 10000)
    # run_benchmark("gaussian_mixture", "fmpe")
    # run_benchmark("gaussian_mixture", "cm", "base", 10000)
