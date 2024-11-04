# Experiments: Consistency Models for Posterior Approximation

## Setup

We provide an Apptainer definition file `cmpe-env-cuda.def`, that installs all necessary dependencies in an Apptainer container.
If no GPU support is required, the `cmpe-env.def` creates a container without CUDA support.
The container can be built using the following command:

```bash
$ sudo apptainer build cmpe-env.sif cmpe-env[-cuda].def
```

After the container is build, the contained Python can be used in the following way:

```bash
apptainer exec --bind /path/to/cmp path/to/cmpe-env.sif python <filename>
```

Depending on your file system structure, the `--bind` option might not be necessary.

A JupyterLab server in the environment can be started with:

```bash
apptainer exec --bind /path/to/cmp path/to/cmpe-env.sif jupyter lab --no-browser
```

## Experiments 1-3

The low-dimensional experiments are located in `experiments/benchmarks`. Note that for producing a reference posterior for GMM, a working Stan installation (accessible via `cmdstanpy`) is required.

## Experiment 4

The Bayesian Denoising experiment is located in `experiments/bayesian_denoising`. See the corresponding README for details on how to run the experiment.

## Experiment 5

The tumor model experiment is located in `experiments/tumor_model`. The experiment builds on the PyABC implementation ([Link](https://pyabc.readthedocs.io/en/latest/examples/multiscale_agent_based.html)), which contains details about the simulator and data.
