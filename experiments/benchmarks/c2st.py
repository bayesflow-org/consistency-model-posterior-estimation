import numpy as np
from bayesflow.exceptions import ShapeError
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def c2st(
    source_samples,
    target_samples,
    normalize=True,
    seed=123,
    hidden_units_per_dim=16,
):
    """C2ST metric [1] using an sklearn neural network classifier (i.e., MLP).
    Code adapted from https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py

    [1] Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv:1610.06545.

    Parameters
    ----------
    source_samples       : np.ndarray or tf.Tensor
        Source samples (e.g., approximate posterior samples)
    target_samples       : np.ndarray or tf.Tensor
        Target samples (e.g., samples from a reference posterior)
    normalize            : bool, optional, default: True
        Whether the data shall be z-standardized relative to source_samples
    seed                 : int, optional, default: 123
        RNG seed for the MLP and k-fold CV
    hidden_units_per_dim : int, optional, default: 16
        Number of hidden units in the MLP, relative to the input dimensions.
        Example: source samples are 5D, hidden_units_per_dim=16 -> 80 hidden units per layer

    Returns
    -------
    c2st_score  :  float
        The resulting C2ST score

    """

    x = np.array(source_samples)
    y = np.array(target_samples)

    num_dims = x.shape[1]
    if not num_dims == y.shape[1]:
        raise ShapeError(
            f"source_samples and target_samples can have different number of observations (1st dim)"
            f"but must have the same dimensionality (2nd dim)"
            f"found: source_samples {source_samples.shape[1]}, target_samples {target_samples.shape[1]}"
        )

    if normalize:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / x_std
        y = (y - x_mean) / x_std

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(hidden_units_per_dim * num_dims, hidden_units_per_dim * num_dims),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((x, y))
    target = np.concatenate(
        (
            np.zeros((x.shape[0],)),
            np.ones((y.shape[0],)),
        )
    )

    # simple train test split
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed)
    clf.fit(x_train, y_train)
    c2st_score = clf.score(x_test, y_test)

    return c2st_score