import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(x, y, seed=1, n_folds=5, scoring="accuracy", normalize=True):
    """C2ST metric [1] using an sklearn MLP classifier with 10 hidden dims per dimension of the samples.
    Code adapted from https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py
    """

    x = np.array(x)
    y = np.array(y)

    if normalize:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / x_std
        y = (y - x_mean) / x_std

    num_dims = x.shape[1]
    assert num_dims == y.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * num_dims, 10 * num_dims),
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

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return tf.convert_to_tensor(np.atleast_1d(scores))
