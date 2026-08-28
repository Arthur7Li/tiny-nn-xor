import numpy as np


def load_xor():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)
    return X, y


def load_circles(n_samples: int = 200, noise: float = 0.08, seed: int | None = None):
    """
    Two concentric circles (classic nonlinear 2D toy dataset).

    Parameters
    ----------
    n_samples : int
        Total number of points (split evenly between the two circles).
    noise : float
        Standard deviation of Gaussian noise added to each point.
    seed : int or None
        Seed for the local random generator (does not affect global NumPy state).

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
    y : np.ndarray, shape (n_samples, 1), labels in {0, 1}
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2

    theta_outer = rng.uniform(0, 2 * np.pi, n_per_class)
    outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)

    theta_inner = rng.uniform(0, 2 * np.pi, n_per_class)
    inner = 0.5 * np.stack([np.cos(theta_inner), np.sin(theta_inner)], axis=1)

    X = np.vstack([outer, inner])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate(
        [np.zeros(n_per_class), np.ones(n_per_class)]
    ).reshape(-1, 1)

    return X, y


def load_moons(n_samples: int = 200, noise: float = 0.12, seed: int | None = None):
    """
    Two interleaving half-moons (classic nonlinear 2D toy dataset).

    Parameters
    ----------
    n_samples : int
        Total number of points (split evenly between the two moons).
    noise : float
        Standard deviation of Gaussian noise added to each point.
    seed : int or None
        Seed for the local random generator (does not affect global NumPy state).

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
    y : np.ndarray, shape (n_samples, 1), labels in {0, 1}
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2

    theta_top = rng.uniform(0, np.pi, n_per_class)
    top = np.stack([np.cos(theta_top), np.sin(theta_top)], axis=1)

    theta_bottom = rng.uniform(0, np.pi, n_per_class)
    bottom = np.stack(
        [1.0 - np.cos(theta_bottom), 1.0 - np.sin(theta_bottom) - 0.5], axis=1
    )

    X = np.vstack([top, bottom])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate(
        [np.zeros(n_per_class), np.ones(n_per_class)]
    ).reshape(-1, 1)

    return X, y
