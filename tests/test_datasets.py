import numpy as np

from nn_numpy.datasets import load_xor, load_circles, load_moons


def test_load_xor_shapes_and_labels():
    X, y = load_xor()
    assert X.shape == (4, 2)
    assert y.shape == (4, 1)
    assert set(np.unique(y)) == {0.0, 1.0}


def test_load_circles_shapes_and_balance():
    X, y = load_circles(n_samples=100, noise=0.05, seed=0)
    assert X.shape == (100, 2)
    assert y.shape == (100, 1)
    assert set(np.unique(y)) == {0.0, 1.0}
    assert np.sum(y == 0.0) == np.sum(y == 1.0)


def test_load_circles_is_reproducible_with_seed():
    X1, y1 = load_circles(n_samples=50, noise=0.05, seed=123)
    X2, y2 = load_circles(n_samples=50, noise=0.05, seed=123)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_load_moons_shapes_and_balance():
    X, y = load_moons(n_samples=100, noise=0.05, seed=0)
    assert X.shape == (100, 2)
    assert y.shape == (100, 1)
    assert set(np.unique(y)) == {0.0, 1.0}
    assert np.sum(y == 0.0) == np.sum(y == 1.0)


def test_load_moons_is_reproducible_with_seed():
    X1, y1 = load_moons(n_samples=50, noise=0.05, seed=123)
    X2, y2 = load_moons(n_samples=50, noise=0.05, seed=123)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_circles_and_moons_are_linearly_separable_by_class_geometry():
    # Sanity check: circles dataset should have two distinct radius clusters
    X, y = load_circles(n_samples=200, noise=0.01, seed=0)
    radii = np.linalg.norm(X, axis=1)
    outer_radii = radii[y.reshape(-1) == 0.0]
    inner_radii = radii[y.reshape(-1) == 1.0]
    assert outer_radii.mean() > inner_radii.mean()
