import numpy as np

from nn_numpy.activations import (
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
)


def _numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + eps
        f_plus = f(x.copy())

        x[idx] = orig - eps
        f_minus = f(x.copy())

        x[idx] = orig
        grad[idx] = (f_plus[idx] - f_minus[idx]) / (2 * eps)
        it.iternext()
    return grad


def test_sigmoid_matches_numerical_gradient():
    x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    analytical = sigmoid_derivative(x)
    numerical = _numerical_gradient(sigmoid, x)
    assert np.allclose(analytical, numerical, atol=1e-4)


def test_tanh_matches_numerical_gradient():
    x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    analytical = tanh_derivative(x)
    numerical = _numerical_gradient(tanh, x)
    assert np.allclose(analytical, numerical, atol=1e-4)


def test_relu_matches_numerical_gradient_away_from_kink():
    x = np.array([[-2.0, -0.5, 0.5, 2.0]])
    analytical = relu_derivative(x)
    numerical = _numerical_gradient(relu, x)
    assert np.allclose(analytical, numerical, atol=1e-4)


def test_sigmoid_output_range():
    x = np.array([[-10.0, 0.0, 10.0]])
    y = sigmoid(x)
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_tanh_output_range():
    x = np.array([[-10.0, 0.0, 10.0]])
    y = tanh(x)
    assert np.all((y >= -1.0) & (y <= 1.0))
