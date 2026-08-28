import numpy as np

from nn_numpy.losses import binary_crossentropy, binary_crossentropy_derivative


def _numerical_gradient_wrt_ypred(y_true, y_pred, eps=1e-6):
    grad = np.zeros_like(y_pred)
    it = np.nditer(y_pred, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = y_pred[idx]

        y_pred[idx] = orig + eps
        loss_plus = binary_crossentropy(y_true, y_pred.copy()) * y_true.size

        y_pred[idx] = orig - eps
        loss_minus = binary_crossentropy(y_true, y_pred.copy()) * y_true.size

        y_pred[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def test_bce_gradient_matches_numerical_gradient():
    y_true = np.array([[0.0], [1.0], [1.0], [0.0]])
    y_pred = np.array([[0.2], [0.7], [0.6], [0.4]])

    analytical = binary_crossentropy_derivative(y_true, y_pred)
    numerical = _numerical_gradient_wrt_ypred(y_true, y_pred)

    assert np.allclose(analytical, numerical, atol=1e-3)


def test_bce_loss_is_zero_for_perfect_predictions():
    y_true = np.array([[0.0], [1.0]])
    y_pred = np.array([[1e-12], [1.0 - 1e-12]])
    loss = binary_crossentropy(y_true, y_pred)
    assert loss < 1e-6


def test_bce_loss_is_positive_for_wrong_predictions():
    y_true = np.array([[0.0], [1.0]])
    y_pred = np.array([[0.9], [0.1]])
    loss = binary_crossentropy(y_true, y_pred)
    assert loss > 1.0
