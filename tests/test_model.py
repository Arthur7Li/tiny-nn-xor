import numpy as np

from nn_numpy.model import NeuralNetwork
from nn_numpy.datasets import load_xor


def _numerical_gradient_2d(f, param, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + eps
        f_plus = f()

        param[idx] = orig - eps
        f_minus = f()

        param[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def test_full_network_gradient_check_on_random_data():
    np.random.seed(1)
    X = np.random.randn(6, 2)
    y = (np.random.rand(6, 1) > 0.5).astype(float)

    model = NeuralNetwork(input_dim=2, hidden_dim=4, learning_rate=1e-3)

    W1_before = model.layer1.W.copy()
    lr = model.learning_rate
    n_samples = X.shape[0]

    y_pred = model.forward(X)
    model.backward(y)

    dW1_analytical = (W1_before - model.layer1.W) * n_samples / lr

    def loss_fn():
        return model.compute_loss(y, model.forward(X)) * n_samples

    model.layer1.W = W1_before.copy()
    dW1_numerical = _numerical_gradient_2d(loss_fn, model.layer1.W)

    assert np.allclose(dW1_analytical, dW1_numerical, atol=1e-2)


def test_network_learns_xor_to_high_accuracy():
    np.random.seed(42)
    X, y = load_xor()
    model = NeuralNetwork(input_dim=2, hidden_dim=8, learning_rate=0.1)
    model.fit(X, y, epochs=5000, verbose=False)
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    assert accuracy >= 0.99


def test_predict_returns_binary_labels():
    np.random.seed(0)
    X, y = load_xor()
    model = NeuralNetwork(input_dim=2, hidden_dim=4, learning_rate=0.1)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0.0, 1.0})
