import numpy as np

from nn_numpy.layers import Dense


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


def test_dense_forward_output_shape():
    np.random.seed(0)
    layer = Dense(input_dim=3, output_dim=5)
    x = np.random.randn(4, 3)
    out = layer.forward(x)
    assert out.shape == (4, 5)


def test_dense_backward_gradients_match_numerical():
    np.random.seed(0)
    layer = Dense(input_dim=3, output_dim=2)
    x = np.random.randn(4, 3)
    dout = np.random.randn(4, 2)
    lr = 1e-3
    n_samples = x.shape[0]

    W_before = layer.W.copy()
    b_before = layer.b.copy()

    layer.forward(x)
    layer.backward(dout, learning_rate=lr)

    dW_analytical = (W_before - layer.W) * n_samples / lr
    db_analytical = (b_before - layer.b) * n_samples / lr

    def loss_fn():
        y = x @ layer.W + layer.b
        return np.sum(dout * y)

    layer.W = W_before.copy()
    layer.b = b_before.copy()
    dW_numerical = _numerical_gradient_2d(loss_fn, layer.W)

    layer.W = W_before.copy()
    layer.b = b_before.copy()
    db_numerical = _numerical_gradient_2d(loss_fn, layer.b)

    assert np.allclose(dW_analytical, dW_numerical, atol=1e-3)
    assert np.allclose(db_analytical, db_numerical, atol=1e-3)


def test_dense_default_init_is_xavier_like():
    layer = Dense(input_dim=100, output_dim=10)
    expected_scale = np.sqrt(1.0 / 100)
    assert np.isclose(layer.W.std(), expected_scale, rtol=0.5)
