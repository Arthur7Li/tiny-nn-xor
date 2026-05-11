import numpy as np


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error (MSE) loss.

    Parameters
    ----------
    y_true : np.ndarray
        True target values, shape (n_samples, 1) or (n_samples,).
    y_pred : np.ndarray
        Predicted values, same shape as y_true.

    Returns
    -------
    float
        Scalar MSE loss.
    """
    # Ensure both are 2D column vectors for simplicity
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    return float(np.mean((y_true - y_pred) ** 2))


def mse_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of the MSE loss with respect to y_pred.

    Parameters
    ----------
    y_true : np.ndarray
        True target values, shape (n_samples, 1) or (n_samples,).
    y_pred : np.ndarray
        Predicted values, same shape as y_true.

    Returns
    -------
    np.ndarray
        Gradient dL/dy_pred with same shape as y_pred.
    """
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    n_samples = y_true.shape[0]
    # d/dy_pred (1/n * sum (y_true - y_pred)^2) = 2/n * (y_pred - y_true)
    return (2.0 / n_samples) * (y_pred - y_true)


# For testing purposes
if __name__ == "__main__":
    y_true = np.array([[0.0],
                       [1.0],
                       [1.0],
                       [0.0]])
    y_pred = np.array([[0.2],
                       [0.8],
                       [0.4],
                       [0.1]])

    loss = mse_loss(y_true, y_pred)
    grad = mse_loss_derivative(y_true, y_pred)

    print("y_true:\n", y_true)
    print("y_pred:\n", y_pred)
    print("MSE loss:", loss)
    print("dL/dy_pred:\n", grad)