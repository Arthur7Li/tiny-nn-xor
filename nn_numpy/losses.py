import numpy as np


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Binary cross-entropy (BCE) loss for targets in {0,1} and predictions in (0,1).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels, shape (n_samples, 1) or (n_samples,).
    y_pred : np.ndarray
        Predicted probabilities, same shape as y_true.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        Scalar BCE loss.
    """
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    # BCE: -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )
    loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(loss)


def binary_crossentropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Derivative of BCE loss with respect to y_pred.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted probabilities.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Gradient dL/dy_pred, same shape as y_pred.
    """
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    # Note: the loss itself is averaged across samples, but we keep the
    # gradient as the derivative of the summed loss. The parameter updates
    # are averaged inside Dense.backward.
    grad = -(y_true / y_pred - (1.0 - y_true) / (1.0 - y_pred))
    return grad


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

    loss = binary_crossentropy(y_true, y_pred)
    grad = binary_crossentropy_derivative(y_true, y_pred)

    print("y_true:\n", y_true)
    print("y_pred:\n", y_pred)
    print("BCE loss:", loss)
    print("dL/dy_pred:\n", grad)