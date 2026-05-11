import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Output array with sigmoid applied elementwise.
    """
    # Clip to avoid overflow in exp for very large positive/negative values
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the sigmoid activation with respect to x.

    Parameters
    ----------
    x : np.ndarray
        Input array (pre-activation).

    Returns
    -------
    np.ndarray
        Derivative d(sigmoid)/dx evaluated at x.
    """
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Output array with ReLU applied elementwise.
    """
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the ReLU activation with respect to x.

    Parameters
    ----------
    x : np.ndarray
        Input array (pre-activation).

    Returns
    -------
    np.ndarray
        Derivative d(ReLU)/dx: 1 where x > 0, else 0.
    """
    grad = np.zeros_like(x)
    grad[x > 0] = 1.0
    return grad

# For testing purposes
if __name__ == "__main__":
    x = np.array([[-1.0, 0.0, 1.0],
                  [2.0, -3.0, 4.0]])

    print("x:\n", x)
    print("sigmoid(x):\n", sigmoid(x))
    print("sigmoid_derivative(x):\n", sigmoid_derivative(x))
    print("relu(x):\n", relu(x))
    print("relu_derivative(x):\n", relu_derivative(x))