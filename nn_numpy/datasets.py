import numpy as np


def load_xor() -> tuple[np.ndarray, np.ndarray]:
    """
    Return the classic 2D XOR dataset.

    Returns
    -------
    X : np.ndarray
        Inputs of shape (4, 2).
    y : np.ndarray
        Binary labels of shape (4, 1).
    """
    # Four possible input combinations
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    # XOR labels: 1 if exactly one of the inputs is 1, else 0
    y = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ],
        dtype=float,
    )

    return X, y


if __name__ == "__main__":
    X, y = load_xor()
    print("X:\n", X)
    print("y:\n", y)