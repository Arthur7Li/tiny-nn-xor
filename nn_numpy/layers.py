import numpy as np


class Dense:
    """
    A fully connected (dense) layer: y = x @ W + b

    Attributes
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output units.
    W : np.ndarray
        Weight matrix of shape (input_dim, output_dim).
    b : np.ndarray
        Bias vector of shape (1, output_dim).
    """

    def __init__(self, input_dim: int, output_dim: int, weight_scale: float | None = None):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Use Xavier-style initialization by default for tanh/sigmoid networks.
        if weight_scale is None:
            weight_scale = np.sqrt(1.0 / input_dim)

        self.W = weight_scale * np.random.randn(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))

        # Cache for backprop
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the dense layer.

        Parameters
        ----------
        x : np.ndarray
            Input of shape (n_samples, input_dim).

        Returns
        -------
        np.ndarray
            Output of shape (n_samples, output_dim).
        """
        assert x.ndim == 2 and x.shape[1] == self.input_dim, (
            f"Dense.forward expected input shape (*,{self.input_dim}), got {x.shape}"
        )
        self._x = x  # cache input for use in backward pass
        return x @ self.W + self.b  # (n_samples, input_dim) @ (input_dim, output_dim)

    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass for the dense layer.

        Parameters
        ----------
        dout : np.ndarray
            Gradient of the loss with respect to this layer's output,
            shape (n_samples, output_dim).
        learning_rate : float
            Learning rate for gradient descent update.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to this layer's input,
            shape (n_samples, input_dim).
        """
        x = self._x  # (n_samples, input_dim)
        assert x is not None, "Dense.backward called before forward"
        assert dout.ndim == 2 and dout.shape[1] == self.output_dim, (
            f"Dense.backward expected dout shape (*,{self.output_dim}), got {dout.shape}"
        )

        n_samples = x.shape[0]

        # Gradients of loss w.r.t. parameters
        # dL/dW = x^T @ dout
        dW = x.T @ dout  # (input_dim, n_samples) @ (n_samples, output_dim)

        # dL/db = sum over samples of dout
        db = np.sum(dout, axis=0, keepdims=True)  # (1, output_dim)

        # Gradient w.r.t. input to pass to previous layer:
        # dL/dx = dout @ W^T
        dx = dout @ self.W.T  # (n_samples, output_dim) @ (output_dim, input_dim)

        # Gradient descent parameter update (average over batch)
        self.W -= learning_rate * dW / n_samples
        self.b -= learning_rate * db / n_samples

        return dx
    

# For testing purposes
if __name__ == "__main__":
    np.random.seed(42)

    # Simple test: 3 samples, 2 input features, 2 output units
    x = np.random.randn(3, 2)
    layer = Dense(input_dim=2, output_dim=2)

    # Fake upstream gradient dout (from next layer / loss)
    dout = np.random.randn(3, 2)

    print("Initial W:\n", layer.W)
    print("Initial b:\n", layer.b)

    out = layer.forward(x)
    print("Forward output:\n", out)

    dx = layer.backward(dout, learning_rate=0.1)
    print("Gradient w.r.t input dx:\n", dx)
    print("Updated W:\n", layer.W)
    print("Updated b:\n", layer.b)