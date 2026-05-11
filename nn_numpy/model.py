import numpy as np

from .layers import Dense
from .activations import sigmoid, tanh, tanh_derivative
from .losses import binary_crossentropy, binary_crossentropy_derivative


class NeuralNetwork:
    """
    A simple 2-layer neural network for binary classification.

    Architecture:
        input_dim -> hidden_dim (tanh) -> 1 output (sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

        # Layers
        self.layer1 = Dense(input_dim=input_dim, output_dim=hidden_dim)
        self.layer2 = Dense(input_dim=hidden_dim, output_dim=1)

        # Caches for forward pass (pre-activations)
        self._z1 = None  # pre-activation for layer1
        self._a1 = None  # activation after layer1 (tanh)
        self._z2 = None  # pre-activation for layer2
        self._a2 = None  # output after sigmoid

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of the network.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, input_dim).

        Returns
        -------
        np.ndarray
            Output probabilities of shape (n_samples, 1).
        """
        assert X.ndim == 2 and X.shape[1] == self.layer1.input_dim, (
            f"NeuralNetwork.forward expected input shape (*,{self.layer1.input_dim}), got {X.shape}"
        )

        # First layer: linear -> tanh
        self._z1 = self.layer1.forward(X)
        self._a1 = tanh(self._z1)

        # Second layer: linear -> sigmoid
        self._z2 = self.layer2.forward(self._a1)
        self._a2 = sigmoid(self._z2)
        return self._a2

    def backward(self, y_true: np.ndarray) -> None:
        """
        Backward pass of the network. Updates weights in-place.

        Parameters
        ----------
        y_true : np.ndarray
            True labels, shape (n_samples, 1) or (n_samples,).
        """
        # Gradient of loss w.r.t. output (a2)
        dL_da2 = binary_crossentropy_derivative(y_true, self._a2)  # (n_samples, 1)

        # For sigmoid, derivative w.r.t. z2 is:
        # dL/dz2 = dL/da2 * da2/dz2
        # Since sigmoid_derivative(z2) = a2 * (1 - a2), we can reuse a2 here.
        da2_dz2 = self._a2 * (1.0 - self._a2)
        dL_dz2 = dL_da2 * da2_dz2  # elementwise

        # Backprop through second dense layer
        dL_da1 = self.layer2.backward(dL_dz2, learning_rate=self.learning_rate)

        assert self._z1 is not None, "Forward pass must be called before backward"

        # tanh derivative on z1
        dz1 = tanh_derivative(self._z1)
        dL_dz1 = dL_da1 * dz1

        # Backprop through first dense layer
        _ = self.layer1.backward(dL_dz1, learning_rate=self.learning_rate)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.

        Returns
        -------
        float
            Scalar loss.
        """
        return binary_crossentropy(y_true, y_pred)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        verbose: bool = True,
    ) -> list[float]:
        """
        Train the network using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, input_dim).
        y : np.ndarray
            True labels of shape (n_samples, 1) or (n_samples,).
        epochs : int
            Number of training epochs.
        verbose : bool
            If True, print loss every 100 epochs.

        Returns
        -------
        list[float]
            List of loss values per epoch.
        """
        losses = []
        for epoch in range(1, epochs + 1):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)

            # Backward pass (updates weights)
            self.backward(y)

            if verbose and (epoch % 100 == 0 or epoch == 1):
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

        return losses

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for given inputs.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        threshold : float
            Threshold to convert probabilities to 0/1.

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1), shape (n_samples, 1).
        """
        probs = self.forward(X)
        return (probs >= threshold).astype(float)
    

# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Dummy dataset: 10 samples, 2 features
    X = np.random.randn(10, 2)
    y = (np.sum(X, axis=1) > 0).astype(float).reshape(-1, 1)  # arbitrary labels

    model = NeuralNetwork(input_dim=2, hidden_dim=4, learning_rate=0.1)
    losses = model.fit(X, y, epochs=50, verbose=True)

    y_pred = model.predict(X)
    print("Predictions:\n", y_pred.T)
    print("True labels:\n", y.T)