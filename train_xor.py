import numpy as np
import matplotlib.pyplot as plt

from nn_numpy.datasets import load_xor
from nn_numpy.model import NeuralNetwork


def main() -> None:
    # For reproducibility
    np.random.seed(42)

    # 1. Load data
    X, y = load_xor()

    # 2. Initialize model
    # 2 inputs -> 8 hidden units -> 1 output
    model = NeuralNetwork(input_dim=2, hidden_dim=8, learning_rate=0.1)

    # 3. Train
    epochs = 5000
    print(f"Training for {epochs} epochs on XOR...")
    losses = model.fit(X, y, epochs=epochs, verbose=True)

    # 4. Evaluate
    y_pred_probs = model.forward(X)
    y_pred = model.predict(X, threshold=0.5)

    accuracy = np.mean(y_pred == y)
    print("\nFinal predictions (probabilities):\n", y_pred_probs)
    print("Final predictions (thresholded):\n", y_pred)
    print("True labels:\n", y)
    print(f"Training accuracy on XOR: {accuracy * 100:.1f}%")

    # 5. Plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss on XOR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()