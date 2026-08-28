import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from nn_numpy.datasets import load_xor
from nn_numpy.model import NeuralNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny from-scratch NumPy MLP on the XOR problem."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=8,
        help="Number of hidden units (default: 8, matches the v0.2 100%%-accuracy run).",
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="Number of training epochs (default: 5000).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1,
        help="Learning rate for gradient descent (default: 0.1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save the loss-curve plot in (default: current directory).",
    )
    parser.add_argument(
        "--plot-name", type=str, default="train_run.png",
        help="Filename for the saved loss-curve plot (default: train_run.png).",
    )
    parser.add_argument(
        "--no-save-plot", action="store_true",
        help="Skip saving the loss-curve plot to disk.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the loss-curve plot interactively (blocks until closed). "
             "Off by default so the script also runs headless/in CI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # For reproducibility
    np.random.seed(args.seed)

    # 1. Load data
    X, y = load_xor()

    # 2. Initialize model
    model = NeuralNetwork(
        input_dim=2, hidden_dim=args.hidden_dim, learning_rate=args.learning_rate
    )

    # 3. Train
    print(
        f"Training for {args.epochs} epochs on XOR "
        f"(hidden_dim={args.hidden_dim}, lr={args.learning_rate}, seed={args.seed})..."
    )
    losses = model.fit(X, y, epochs=args.epochs, verbose=True)

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
    plt.title(
        f"Training loss on XOR (hidden_dim={args.hidden_dim}, "
        f"lr={args.learning_rate}, final acc={accuracy * 100:.0f}%)"
    )
    plt.grid(True)
    plt.tight_layout()

    if not args.no_save_plot:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, args.plot_name)
        plt.savefig(save_path)
        print(f"Saved loss curve to {save_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
