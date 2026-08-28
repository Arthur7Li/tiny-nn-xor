import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from nn_numpy.datasets import load_circles, load_moons
from nn_numpy.model import NeuralNetwork
from nn_numpy.visualize import plot_decision_boundary


DATASET_LOADERS = {
    "circles": load_circles,
    "moons": load_moons,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the from-scratch NumPy MLP on a nonlinear 2D toy dataset "
            "(circles or moons) and plot its decision boundary."
        )
    )
    parser.add_argument(
        "--dataset", type=str, default="moons", choices=sorted(DATASET_LOADERS),
        help="Toy dataset to train on (default: moons).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of data points to generate (default: 200).",
    )
    parser.add_argument(
        "--noise", type=float, default=0.1,
        help="Standard deviation of Gaussian noise added to the dataset (default: 0.1).",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=16,
        help="Number of hidden units (default: 16).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3000,
        help="Number of training epochs (default: 3000).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.5,
        help="Learning rate for gradient descent (default: 0.5).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save plots in (default: current directory).",
    )
    parser.add_argument(
        "--no-save-plot", action="store_true",
        help="Skip saving plots to disk.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively (blocks until closed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # For reproducibility of weight initialization (dataset generation uses its own
    # independent RNG seeded via --seed, see nn_numpy.datasets).
    np.random.seed(args.seed)

    # 1. Load data
    load_fn = DATASET_LOADERS[args.dataset]
    X, y = load_fn(n_samples=args.n_samples, noise=args.noise, seed=args.seed)

    # 2. Initialize model
    model = NeuralNetwork(
        input_dim=2, hidden_dim=args.hidden_dim, learning_rate=args.learning_rate
    )

    # 3. Train
    print(
        f"Training for {args.epochs} epochs on '{args.dataset}' "
        f"(n_samples={args.n_samples}, noise={args.noise}, "
        f"hidden_dim={args.hidden_dim}, lr={args.learning_rate}, seed={args.seed})..."
    )
    losses = model.fit(X, y, epochs=args.epochs, verbose=True)

    # 4. Evaluate
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"Training accuracy on {args.dataset}: {accuracy * 100:.1f}%")

    if not args.no_save_plot:
        os.makedirs(args.output_dir, exist_ok=True)

    # 5. Plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training loss on {args.dataset} (final acc={accuracy * 100:.0f}%)")
    plt.grid(True)
    plt.tight_layout()

    if not args.no_save_plot:
        loss_path = os.path.join(args.output_dir, f"train_{args.dataset}.png")
        plt.savefig(loss_path)
        print(f"Saved loss curve to {loss_path}")

    # 6. Plot decision boundary
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_decision_boundary(
        model, X, y, resolution=300, ax=ax,
        title=f"{args.dataset.capitalize()} decision boundary (acc={accuracy * 100:.0f}%)",
    )
    fig.tight_layout()

    if not args.no_save_plot:
        boundary_path = os.path.join(args.output_dir, f"decision_boundary_{args.dataset}.png")
        fig.savefig(boundary_path)
        print(f"Saved decision boundary to {boundary_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
