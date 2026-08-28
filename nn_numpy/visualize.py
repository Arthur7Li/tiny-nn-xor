import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 300,
    padding: float = 0.5,
    ax=None,
    title: str | None = None,
):
    """
    Plot the model's decision boundary over a 2D input space, along with
    the training points colored by their true label.

    Parameters
    ----------
    model : NeuralNetwork
        A trained network whose `forward` method returns probabilities in
        [0, 1] for 2D inputs.
    X : np.ndarray, shape (n_samples, 2)
        Training inputs.
    y : np.ndarray, shape (n_samples, 1) or (n_samples,)
        True binary labels.
    resolution : int
        Number of grid points per axis. Higher values give a smoother
        boundary but take longer to compute.
    padding : float
        Extra margin added around the data range when building the grid.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. If None, a new figure and axes are created.
    title : str or None
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the decision boundary was drawn on.
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    probs = model.forward(grid).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.contourf(xx, yy, probs, levels=50, cmap="RdBu_r", alpha=0.75, vmin=0, vmax=1)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)

    y_flat = np.asarray(y).reshape(-1)
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y_flat, cmap="RdBu_r", edgecolors="black", linewidths=1.2, s=90,
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if title:
        ax.set_title(title)

    return ax
