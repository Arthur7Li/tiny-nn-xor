import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from nn_numpy.model import NeuralNetwork
from nn_numpy.datasets import load_xor
from nn_numpy.visualize import plot_decision_boundary


def test_plot_decision_boundary_runs_and_returns_axes():
    np.random.seed(0)
    X, y = load_xor()
    model = NeuralNetwork(input_dim=2, hidden_dim=4, learning_rate=0.1)
    model.fit(X, y, epochs=50, verbose=False)

    ax = plot_decision_boundary(model, X, y, resolution=20)
    assert ax is not None
    assert len(ax.collections) > 0
    plt.close(ax.figure)


def test_plot_decision_boundary_accepts_existing_axes():
    np.random.seed(0)
    X, y = load_xor()
    model = NeuralNetwork(input_dim=2, hidden_dim=4, learning_rate=0.1)
    model.fit(X, y, epochs=50, verbose=False)

    fig, ax = plt.subplots()
    returned_ax = plot_decision_boundary(model, X, y, resolution=20, ax=ax, title="test")
    assert returned_ax is ax
    assert ax.get_title() == "test"
    plt.close(fig)
