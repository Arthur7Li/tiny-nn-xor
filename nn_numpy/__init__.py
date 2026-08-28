"""
nn_numpy: a tiny from-scratch neural network library built with NumPy.

Public API
----------
NeuralNetwork          Simple 2-layer MLP (tanh hidden, sigmoid output) with
                        manual forward/backward passes and gradient descent.
Dense                  Fully connected layer used inside NeuralNetwork.
load_xor               Classic 4-point XOR dataset.
load_circles           Two concentric circles (nonlinear 2D toy dataset).
load_moons             Two interleaving half-moons (nonlinear 2D toy dataset).
plot_decision_boundary Visualize a trained model's decision boundary in 2D.

Example
-------
>>> from nn_numpy import NeuralNetwork, load_xor
>>> X, y = load_xor()
>>> model = NeuralNetwork(input_dim=2, hidden_dim=8, learning_rate=0.1)
>>> _ = model.fit(X, y, epochs=5000, verbose=False)
>>> model.predict(X).ravel()
array([0., 1., 1., 0.])
"""

from .model import NeuralNetwork
from .layers import Dense
from .datasets import load_xor, load_circles, load_moons
from .visualize import plot_decision_boundary

__version__ = "0.3.0"

__all__ = [
    "NeuralNetwork",
    "Dense",
    "load_xor",
    "load_circles",
    "load_moons",
    "plot_decision_boundary",
]
