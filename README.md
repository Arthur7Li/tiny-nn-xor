# Tiny Neural Network from Scratch (NumPy)

Implementing a tiny neural network **from scratch in NumPy** to solve the classic XOR problem.  
This project is designed as an educational intro to:

- How a 2-layer neural network performs a **forward pass**
- How **backpropagation** computes gradients for each parameter
- How a simple **training loop** with gradient descent can learn a nonlinear function

No deep learning frameworks (PyTorch, TensorFlow, Keras) are used—only NumPy.

---

## Project structure

```text
.
├── nn_numpy/
│   ├── __init__.py
│   ├── activations.py    # sigmoid, tanh, ReLU and their derivatives
│   ├── datasets.py       # XOR dataset helper
│   ├── layers.py         # Dense layer (fully connected) with forward/backward
│   ├── losses.py         # MSE and binary cross-entropy losses + derivatives
│   └── model.py          # NeuralNetwork class and training loop
├── train_xor.py          # Entry point: trains the model on XOR
├── requirements.txt
└── README.md
```

At a high level, `train_xor.py` loads the XOR data, builds a small MLP using `nn_numpy.model.NeuralNetwork`, trains it, prints accuracy, and plots the training loss.

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<this-repo-name>.git
cd <this-repo-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate
# or cmd
.\.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train on XOR

```bash
python train_xor.py
```

You should see the loss decreasing over epochs and the final training accuracy reported as 100% for the improved model.

---

## How it works

### Architecture

For XOR, the network uses a simple **2-layer MLP**:

- Input layer: 2 features (the two XOR bits)  
- Hidden layer: `hidden_dim` units (e.g., 4 in v0.1, 8 in v0.2)  
  - Activation: **tanh** in the final version (ReLU in the initial version)  
- Output layer: 1 unit  
  - Activation: **sigmoid**, interpreted as a probability in \([0, 1]\)

Mathematically:

- Hidden pre-activation:  
  \[
  z_1 = X W_1 + b_1
  \]
- Hidden activation:  
  \[
  a_1 = \tanh(z_1) \quad \text{(or ReLU in v0.1)}
  \]
- Output pre-activation:  
  \[
  z_2 = a_1 W_2 + b_2
  \]
- Output activation (prediction):  
  \[
  \hat{y} = \sigma(z_2)
  \]

### Forward and backward passes

- **Forward pass**: `NeuralNetwork.forward(X)` computes these steps and caches intermediate values (`z1`, `a1`, `z2`, `a2`) for backprop.  
- **Loss**:
  - v0.1: mean squared error (MSE)  
  - v0.2: **binary cross-entropy (BCE)**, more appropriate for sigmoid outputs  
- **Backward pass**: `NeuralNetwork.backward(y_true)`:
  - Starts from the derivative of the loss with respect to the output (`dL/dy_pred`)
  - Applies the derivative of the sigmoid and tanh to propagate gradients back to `W2`, `b2`, `W1`, and `b1`
  - Uses the `Dense.backward` method to compute gradients and apply gradient descent updates

All operations are implemented with NumPy array math; no automatic differentiation is used.

---

## Versions and progress

A key goal of this project is to show the **iterative improvement** of a simple neural network:

### v0.1 – First working model (75% accuracy)

- Architecture:  
  - Input → **4 ReLU** hidden units → 1 sigmoid output  
- Loss: **MSE (mean squared error)**  
- Initialization: small random weights, zero biases  
- Result:  
  - Training loss decreases slowly from ~0.25  
  - Final training accuracy: **75%** on XOR (3/4 points correct)  
- Plot: `train_1.png`  
  - Shows loss slowly decreasing but not converging to a near-zero value

### v0.2 – Improved model (100% accuracy)

Refinements:

- Switched hidden activation to **tanh**  
- Kept **sigmoid** for the output layer  
- Switched loss to **binary cross-entropy (BCE)** for sigmoid outputs  
- Updated `Dense` layer to use a **Xavier-style initialization** for more stable gradients  
- Added runtime assertions in `Dense.forward`, `Dense.backward`, and `NeuralNetwork.backward` to check shapes and ensure the forward pass runs before backprop

Result:

- Training loss now decreases significantly and converges  
- Final training accuracy: **100%** on XOR (4/4 points correct)  
- Plot: `train_final.png`  
  - Shows a clear downward trend in loss as the network learns XOR

You can find the corresponding snapshots in Git history:

- `v0.1`: initial MLP implementation with MSE and ReLU  
- `v0.2`: BCE + tanh + improved initialization and assertions

---

## Training curves

Below are the loss curves for the two main versions of the model:

- **v0.1 – 75% accuracy (ReLU + MSE)**  
  ![Training loss for v0.1 (XOR, 75% accuracy)](train_1.png)

- **v0.2 – 100% accuracy (tanh + BCE)**  
  ![Training loss for v0.2 (XOR, 100% accuracy)](train_final.png)

---

## XOR dataset

The XOR dataset is defined in `nn_numpy/datasets.py` as:

- Inputs:

  \[
  X = \{(0,0), (0,1), (1,0), (1,1)\}
  \]

- Labels:

  \[
  y = \{0, 1, 1, 0\}
  \]

The network is trained on all four points as a tiny “batch,” which makes it easy to see the effect of the forward and backward passes step by step.

---

## Possible extensions

If you want to extend this project, good next steps include:

- Visualizing the **decision boundary** in the input space on a grid  
- Swapping out XOR for a slightly larger 2D toy dataset (e.g., circles or moons)  
- Adding support for more layers (deep MLP) or different activations  
- Implementing simple **regularization** (L2 weight decay)  
- Adding lightweight unit tests for `Dense`, activations, and loss functions

---

## Acknowledgements

This project was developed as a learning exercise with the help of:

- **Perplexity’s Learn Mode AI tutor**, for step-by-step guidance on neural network concepts, project structure, and implementation details.
- **GitHub Copilot**, for in-editor suggestions and refinements, including improvements to gradient scaling, weight initialization, and runtime assertions.

Both tools were used as assistants while keeping full understanding and control of the final code and architecture.