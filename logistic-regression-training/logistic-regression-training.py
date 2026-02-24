import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N, D = X.shape

    # Initialize parameters to zeros
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        # --- Forward pass ---
        z = X @ w + b              # logits, shape (N,)
        p = _sigmoid(z)            # probabilities, shape (N,)

        # --- Compute gradients ---
        error = p - y              # shape (N,)
        dw = (1 / N) * (X.T @ error)   # shape (D,)
        db = (1 / N) * np.sum(error)    # scalar

        # --- Update parameters ---
        w = w - lr * dw
        b = b - lr * db

    return w, float(b)
    pass