"""
Minimal skeleton for implementing a single‑layer neural network (linear regression)
from scratch with NumPy. Fill in each TODO section with your own code.

Includes boilerplate for:
* Data generation
* Model (forward + **backward** stubs)
* Loss / gradients helpers
* SGD optimizer
* Minibatching
* Training loop outline
* Convergence test & loss plotting utilities
"""
from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# 1. DATA ─────────────────────────────────────────────────────────────────────-
# -----------------------------------------------------------------------------


def generate_data(
    n_samples: int,
    n_features: int,
    *,
    noise_std: float = 0.1,
    seed: int | None = 42,
):
    """Generate a synthetic linear‑regression dataset.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples, 1)
    true_w : ndarray, shape (n_features, 1)
        The hidden ground‑truth weights (useful for convergence tests).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=(n_features, 1))
    y = X @ true_w + rng.normal(scale=noise_std, size=(n_samples, 1))
    return X, y, true_w


# -----------------------------------------------------------------------------
# 2. MODEL ─────────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


class LinearRegression:
    """Simple linear layer with learnable weight matrix W.

    Forward:  ŷ = X @ W
    Backward: ∂L/∂W is supplied by `backward` or external helper.
    """

    def __init__(self, n_features: int, *, seed: int | None = None):
        """TODO: initialise learnable parameters (e.g. W ∈ ℝᵈˣ¹)."""
        # Example starter — delete when you implement your own.
        # rng = np.random.default_rng(seed)
        # self.W = rng.normal(scale=1e-2, size=(n_features, 1))
        pass

    # Forward pass ----------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """TODO: compute predictions given inputs X."""
        raise NotImplementedError

    __call__ = forward  # convenient shorthand

    # Backward pass ----------------------------------------------------------------
    def backward(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """TODO: return ∂L/∂W for the supplied minibatch.

        Hint: For MSE loss, gradient = (2 / n) * Xᵀ·(ŷ − y).
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# 3. LOSS + GRADIENTS ─────────────────────────────────────────────────────────-
# -----------------------------------------------------------------------------


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """TODO: mean‑squared error."""
    raise NotImplementedError


def mse_grad(X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """TODO: gradient of MSE w.r.t. weights.

    You can choose to implement gradients here **or** in `LinearRegression.backward`.
    """
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 4. OPTIMIZER ─────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


def sgd_step(param: np.ndarray, grad: np.ndarray, lr: float = 1e-2):
    """TODO: in‑place SGD parameter update."""
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 5. MINIBATCHING UTIL ─────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


def minibatches(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int | None = None,
):
    """TODO: yield successive mini‑batches of (X_batch, y_batch)."""
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 6. TRAINING LOOP ─────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


def train(
    model: LinearRegression,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-2,
    verbose: bool = True,
):
    """Core training routine.

    Skeleton outline:
    >>> history = []
    >>> for epoch in range(epochs):
    >>>     for X_b, y_b in minibatches(...):
    >>>         # 1) Forward
    >>>         y_pred_b = model(X_b)
    >>>         # 2) Backward (choose one)
    >>>         grad_W = model.backward(X_b, y_pred_b, y_b)
    >>>         # OR: grad_W = mse_grad(X_b, y_pred_b, y_b)
    >>>         # 3) Parameter update
    >>>         sgd_step(model.W, grad_W, lr)
    >>>     # (Optional) evaluate full‑dataset loss, append to history
    >>> return history

    Returns
    -------
    history : list[float]
        Per‑epoch loss values (useful for `plot_history`).
    """
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 7. TEST: CONVERGENCE CHECK ─────────────────────────────────────────────────--
# -----------------------------------------------------------------------------


def test_convergence(
    *,
    n_samples: int = 1_000,
    n_features: int = 5,
    noise_std: float = 0.1,
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-2,
    tol: float = 1e-2,
    seed: int | None = 42,
):
    """Train on a synthetic dataset and assert that \|W_* − W_true\| < tol."""
    X, y, true_w = generate_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        seed=seed,
    )

    model = LinearRegression(n_features, seed=seed)
    train(
        model,
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=False,
    )

    diff = np.linalg.norm(model.W - true_w)
    print(f"‖W_true − W_learned‖ = {diff:.4e}")
    assert diff < tol, (
        f"Model did not converge: distance {diff:.4e} > tol {tol}"
    )


# -----------------------------------------------------------------------------
# 8. PLOTTING ─────────────────────────────────────────────────────────────────-
# -----------------------------------------------------------------------------


def plot_history(history: list[float]):
    """Plot training‑loss curve. Requires Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError("Matplotlib not found — install it or skip plotting.")

    plt.figure()
    plt.plot(history)  # default color, keeps policy guideline
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()


# -----------------------------------------------------------------------------
# 9. SCRIPT ENTRY POINT ────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Hyper‑parameters (edit freely) ---
    N_SAMPLES = 1_000
    N_FEATURES = 5
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-2

    # --- 1) Data ---
    X, y, _ = generate_data(N_SAMPLES, N_FEATURES)

    # --- 2) Model ---
    model = LinearRegression(N_FEATURES)

    # --- 3) Train ---
    history = train(
        model,
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
    )

    # --- 4) Test convergence (optional) ---
    # test_convergence()

    # --- 5) Plot training loss (optional) ---
    # plot_history(history)
