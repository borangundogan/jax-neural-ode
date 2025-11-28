import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import model 
from model import init_model_params, apply_model, ModelParams


# ----------------- Dataset helpers -----------------
def load_lorenz_dataset(path: str = "data/lorenz_dataset.npy") -> Tuple[jnp.ndarray, jnp.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    X_np, y_np = data["X"], data["y"]  # X: (N, T, 3), y: (N,)
    X = jnp.array(X_np, dtype=jnp.float32)
    y = jnp.array(y_np, dtype=jnp.int32)
    return X, y


def train_val_split(
    X: jnp.ndarray,
    y: jnp.ndarray,
    val_frac: float = 0.2,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N = X.shape[0]
    idx = jax.random.permutation(key, N)
    split = int(N * (1.0 - val_frac))
    train_idx = idx[:split]
    val_idx = idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    return X_train, y_train, X_val, y_val


def batch_iter(
    X: jnp.ndarray,
    y: jnp.ndarray,
    batch_size: int,
    key: jax.Array,
):
    N = X.shape[0]
    idx = jax.random.permutation(key, N)

    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]


# ----------------- Loss & metrics -----------------
def forward_batch(
    params: ModelParams,
    X_batch: jnp.ndarray,
    dt: float,
    steps: int,
):
    """
    X_batch: (B, T, 3)
    We use the first state as initial condition for the Neural ODE.
    """
    z0_batch = X_batch[:, 0, :]  # (B, 3)

    # vmap over batch dimension
    traj_batch, logits_batch = jax.vmap(
        lambda z0: apply_model(params, z0, dt, steps)
    )(z0_batch)

    return traj_batch, logits_batch


def loss_fn(
    params: ModelParams,
    X_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    dt: float,
    steps: int,
    traj_weight: float = 0.1,
):
    """
    Combined loss:
      - classification loss (cross entropy)
      - optional trajectory reconstruction loss (MSE) with small weight
    """
    traj_pred, logits = forward_batch(params, X_batch, dt, steps)

    # Classification loss
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, y_batch
    ).mean()

    # Trajectory reconstruction loss (compare to ground-truth Lorenz traj)
    traj_true = X_batch  # (B, T, 3)
    traj_mse = jnp.mean((traj_pred - traj_true) ** 2)

    total_loss = ce_loss + traj_weight * traj_mse

    # Accuracy for logging
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y_batch)

    metrics = {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "traj_mse": traj_mse,
        "acc": acc,
    }
    return total_loss, metrics


# ----------------- Training step (JIT) -----------------
def make_train_step(optimizer, dt: float, steps: int, traj_weight: float = 0.1):
    """Closure creating a jit-compiled training step."""

    def _train_step(params: ModelParams, opt_state, X_batch, y_batch):
        (loss_value, metrics), grads = jax.value_and_grad(
            lambda p, xb, yb: loss_fn(p, xb, yb, dt, steps, traj_weight),
            has_aux=True,
        )(params, X_batch, y_batch)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, metrics

    return jax.jit(_train_step)


# ----------------- Evaluation (no grad) -----------------
def evaluate(
    params: ModelParams,
    X: jnp.ndarray,
    y: jnp.ndarray,
    dt: float,
    steps: int,
    batch_size: int = 32,
):
    N = X.shape[0]
    num_batches = max(1, N // batch_size)

    def _eval_batch(Xb, yb):
        _, logits = forward_batch(params, Xb, dt, steps)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == yb)
        return ce, acc

    ce_sum = 0.0
    acc_sum = 0.0

    for i in range(0, N, batch_size):
        Xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        ce, acc = _eval_batch(Xb, yb)
        ce_sum += float(ce) * Xb.shape[0]
        acc_sum += float(acc) * Xb.shape[0]

    return ce_sum / N, acc_sum / N


# ----------------- Main training loop -----------------
def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    lr = 1e-3
    traj_weight = 0.1

    # Load dataset
    X, y = load_lorenz_dataset("data/lorenz_dataset.npy")
    steps = X.sh
