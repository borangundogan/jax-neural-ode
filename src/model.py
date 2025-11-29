import jax
import jax.numpy as jnp
from typing import List, Dict, NamedTuple

from solver import integrate_ode


# ---------- Basic MLP helpers ----------

def init_mlp_params(key, layer_sizes: List[int]) -> List[Dict[str, jnp.ndarray]]:
    """
    Initialize parameters for an MLP with given layer sizes.
    layer_sizes: [input_dim, hidden1, ..., hiddenN, output_dim]
    Returns a list of dicts with 'W' and 'b'.
    """
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)

    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        # He initialization (scaled normal)
        W = jax.random.normal(k, (n_out, n_in)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros((n_out,))
        params.append({"W": W, "b": b})

    return params


def mlp_apply(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    """
    Apply MLP to a single input vector x.
    Uses tanh nonlinearity for hidden layers, no activation on last layer.
    """
    for i, layer in enumerate(params):
        W, b = layer["W"], layer["b"]
        x = W @ x + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


# ---------- Model param ----------

class ModelParams(NamedTuple):
    dyn_params: List[Dict[str, jnp.ndarray]]  # Neural ODE for MLP
    clf_W: jnp.ndarray                        # classifier weight
    clf_b: jnp.ndarray                        # classifier bias


# ---------- Neural ODE dynamic ----------

def neural_ode_rhs(state: jnp.ndarray, t: float, dyn_params) -> jnp.ndarray:
    """
    Right-hand-side of the Neural ODE:
        dz/dt = f_theta(z)

    Burada zamanı explicit olarak kullanmıyoruz (time-invariant field).
    """
    return mlp_apply(dyn_params, state)


def neural_ode_trajectory(
    params: ModelParams,
    z0: jnp.ndarray,
    dt: float,
    steps: int,
) -> jnp.ndarray:
    """
    Integrate the Neural ODE dynamics starting from z0.

    Returns trajectory of shape (steps, dim).
    """
    traj = integrate_ode(neural_ode_rhs, z0, params.dyn_params, dt, steps)
    return traj


def apply_model(
    params: ModelParams,
    z0: jnp.ndarray,
    dt: float,
    steps: int,
) -> (jnp.ndarray, jnp.ndarray):
    """
    Full model forward:
    1) Solve Neural ODE from initial state z0
    2) Take final state z_T
    3) Apply linear classifier to get class logits

    Returns:
        traj: (steps, dim)
        logits: (num_classes,)
    """
    traj = neural_ode_trajectory(params, z0, dt, steps)
    z_T = traj[-1]  # final state

    logits = params.clf_W @ z_T + params.clf_b
    return traj, logits


# ---------- Param init func ----------

def init_model_params(
    key,
    state_dim: int = 3,
    hidden_dims=(32, 32),
    num_classes: int = 2,
) -> ModelParams:
    """
    Initialize full model parameters:
    - MLP for dynamics f_theta(z)
    - Linear classifier on top of final state

    state_dim: dimension of z (3 for Lorenz)
    hidden_dims: tuple of hidden layer sizes
    num_classes: number of trajectory classes
    """
    key_dyn, key_clf = jax.random.split(key, 2)

    layer_sizes = [state_dim, *hidden_dims, state_dim]
    dyn_params = init_mlp_params(key_dyn, layer_sizes)

    # Classifier: from final state (state_dim) -> num_classes
    clf_W = jax.random.normal(key_clf, (num_classes, state_dim)) * jnp.sqrt(
        2.0 / state_dim
    )
    clf_b = jnp.zeros((num_classes,))

    return ModelParams(dyn_params=dyn_params, clf_W=clf_W, clf_b=clf_b)
