import jax
import jax.numpy as jnp

import numpy as np

def lorenz_ode(state, t, params):
    """
    Lorenz system ODE.
    
    state: (3,) array -> [x, y, z]
    t: scalar time (unused here but kept for general ODE API)
    params: dict with keys: sigma, rho, beta
    """
    x, y, z = state
    sigma = params["sigma"]
    rho = params["rho"]
    beta = params["beta"]

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return jnp.array([dx, dy, dz])

def rk4_step(f, state, t, dt, params):
    """
    One RK4 integration step.
    f: ODE function(state, t, params) -> derivative
    state: current state vector
    t: current time
    dt: step size
    params: parameters for the ODE
    """
    k1 = f(state, t, params)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt, params)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt, params)
    k4 = f(state + dt * k3, t + dt, params)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# @jax.jit
def integrate_lorenz(state0, params, dt, steps):
    """
    Integrate Lorenz system using RK4.
    
    state0: initial state (3,)
    params: dict with sigma, rho, beta
    dt: timestep
    steps: number of integration steps

    Returns trajectory of shape (steps, 3)
    """
    def step_fn(carry, t):
        state = carry
        new_state = rk4_step(lorenz_ode, state, t * dt, dt, params)
        return new_state, new_state

    # Run scan (efficient JAX loop)
    _, traj = jax.lax.scan(step_fn, state0, jnp.arange(steps))

    return traj

def generate_lorenz_dataset(
    num_samples=200,
    dt=0.01,
    steps=1000,
    rho_values=(28.0, 35.0),
    seed=0
):
    """
    Generate Lorenz trajectories for two different rho values.
    
    num_samples: total number of trajectories (split across classes)
    dt: time step
    steps: number of integration steps per trajectory
    rho_values: (rho_class0, rho_class1)
    """

    key = jax.random.PRNGKey(seed)

    half = num_samples // 2

    X_list = []
    y_list = []

    for class_id, rho in enumerate(rho_values):
        for i in range(half):

            # random initial state (uniform in [-1,1])
            key, subkey = jax.random.split(key)
            state0 = jax.random.uniform(subkey, shape=(3,), minval=-1.0, maxval=1.0)

            params = {"sigma": 10.0, "rho": rho, "beta": 8/3}

            traj = integrate_lorenz(state0, params, dt, steps)   # (steps, 3)

            X_list.append(np.array(traj))   # convert JAX array to numpy for saving
            y_list.append(class_id)

    X = np.stack(X_list, axis=0)   # (num_samples, steps, 3)
    y = np.array(y_list)           # (num_samples,)

    return X, y


def save_dataset(path="data/lorenz_dataset.npy", **kwargs):
    X, y = generate_lorenz_dataset(**kwargs)
    np.save(path, {"X": X, "y": y})
    print(f"Dataset saved to {path}. Shapes: X={X.shape}, y={y.shape}")


# CLI entrypoint
if __name__ == "__main__":
    save_dataset(
        path="data/lorenz_dataset.npy",
        num_samples=200,
        dt=0.01,
        steps=1000,
        rho_values=(28.0, 35.0),
        seed=0
    )
