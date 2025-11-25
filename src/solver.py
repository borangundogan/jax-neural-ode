import jax
import jax.numpy as jnp


def rk4_step_general(f, state, t, dt, params):
    """
    Generic RK4 integration step for any ODE.
    
    f: function(state, t, params) -> derivative
    state: current state vector
    t: time
    dt: timestep
    params: dictionary or any structure containing parameters
    """

    k1 = f(state, t, params)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt, params)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt, params)
    k4 = f(state + dt * k3, t + dt, params)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_ode(f, y0, params, dt, steps):
    """
    Generic ODE integrator using RK4 + JAX scan.

    f: ODE function(state, t, params) -> derivative
    y0: initial state (vector)
    params: parameters for the ODE
    dt: timestep
    steps: number of integration steps

    Returns trajectory of shape (steps, dim)
    """

    def step_fn(carry, t):
        state = carry
        new_state = rk4_step_general(f, state, t * dt, dt, params)
        return new_state, new_state

    # Efficient unrolled loop
    _, traj = jax.lax.scan(step_fn, y0, jnp.arange(steps))

    return traj


def batch_integrate_ode(f, y0_batch, params, dt, steps):
    """
    Batch ODE integration using vmap.

    y0_batch: (batch_size, dim)
    """

    single_solver = lambda y0: integrate_ode(f, y0, params, dt, steps)

    return jax.vmap(single_solver)(y0_batch)

if __name__ == "__main__":
    # simple ODE: dx/dt = -x
    def simple_ode(x, t, params):
        return -x

    y0 = jnp.array([1.0])
    traj = integrate_ode(simple_ode, y0, None, dt=0.1, steps=50)
    print(traj[:5])
