# JAX Neural ODE – Lorenz System

This project implements a **Neural ODE** in **JAX** and trains it on trajectories from the **Lorenz system**, a classical chaotic dynamical system. The goal is to:

- Generate ground-truth trajectories from the Lorenz ODE
- Implement a differentiable ODE solver (RK4) in JAX
- Define a Neural ODE model $$dz/dt = f_θ(z, t)$$ using an MLP
- Train the model to **classify trajectories** generated under different Lorenz parameters

The project is designed for local CPU/MPS execution and focuses on **clear, research-grade, but minimal** code.

---

## 1. Background

### Ordinary Differential Equation (ODE)

We model time-evolving systems via the time derivative:

$$
\frac{dz}{dt} = f(z, t)
$$

Given an initial state \( z_0 \) and a function \( f \), an ODE solver numerically integrates this equation to produce a trajectory \( z(t) \).

### Neural ODE

A Neural ODE replaces that time-derivative function with a neural network:

$$
\frac{dz}{dt} = f_\theta(z, t)
$$

This yields a **continuous-time neural network** whose forward pass is defined by integrating an ODE. Gradients are computed through the ODE solver using automatic differentiation.

### Lorenz System

The Lorenz system is a classical 3D chaotic system:

$$
\begin{aligned}
\dot{x} &= \sigma (y - x), \\
\dot{y} &= x(\rho - z) - y, \\
\dot{z} &= xy - \beta z
\end{aligned}
$$

For different values of the parameter \( \rho \), the system exhibits different qualitative behaviors. In this project, we treat trajectories generated under different parameter settings as different **classes**, and train a Neural ODE to distinguish them.

---

## 2. Project Structure

```text
jax-neural-ode/
│
├── data/
│   └── lorenz_dataset.npy         # Generated Lorenz trajectories + labels
│
├── src/
│   ├── lorenz.py                  # Lorenz ODE and dataset generation
│   ├── solver.py                  # Generic RK4 / ODE solver in JAX
│   ├── model.py                   # Neural ODE model (MLP + ODE dynamics)
│   ├── train.py                   # Training loop and evaluation
│   └── utils.py                   # Plotting, batching, helper functions
│
├── notebooks/
│   └── 01_lorenz_visualization.ipynb  # Trajectory visualization / experiments
│
└── README.md
```

## 3. Installation

It is recommended to use a virtual environment.

**Using uv (recommended)**

```bash
cd jax-neural-ode
uv init .
uv add jax jaxlib optax numpy matplotlib tqdm
```

**Using pip and venv**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

pip install jax jaxlib optax numpy matplotlib tqdm
```

Note: On Apple Silicon (M1/M2), you may want to install the appropriate jaxlib build for CPU/MPS if needed. For this project, CPU execution is sufficient.

## 4. Usage

### 4.1 Generate Lorenz Dataset

This step integrates the Lorenz system and stores trajectories and labels in `data/lorenz_dataset.npy`.

```bash
python -m src.lorenz
```

This will:

- Sample multiple initial conditions and parameter settings
- Integrate the Lorenz ODE with a fixed-step RK4 solver
- Save:
  - `X`: trajectories of shape `(num_samples, time_steps, 3)`
  - `y`: integer labels (e.g., 0/1 for different ρ values)

### 4.2 Train the Neural ODE

```bash
python -m src.train
```

This will:

- Load `data/lorenz_dataset.npy`
- Initialize Neural ODE parameters
- Train using an optimizer from optax (e.g., Adam)
- Periodically log training loss and classification accuracy

### 4.3 Visualize Trajectories

You can use the provided notebook:

```bash
jupyter notebook notebooks/01_lorenz_visualization.ipynb
```

The notebook can visualize:

- Ground-truth Lorenz trajectories
- Neural ODE reconstructed trajectories
- Decision boundaries / logits over time (optional)

## 5. Implementation Details

### ODE Solver

We implement a standard RK4 integrator:

\[
z_{n+1} = z_n + \frac{h}{6} \left(k_1 + 2k_2 + 2k_3 + k_4\right)
\]

with:

\[
k_1 = f(z_n, t_n), \quad
k_2 = f(z_n + \tfrac{h}{2}k_1, t_n + \tfrac{h}{2}), \quad
k_3 = f(z_n + \tfrac{h}{2}k_2, t_n + \tfrac{h}{2}), \quad
k_4 = f(z_n + hk_3, t_n + h).
\]

The solver is implemented in a pure functional, JAX-compatible way and can be wrapped with `jax.jit` and `jax.vmap` for performance.

### Neural ODE Dynamics

The dynamics function has the form:

```text
dz_dt = f_theta(z, t)  # MLP-based vector field
```

where `f_theta` is a small MLP taking the 3D state (and optionally time) and returning a 3D derivative. A simple classification head is applied to the final state \( z(T) \) to produce class logits.

## 6. Roadmap

Planned / possible extensions:

- Add different chaotic systems (e.g., Rossler)
- Compare different solvers (Euler vs RK4 vs adaptive)
- Add regularization terms on the vector field
- Use the Neural ODE as a building block inside larger models
