import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_dataset(path="data/lorenz_dataset.npy"):
    data = np.load(path, allow_pickle=True).item()
    X, y = data["X"], data["y"]
    return X, y


def visualize_trajectory(traj, title="Lorenz Trajectory"):
    """
    traj: (steps, 3)
    """
    x = traj[:, 0]
    y_ = traj[:, 1]
    z = traj[:, 2]

    fig = plt.figure(figsize=(12, 6))

    # 3D trajectory plot
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(x, y_, z, lw=1)
    ax.set_title(f"{title} (3D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Individual components
    ax2 = fig.add_subplot(122)
    ax2.plot(x, label="x")
    ax2.plot(y_, label="y")
    ax2.plot(z, label="z")
    ax2.set_title("Time-series components")
    ax2.set_xlabel("Time step")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = load_dataset()

    print("Dataset loaded.")
    print("X shape:", X.shape)  # (200, 1000, 3)
    print("y shape:", y.shape)

    # pick one sample
    idx = 0
    traj = X[idx]

    print("Plotting sample index:", idx, "Label:", y[idx])
    visualize_trajectory(traj, title=f"Sample {idx} (class={y[idx]})")
