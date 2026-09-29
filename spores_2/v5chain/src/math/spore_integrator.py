import numpy as np


class SporeIntegrator:
    """
    2D robot integrator: state = [x, y, theta], control = (u_theta, u_xy).

    u_theta — angular velocity (rotation)
    u_xy    — linear velocity along current heading theta

    x_new     = x + cos(theta) * u_xy * dt
    y_new     = y + sin(theta) * u_xy * dt
    theta_new = theta + u_theta * dt
    """

    def __init__(self):
        self.state = np.zeros(3)

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.state[:] = [x, y, theta]

    def step(self, u, dt: float) -> np.ndarray:
        x, y, theta = self.state
        u_theta, u_xy = u[0], u[1]
        self.state = np.array([
            x + np.cos(theta) * u_xy * dt,
            y + np.sin(theta) * u_xy * dt,
            theta + u_theta * dt,
        ])
        return self.state
