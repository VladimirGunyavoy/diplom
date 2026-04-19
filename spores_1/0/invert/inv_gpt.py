import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры маятника
g = 9.81     # Ускорение свободного падения, м/с^2
L = 1.0      # Длина маятника, м
m = 1.0      # Масса, кг (условно)
I = m * L**2 # Момент инерции маятника
max_u = 1.0  # Максимальный управляющий момент (ограничение по модулю)

# Параметры управления (простейший PD-контроллер)
Kp = 10.0
Kd = 5.0

def control_law(theta, omega):
    """
    Простейший PD-закон управления с насыщением.
    Здесь желаемое положение theta_des = 0, желаемая скорость omega_des = 0.
    """
    u = -Kp * theta - Kd * omega
    # Ограничим по модулю
    u = np.clip(u, -max_u, max_u)
    return u

def inverted_pendulum_equations(t, y):
    """
    Правые части уравнений обратного маятника:
    y[0] = theta (угол), y[1] = omega (угловая скорость).
    """
    theta, omega = y
    u = control_law(theta, omega)
    # dtheta/dt = omega
    # domega/dt = (g/L)*sin(theta) + (1/I)*u
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta) + (1.0 / I) * u
    return [dtheta_dt, domega_dt]

# Зададим 10 разных начальных условий
initial_conditions = [
    (0.2 * i, 0.5 * ((-1)**i)) for i in range(4, 5)
]

# Диапазон времени моделирования
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 300)

plt.figure(figsize=(8, 6))

for i in initial_conditions:
    plt.plot(i[0], i[1], 'o', label='start')


for i, ic in enumerate(initial_conditions, start=1):
    sol = solve_ivp(
        inverted_pendulum_equations, t_span, ic, t_eval=t_eval, 
        vectorized=False, rtol=1e-6, atol=1e-8
    )
    theta_vals = sol.y[0]
    omega_vals = sol.y[1]
    plt.plot(theta_vals, omega_vals, 
             linestyle='--',
             label=f"Trajectory {i}")

plt.title("Inverted Pendulum Phase Portrait")
plt.xlabel("Theta (rad)")
plt.ylabel("Omega (rad/s)")
plt.legend(loc="best")  # Легенда на английском
plt.grid(True)
plt.show()
