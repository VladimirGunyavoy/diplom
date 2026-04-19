import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы
m = 1.0    # масса (кг)
L = 1.0    # длина (м)
g = 9.81   # ускорение свободного падения (м/с²)
b = 0.1    # коэффициент трения

# Параметры управления
Kp = 15.0  # пропорциональный коэффициент
Kd = 5.0   # дифференциальный коэффициент
u_max = 8.0 # максимальное управляющее воздействие

# Уравнение динамики системы
def pendulum_dynamics(t, state):
    theta, theta_dot = state
    theta = theta % (2*np.pi) # Нормализация угла
    
    # Расчет управления с насыщением
    u = -Kp*theta - Kd*theta_dot
    u = np.clip(u, -u_max, u_max)
    
    # Дифференциальные уравнения
    theta_dot_dot = (g/L)*np.sin(theta) - (b/(m*L**2))*theta_dot + u/(m*L**2)
    return [theta_dot, theta_dot_dot]

# Генерация начальных условий
np.random.seed(42)
initial_conditions = []
for _ in range(10):
    theta0 = (np.random.rand() - 0.5)*2*np.pi  # Угол от -π до π
    theta_dot0 = (np.random.rand() - 0.5)*4    # Скорость от -2 до 2
    initial_conditions.append([theta0, theta_dot0])

# Интегрирование уравнений
t_span = [0, 10] # Время моделирования
t_eval = np.linspace(*t_span, 1000)

solutions = []
for ic in initial_conditions:
    sol = solve_ivp(pendulum_dynamics, t_span, ic, t_eval=t_eval, rtol=1e-6)
    solutions.append(sol)

# Построение фазового портрета
plt.figure(figsize=(12, 8))
plt.title('Фазовый портрет обратного маятника\nс ограниченным управлением')
plt.xlabel('Угол θ [рад]')
plt.ylabel('Угловая скорость θ\' [рад/с]')

# Отрисовка траекторий
for sol in solutions:
    theta = np.arctan2(np.sin(sol.y[0]), np.cos(sol.y[0])) # Нормализация угла
    plt.plot(theta, sol.y[1], linewidth=1.5)

# Добавление ограничения управления
xx, yy = np.meshgrid(np.linspace(-np.pi, np.pi, 20), 
                     np.linspace(-4, 4, 20))
u_vals = -Kp*xx - Kd*yy
u_vals = np.clip(u_vals, -u_max, u_max)

plt.contour(xx, yy, np.abs(u_vals), levels=[u_max], colors='red', linestyles='dashed')
plt.text(np.pi/2, 3.5, 'Граница насыщения управления', color='red')

plt.xlim(-np.pi, np.pi)
plt.ylim(-4, 4)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.show()