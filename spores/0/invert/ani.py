import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from matplotlib.patches import Circle

# ------ НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ------
# Параметры маятника
g = 9.8       # ускорение свободного падения (м/с^2)
L = 1.0       # длина маятника (м)
m = 1.0       # масса маятника (кг)
b = 0.1       # коэффициент затухания

# Параметры PID-регулятора
Kp = 30.0     # пропорциональный коэффициент
Kd = 10.0     # дифференциальный коэффициент
Ki = 0.1      # интегральный коэффициент

# Начальные условия
theta_0 = np.pi - 0.3  # начальный угол (рад), почти вверху
omega_0 = 0.0          # начальная угловая скорость (рад/с)
integral_0 = 0.0       # начальное значение интеграла ошибки

# Параметры симуляции
sim_time = 3       # время симуляции (с)
dt = 0.02              # шаг по времени (с)

# ------ ФУНКЦИЯ ДЛЯ РАСЧЕТА УПРАВЛЯЮЩЕГО ВОЗДЕЙСТВИЯ ------
def calculate_control(theta, omega, integral):
    """
    Рассчитывает управляющее воздействие для стабилизации в верхнем положении.
    
    Args:
        theta (float): текущий угол отклонения маятника
        omega (float): текущая угловая скорость маятника
        integral (float): накопленный интеграл ошибки
        
    Returns:
        tuple: (управляющее воздействие, ошибка)
    """
    # Расчет ошибки (отклонение от верхнего положения π)
    error = (theta - np.pi) % (2*np.pi)
    if error > np.pi:
        error -= 2*np.pi
    
    # Расчет управляющего сигнала по PID закону
    control = Kp * error + Kd * omega + Ki * integral
    return control, error

# ------ ФУНКЦИЯ ДЛЯ РАСЧЕТА ПРОИЗВОДНЫХ ------
def pendulum_dynamics(t, state):
    """
    Определяет динамику маятника с управлением для стабилизации.
    
    Args:
        t (float): текущее время
        state (list): текущее состояние [theta, omega, integral]
        
    Returns:
        list: производные состояния [d_theta, d_omega, d_integral]
    """
    theta, omega, integral = state
    
    control, error = calculate_control(theta, omega, integral)
    
    # Производная угла - это угловая скорость
    d_theta = omega
    
    # Производная угловой скорости с учетом управляющего воздействия
    d_omega = g/L * np.sin(theta) - b * omega - control / (m * L**2)
    
    # Производная интеграла ошибки
    d_integral = error
    
    return [d_theta, d_omega, d_integral]

# ------ СОЗДАНИЕ ВРЕМЕННОЙ СЕТКИ И РЕШЕНИЕ УРАВНЕНИЙ ------
# Временной интервал для симуляции
t_span = (0, sim_time)
t_eval = np.arange(0, sim_time, dt)

# Решение системы дифференциальных уравнений
print("Выполняется расчет динамики системы...")
solution = solve_ivp(
    pendulum_dynamics,
    t_span,
    [theta_0, omega_0, integral_0],
    t_eval=t_eval,
    method='RK45'
)

# Расчет управляющего воздействия для каждого момента времени
print("Расчет управляющего воздействия...")
controls = []
errors = []
for i in range(len(t_eval)):
    theta = solution.y[0][i]
    omega = solution.y[1][i]
    integral = solution.y[2][i]
    control, error = calculate_control(theta, omega, integral)
    controls.append(control)
    errors.append(error)

controls = np.array(controls)
errors = np.array(errors)

# ------ СОЗДАНИЕ АНИМАЦИИ ------
print("Создание анимации...")
# Создание фигуры с двумя графиками
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

# Верхний график для анимации маятника
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title('Inverted pendulum stabilisation', fontsize=16)

# Нижний график для отображения угла и управляющего воздействия
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, t_span[1])
y_min = min(min(controls), min(errors)) - 1
y_max = max(max(controls), max(errors)) + 1
y_range = max(abs(y_min), abs(y_max))
ax2.set_ylim(-y_range, y_range)
ax2.set_xlabel('Время (с)', fontsize=12)
ax2.set_ylabel('Значение', fontsize=12)
ax2.grid(True)
ax2.set_title('Угол отклонения и управляющее воздействие', fontsize=14)

# Линии для графиков
pendulum_line, = ax1.plot([], [], '-', lw=3, color='blue')
pendulum_mass = Circle((0, 0), radius=0.1, fc='red', ec='black', zorder=3)
ax1.add_patch(pendulum_mass)
pivot = ax1.plot([0], [0], 'ko', markersize=10, zorder=2)[0]
time_template = 'Время = %.1f с'
time_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=12)

# Линии для графика угла и управляющего воздействия
angle_line, = ax2.plot([], [], label='Отклонение от верхнего положения', color='blue', lw=2)
control_line, = ax2.plot([], [], label='Управляющее воздействие', color='red', lw=2)
angle_point, = ax2.plot([], [], 'o', color='blue', markersize=6)
control_point, = ax2.plot([], [], 'o', color='red', markersize=6)
ax2.legend(loc='upper right', fontsize=12)

# Нулевая линия и целевое положение
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.plot([0, 0], [0, -L], 'k-', alpha=0.2)  # Нижнее положение
ax1.plot([0, 0], [0, L], 'r--', alpha=0.7)   # Верхнее (целевое) положение

# Функция инициализации анимации
def init():
    pendulum_line.set_data([], [])
    pendulum_mass.center = (0, 0)
    time_text.set_text('')
    angle_line.set_data([], [])
    control_line.set_data([], [])
    angle_point.set_data([], [])
    control_point.set_data([], [])
    return pendulum_line, pendulum_mass, time_text, angle_line, control_line, angle_point, control_point

# Функция обновления анимации
def update(i):
    theta = solution.y[0][i]
    
    # Расчет координат конца маятника
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    
    # Обновление положения маятника
    pendulum_line.set_data([0, x], [0, y])
    pendulum_mass.center = (x, y)
    # time_text.set_text(time_template % solution.t[i])
    
    # Обновление графика угла и управляющего воздействия
    angle_line.set_data(solution.t[:i+1], errors[:i+1])
    control_line.set_data(solution.t[:i+1], controls[:i+1])
    angle_point.set_data([solution.t[i]], [errors[i]])
    control_point.set_data([solution.t[i]], [controls[i]])
    
    return pendulum_line, pendulum_mass, time_text, angle_line, control_line, angle_point, control_point

# Создание анимации
frames = len(t_eval)
interval = dt * 1000  # интервал между кадрами в мс
ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=interval)

plt.tight_layout()

# Для сохранения анимации в файл, раскомментируйте следующие строки:
from matplotlib.animation import writers
writer = writers['ffmpeg'](fps=int(1/dt))
ani.save('stabilized_pendulum.mp4', writer=writer, dpi=100)

print("Отображение анимации...")
plt.show()