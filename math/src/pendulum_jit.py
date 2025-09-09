import numpy as np
from numba import njit
from functools import lru_cache


@lru_cache(maxsize=1000)
def cached_acceleration(angle, angular_velocity, length, damping, gravity):
    """Кешированная функция для расчета ускорения маятника"""
    return -(gravity / length) * np.sin(angle) - damping * angular_velocity


@njit
def pendulum_acceleration(angle, angular_velocity, length, damping, gravity):
    """JIT-функция для расчета ускорения маятника"""
    return -(gravity / length) * np.sin(angle) - damping * angular_velocity


@njit
def rk45_step(state, dt, length, damping, gravity):
    """JIT-функция для одного шага метода Рунге-Кутта 4-5 порядка"""
    angle, angular_velocity = state[0], state[1]
    
    # Расчет ускорения
    acceleration = pendulum_acceleration(angle, angular_velocity, length, damping, gravity)
    
    # Коэффициенты k1-k6 для RK45
    k1_angle = angular_velocity
    k1_velocity = acceleration
    
    k2_angle = angular_velocity + dt * k1_velocity / 4
    k2_velocity = pendulum_acceleration(angle + dt * k1_angle / 4, 
                                       angular_velocity + dt * k1_velocity / 4, 
                                       length, damping, gravity)
    
    k3_angle = angular_velocity + dt * (3 * k1_velocity + 9 * k2_velocity) / 32
    k3_velocity = pendulum_acceleration(angle + dt * (3 * k1_angle + 9 * k2_angle) / 32,
                                       angular_velocity + dt * (3 * k1_velocity + 9 * k2_velocity) / 32,
                                       length, damping, gravity)
    
    k4_angle = angular_velocity + dt * (1932 * k1_velocity - 7200 * k2_velocity + 7296 * k3_velocity) / 2197
    k4_velocity = pendulum_acceleration(angle + dt * (1932 * k1_angle - 7200 * k2_angle + 7296 * k3_angle) / 2197,
                                       angular_velocity + dt * (1932 * k1_velocity - 7200 * k2_velocity + 7296 * k3_velocity) / 2197,
                                       length, damping, gravity)
    
    k5_angle = angular_velocity + dt * (439 * k1_velocity / 216 - 8 * k2_velocity + 3680 * k3_velocity / 513 - 845 * k4_velocity / 4104)
    k5_velocity = pendulum_acceleration(angle + dt * (439 * k1_angle / 216 - 8 * k2_angle + 3680 * k3_angle / 513 - 845 * k4_angle / 4104),
                                       angular_velocity + dt * (439 * k1_velocity / 216 - 8 * k2_velocity + 3680 * k3_velocity / 513 - 845 * k4_velocity / 4104),
                                       length, damping, gravity)
    
    k6_angle = angular_velocity + dt * (-8 * k1_velocity / 27 + 2 * k2_velocity - 3544 * k3_velocity / 2565 + 1859 * k4_velocity / 4104 - 11 * k5_velocity / 40)
    k6_velocity = pendulum_acceleration(angle + dt * (-8 * k1_angle / 27 + 2 * k2_angle - 3544 * k3_angle / 2565 + 1859 * k4_angle / 4104 - 11 * k5_angle / 40),
                                       angular_velocity + dt * (-8 * k1_velocity / 27 + 2 * k2_velocity - 3544 * k3_velocity / 2565 + 1859 * k4_velocity / 4104 - 11 * k5_velocity / 40),
                                       length, damping, gravity)
    
    # Формула 4-го порядка
    y4 = state + dt * (25 * np.array([k1_angle, k1_velocity]) / 216 + 
                       1408 * np.array([k3_angle, k3_velocity]) / 2565 + 
                       2197 * np.array([k4_angle, k4_velocity]) / 4104 - 
                       np.array([k5_angle, k5_velocity]) / 5)
    
    # Формула 5-го порядка
    y5 = state + dt * (16 * np.array([k1_angle, k1_velocity]) / 135 + 
                       6656 * np.array([k3_angle, k3_velocity]) / 12825 + 
                       28561 * np.array([k4_angle, k4_velocity]) / 56430 - 
                       9 * np.array([k5_angle, k5_velocity]) / 50 + 
                       2 * np.array([k6_angle, k6_velocity]) / 55)
    
    return y5, np.abs(y5 - y4)


@njit
def adaptive_rk45(state, dt, length, damping, gravity, tolerance=1e-6):
    """JIT-функция для адаптивного RK45 с автоматическим выбором шага"""
    current_dt = dt
    
    while True:
        y5, error = rk45_step(state, current_dt, length, damping, gravity)
        
        # Проверка точности
        if np.max(error) < tolerance:
            return y5, current_dt
        
        # Уменьшение шага
        current_dt *= 0.8 * (tolerance / np.max(error)) ** 0.2


@njit
def phase_acceleration_vector(state, length, damping, gravity):
    """
    JIT-функция для расчета вектора фазового ускорения маятника
    
    Args:
        state: вектор состояния [угол, угловая скорость]
        length: длина маятника
        damping: коэффициент затухания
        gravity: ускорение свободного падения
    
    Returns:
        вектор фазового ускорения [угловая скорость, угловое ускорение]
    """
    angle, angular_velocity = state[0], state[1]
    
    # Угловое ускорение (вторая производная угла по времени)
    angular_acceleration = pendulum_acceleration(angle, angular_velocity, length, damping, gravity)
    
    # Вектор фазового ускорения: [dθ/dt, d²θ/dt²]
    return np.array([angular_velocity, angular_acceleration])


@njit
def phase_velocity_vector(state):
    """
    JIT-функция для расчета вектора фазовой скорости
    
    Args:
        state: вектор состояния [угол, угловая скорость]
    
    Returns:
        вектор фазовой скорости [угловая скорость, угловое ускорение]
    """
    angle, angular_velocity = state[0], state[1]
    
    # Для фазовой скорости первая компонента - это угловая скорость
    # Вторая компонента - это угловое ускорение (нужно вычислить)
    # Упрощенная версия без параметров маятника
    angular_acceleration = -9.81 * np.sin(angle) - 0.1 * angular_velocity
    
    return np.array([angular_velocity, angular_acceleration])
