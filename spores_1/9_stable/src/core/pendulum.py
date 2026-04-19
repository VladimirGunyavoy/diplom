import numpy as np
from scipy.linalg import expm

class PendulumSystem:
    """
    Класс, описывающий систему маятника.
    Позволяет выполнять линеаризацию и дискретизацию в произвольном состоянии.
    """
    def __init__(self, 
                 dt=0.1, 
                 damping=0.1, 
                 max_control=1,
                 initial_position=(0, 0),
                 goal_position=(np.pi, 0)):
        # Параметры маятника
        self.g = 9.81
        self.l = 2.0
        self.m = 1.0
        self.dt = dt
        self.damping = damping
        self.max_control = max_control
        self.position = np.array(initial_position)
        self.goal_position = np.array(goal_position)



        
    def get_control_bounds(self):
        return np.array([-self.max_control, self.max_control])
        

    def dynamics_continuous(self, state, control):
        """
        Нелинейная динамика маятника с трением: θ'' = -g/l * sin(θ) - c*θ' + u
        """
        theta, theta_dot = state
        theta_ddot = -self.g/self.l * np.sin(theta) - self.damping * theta_dot + control
        return np.array([theta_dot, theta_ddot])

    def linearize_continuous(self, state):
        """
        Линеаризация нелинейной динамики маятника в произвольном состоянии.
        Возвращает непрерывные матрицы A и B.
        """
        theta_0, _ = state
        
        A_cont = np.array([
            [0.0, 1.0],
            [-self.g / self.l * np.cos(theta_0), -self.damping]
        ])
        
        B_cont = np.array([
            [0.0],
            [1.0]
        ])
        
        return A_cont, B_cont

    def discretize(self, A_cont, B_cont, dt):
        """
        Дискретизация непрерывной системы с помощью матричной экспоненты.
        """
        n = A_cont.shape[0]
        m = B_cont.shape[1]
        
        augmented_matrix = np.zeros((n + m, n + m))
        augmented_matrix[0:n, 0:n] = A_cont
        augmented_matrix[0:n, n:n+m] = B_cont
        
        phi = expm(augmented_matrix * dt)
        
        A_discrete = phi[0:n, 0:n]
        B_discrete = phi[0:n, n:n+m]
        
        return A_discrete, B_discrete

    def discrete_step(self, state, control=0, dt=None):
        """
        Один шаг дискретной системы с перелинеаризацией в текущем состоянии.
        """
        if dt is None:
            dt = self.dt

        if len(state) != 2:
            raise ValueError(f"State must be a 2D vector, got {len(state)}D vector {state}")

        # 1. Линеаризация в текущем состоянии
        A_cont, B_cont = self.linearize_continuous(state)
        
        # 2. Дискретизация полученных матриц
        A_d, B_d = self.discretize(A_cont, B_cont, dt)
        
        # 3. Шаг линеаризованной системы
        # Важно: мы шагаем из текущего состояния state, но используя локальную
        # линеаризованную модель. Это первый порядок аппроксимации.
        state_col = np.asarray(state).reshape(-1, 1)
        control_col = np.asarray(control).reshape(-1, 1)
        
        # Вычисляем смещение от точки линеаризации (которая равна state)
        # В данном случае смещение равно нулю, т.к. мы линеаризуем вокруг state
        # x_k_minus_x0 = state_col - state_col = 0
        
        # Поскольку мы моделируем приращение, нужно добавить нелинейную динамику
        # x(k+1) ≈ x(k) + f(x(k), u(k)) * dt
        # Но для ЛИНЕАРИЗОВАННОЙ модели шаг выглядит так:
        # x(k+1) = Ad*x(k) + Bd*u(k)
        # Эта формула предполагает, что x(k) - это отклонение от точки равновесия,
        # вокруг которой была линеаризация. В нашем случае это сама точка state.
        # Более точная формула шага:
        # x(k+1) = x_op + Ad @ (x(k) - x_op) + Bd @ (u(k) - u_op) + f(x_op, u_op)*dt
        # Где x_op = state. Тогда x(k) - x_op = 0.
        # Получаем: x(k+1) = x_op + f(x_op, u_op)*dt
        # Это просто интегрирование Эйлера.
        
        # Метод с матричной экспонентой дает более точный результат для ЛИНЕЙНОЙ системы.
        # x(k+1) = Ad @ x(k) + Bd @ u(k)
        # Здесь x(k) - это ПОЛНЫЙ вектор состояния.
        next_state = A_d @ state_col + B_d @ control_col
        
        return next_state.flatten()
    
    def simulate_discrete(self, x0, controls, steps):
        """Симуляция системы на несколько шагов с перелинеаризацией на каждом шаге."""
        states = [np.asarray(x0).copy()]
        x = np.asarray(x0).copy()
        
        for i in range(steps):
            u = controls[i] if i < len(controls) else controls[-1]
            x = self.discrete_step(x, u)
            states.append(x.copy())
            
        return np.array(states)
    
    def sample_controls(self, N):
        """Генерирует N равномерно распределенных значений управления."""
        return np.linspace(-self.max_control, self.max_control, N)

# Пример использования
if __name__ == "__main__":
    pendulum = PendulumSystem(dt=0.1, damping=0.1)
    
    print("Система маятника с перелинеаризацией на каждом шаге.")
    
    # --- Пример 1: Симуляция из нижнего положения (стабильно) ---
    print("\n--- Симуляция из нижнего положения [0.1, 0] ---")
    x0_stable = np.array([0.1, 0.0])
    controls_zero = [0.0] * 50
    states_stable = pendulum.simulate_discrete(x0_stable, controls_zero, 50)
    
    for i in range(5):
        print(f"Шаг {i:2d}: θ={states_stable[i, 0]:6.3f}, θ̇={states_stable[i, 1]:6.3f}")
    print("...")
    for i in range(45, 51):
        print(f"Шаг {i:2d}: θ={states_stable[i, 0]:6.3f}, θ̇={states_stable[i, 1]:6.3f}")

    # --- Пример 2: Симуляция из верхнего положения (нестабильно) ---
    print("\n--- Симуляция из верхнего положения [pi - 0.1, 0] ---")
    x0_unstable = np.array([np.pi - 0.1, 0.0])
    states_unstable = pendulum.simulate_discrete(x0_unstable, controls_zero, 50)

    for i in range(5):
        print(f"Шаг {i:2d}: θ={states_unstable[i, 0]:6.3f}, θ̇={states_unstable[i, 1]:6.3f}")
    print("...")
    for i in range(45, 51):
        print(f"Шаг {i:2d}: θ={states_unstable[i, 0]:6.3f}, θ̇={states_unstable[i, 1]:6.3f}")
        
    # --- Пример 3: Линеаризация и дискретизация в разных точках ---
    print("\n--- Матрицы для разных состояний (dt=0.1) ---")
    state_down = np.array([0.0, 0.0])
    state_up = np.array([np.pi, 0.0])
    
    Ac_d, Bc_d = pendulum.linearize_continuous(state_down)
    Ad_d, Bd_d = pendulum.discretize(Ac_d, Bc_d, pendulum.dt)
    e_d = np.linalg.eigvals(Ad_d)
    print(f"Нижняя точка: |λ|={np.abs(e_d[0]):.4f}, {np.abs(e_d[1]):.4f} (стабильно)")

    Ac_u, Bc_u = pendulum.linearize_continuous(state_up)
    Ad_u, Bd_u = pendulum.discretize(Ac_u, Bc_u, pendulum.dt)
    e_u = np.linalg.eigvals(Ad_u)
    print(f"Верхняя точка: |λ|={np.abs(e_u[0]):.4f}, {np.abs(e_u[1]):.4f} (нестабильно)")