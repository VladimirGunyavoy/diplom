import numpy as np
from scipy.linalg import expm

class PendulumSystem:
    """Класс, описывающий систему маятника с точной дискретизацией"""
    def __init__(self, dt=0.1, damping=0.1, verbose=False):
        # Параметры маятника
        self.g = 9.81  # ускорение свободного падения
        self.l = 2.0   # длина маятника
        self.m = 1.0   # масса
        self.dt = dt   # шаг дискретизации
        self.damping = damping  # коэффициент демпфирования (трения)
        self.verbose = verbose  # флаг вывода отладочной информации
        
        # Вычисляем матрицы системы
        self.compute_continuous_matrices()
        self.compute_discrete_matrices_exact()
        
    def dynamics(self, state, control):
        """
        Нелинейная динамика маятника с трением: θ'' = -g/l * sin(θ) - c*θ' + u
        state = [θ, θ']
        control = u (управление)
        c = damping (коэффициент демпфирования)
        """
        theta, theta_dot = state
        theta_ddot = -self.g/self.l * np.sin(theta) - self.damping * theta_dot + control
        return np.array([theta_dot, theta_ddot])
    
    @property
    def A_2(self):
        return self.A_discrete
    
    @property
    def B_2(self):
        return self.B_discrete
    
    
    def compute_continuous_matrices(self):
        """
        Вычисляет непрерывные матрицы A и B для линеаризованной системы
        
        Линеаризация системы ẋ = f(x,u) вокруг точки равновесия (0,0):
        
        f(x,u) = [θ']
                 [-g/l * sin(θ) - c*θ' + u]
        
        A = ∂f/∂x|_(0,0) = [∂f₁/∂θ  ∂f₁/∂θ']  = [0     1  ]
                           [∂f₂/∂θ  ∂f₂/∂θ']    [-g/l  -c]
        
        B = ∂f/∂u|_(0,0) = [∂f₁/∂u] = [0]
                           [∂f₂/∂u]   [1]
        """
        
        # Матрица состояния A
        self.A = np.array([
            [0.0,           1.0],           # ∂θ̇/∂θ = 0, ∂θ̇/∂θ̇ = 1
            [-self.g/self.l, -self.damping] # ∂θ̈/∂θ = -g/l*cos(0) = -g/l, ∂θ̈/∂θ̇ = -c
        ])
        
        # Матрица управления B  
        self.B = np.array([
            [0.0],    # ∂θ̇/∂u = 0
            [1.0]     # ∂θ̈/∂u = 1
        ])
        
        if self.verbose:
            print("Непрерывные матрицы вычислены:")
            print(f"A = \n{self.A}")
            print(f"B = \n{self.B}")
        
    def compute_discrete_matrices_exact(self):
        """
        Точная дискретизация через матричную экспоненту
        
        Для системы ẋ = Ax + Bu решение имеет вид:
        x(t) = e^(At) * x(0) + ∫₀ᵗ e^(A(t-τ)) * B * u(τ) dτ
        
        При постоянном управлении u(τ) = u на интервале [0, dt]:
        x(dt) = e^(A*dt) * x(0) + A⁻¹(e^(A*dt) - I) * B * u
        
        Но более элегантно использовать расширенную матрицу:
        
        Φ = exp([A*dt  B*dt]) = [e^(A*dt)              A⁻¹(e^(A*dt) - I)*B]
               [0     0   ]   [0                      I                    ]
        
        Откуда: A_d = Φ[0:n, 0:n], B_d = Φ[0:n, n:n+m]
        """
        
        n = self.A.shape[0]  # размерность состояния (2)
        m = self.B.shape[1]  # размерность управления (1)
        
        # Создаем расширенную матрицу размера (n+m) × (n+m)
        # Φ_matrix = [A*dt  B*dt]
        #            [0     0   ]
        Phi_matrix = np.zeros((n + m, n + m))
        Phi_matrix[0:n, 0:n] = self.A * self.dt       # Верхний левый блок: A*dt
        Phi_matrix[0:n, n:n+m] = self.B * self.dt     # Верхний правый блок: B*dt
        # Нижний блок остается нулевым
        
        print(f"\nРасширенная матрица (A*dt, B*dt):")
        print(f"Φ_matrix = \n{Phi_matrix}")
        
        # Вычисляем матричную экспоненту
        Phi = expm(Phi_matrix)
        
        print(f"\nМатричная экспонента exp(Φ_matrix):")
        print(f"Φ = \n{Phi}")
        
        # Извлекаем дискретные матрицы
        self.A_discrete = Phi[0:n, 0:n]           # exp(A*dt)
        self.B_discrete = Phi[0:n, n:n+m]         # A⁻¹(exp(A*dt) - I)*B
        
        print(f"\nДискретные матрицы:")
        print(f"A_discrete = \n{self.A_discrete}")
        print(f"B_discrete = \n{self.B_discrete}")
        
        # Проверка: альтернативный расчет для сравнения
        self._verify_discrete_matrices()
        
    def _verify_discrete_matrices(self):
        """
        Проверка корректности дискретизации альтернативным методом
        """
        if not self.verbose:
            return
            
        print(f"\n=== ПРОВЕРКА ДИСКРЕТИЗАЦИИ ===")
        
        # Метод 1: Прямое вычисление exp(A*dt)
        A_d_direct = expm(self.A * self.dt)
        
        # Метод 2: Для случая, когда A обратима
        # B_d = A⁻¹(exp(A*dt) - I) * B
        try:
            A_inv = np.linalg.inv(self.A)
            exp_A_dt = expm(self.A * self.dt)
            I = np.eye(self.A.shape[0])
            B_d_formula = A_inv @ (exp_A_dt - I) @ self.B
            
            print(f"A_discrete (прямой расчет): \n{A_d_direct}")
            print(f"Разность A_discrete: {np.max(np.abs(self.A_discrete - A_d_direct))}")
            
            print(f"B_discrete (формула): \n{B_d_formula}")
            print(f"Разность B_discrete: {np.max(np.abs(self.B_discrete - B_d_formula))}")
            
        except np.linalg.LinAlgError:
            print("Матрица A не обратима, используем только метод через расширенную матрицу")
            
        # Проверка свойств дискретной системы
        eigenvals_cont = np.linalg.eigvals(self.A)
        eigenvals_disc = np.linalg.eigvals(self.A_discrete)
        
        print(f"\nСобственные значения:")
        print(f"Непрерывная система λ: {eigenvals_cont}")
        print(f"Дискретная система λ_d: {eigenvals_disc}")
        print(f"Проверка λ_d = exp(λ*dt): {np.exp(eigenvals_cont * self.dt)}")
        
    def discrete_step(self, state, control):
        """
        Один шаг дискретной системы
        x[k+1] = A_d * x[k] + B_d * u[k]
        """
        state = np.array(state).reshape(-1, 1) if len(np.array(state).shape) == 1 else state
        control = np.array(control).reshape(-1, 1) if len(np.array(control).shape) == 1 else control
        
        next_state = self.A_discrete @ state + self.B_discrete @ control
        return next_state.flatten() if next_state.shape[1] == 1 else next_state
    
    def simulate_discrete(self, x0, controls, steps):
        """
        Симуляция дискретной системы на несколько шагов
        """
        states = [x0]
        x = x0.copy()
        
        for i in range(steps):
            u = controls[i] if i < len(controls) else controls[-1]
            x = self.discrete_step(x, u)
            states.append(x.copy())
            
        return np.array(states)
    
    def print_system_analysis(self):
        """Полный анализ системы"""
        print("="*60)
        print("АНАЛИЗ СИСТЕМЫ МАЯТНИКА")
        print("="*60)
        print(f"Параметры: g={self.g} м/с², l={self.l} м, dt={self.dt} с, c={self.damping}")
        print(f"Характерная частота: ω₀ = √(g/l) = {np.sqrt(self.g/self.l):.3f} рад/с")
        print(f"Период колебаний: T = 2π/ω₀ = {2*np.pi/np.sqrt(self.g/self.l):.3f} с")
        
        # Анализ устойчивости
        eigenvals_cont = np.linalg.eigvals(self.A)
        eigenvals_disc = np.linalg.eigvals(self.A_discrete)
        
        print(f"\nУСТОЙЧИВОСТЬ:")
        if self.damping == 0:
            print(f"Непрерывная система: λ = ±{np.sqrt(self.g/self.l):.3f}j (нейтрально устойчива)")
        else:
            # Для системы с трением: λ = -c/2 ± √((c/2)² - g/l)
            c = self.damping
            omega0_sq = self.g/self.l
            if c*c/4 < omega0_sq:  # Слабое демпфирование
                omega_d = np.sqrt(omega0_sq - c*c/4)
                print(f"Непрерывная система: λ = -{c/2:.3f} ± {omega_d:.3f}j (затухающие колебания)")
            else:  # Сильное демпфирование
                lambda1 = -c/2 + np.sqrt(c*c/4 - omega0_sq)
                lambda2 = -c/2 - np.sqrt(c*c/4 - omega0_sq)
                print(f"Непрерывная система: λ = {lambda1:.3f}, {lambda2:.3f} (апериодическое затухание)")
        
        print(f"Дискретная система: |λ_d| = {np.abs(eigenvals_disc)}")
        print(f"Устойчивость дискретной: {'ДА' if np.all(np.abs(eigenvals_disc) <= 1) else 'НЕТ'}")
        
        # Дополнительная информация о демпфировании
        if self.damping > 0:
            print(f"\nДЕМПФИРОВАНИЕ:")
            print(f"Коэффициент демпфирования: c = {self.damping}")
            print(f"Критическое демпфирование: c_crit = 2√(g/l) = {2*np.sqrt(self.g/self.l):.3f}")
            print(f"Относительное демпфирование: ζ = c/c_crit = {self.damping/(2*np.sqrt(self.g/self.l)):.3f}")
            if self.damping < 2*np.sqrt(self.g/self.l):
                print("Режим: слабое демпфирование (колебательный)")
            else:
                print("Режим: сильное демпфирование (апериодический)")

# Пример использования
if __name__ == "__main__":
    # Создаем систему маятника с демпфированием
    pendulum = PendulumSystem(dt=0.05, damping=0.1, verbose=False)
    
    # Полный анализ системы
    pendulum.print_system_analysis()
    
    print(f"\nA_discrete = \n{pendulum.A_discrete}")
    print(f"B_discrete = \n{pendulum.B_discrete}")
    
    # Пример симуляции с демпфированием
    print(f"\n=== СИМУЛЯЦИЯ С ДЕМПФИРОВАНИЕМ (c={pendulum.damping}) ===")
    
    x0 = np.array([0.5, 0.0])  # начальное отклонение 0.5 рад
    controls = [0.0] * 50      # без управления
    
    states = pendulum.simulate_discrete(x0, controls, 20)
    
    print("Эволюция состояния (θ, θ̇):")
    for i, state in enumerate(states):
        print(f"Шаг {i:2d}: θ={state[0]:6.3f}, θ̇={state[1]:6.3f}")
    
    # Сравнение с системой без демпфирования
    print(f"\n=== СРАВНЕНИЕ С СИСТЕМОЙ БЕЗ ДЕМПФИРОВАНИЯ ===")
    
    pendulum_no_damping = PendulumSystem(dt=0.05, damping=0.0, verbose=False)
    states_no_damping = pendulum_no_damping.simulate_discrete(x0, controls, 20)
    
    print("Без демпфирования (c=0):")
    for i, state in enumerate(states_no_damping):
        print(f"Шаг {i:2d}: θ={state[0]:6.3f}, θ̇={state[1]:6.3f}")
    
    # Демонстрация сильного демпфирования
    print(f"\n=== СИЛЬНОЕ ДЕМПФИРОВАНИЕ ===")
    
    pendulum_strong = PendulumSystem(dt=0.05, damping=1.0, verbose=False)
    pendulum_strong.print_system_analysis()
    
    states_strong = pendulum_strong.simulate_discrete(x0, controls, 20)
    
    print("Эволюция состояния с сильным демпфированием:")
    for i, state in enumerate(states_strong):
        print(f"Шаг {i:2d}: θ={state[0]:6.3f}, θ̇={state[1]:6.3f}")