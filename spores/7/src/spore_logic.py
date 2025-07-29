import numpy as np
from .pendulum import PendulumSystem

class SporeLogic:
    """
    Класс для математической логики споры.
    Работает только в 2D пространстве состояний (x, z).
    Не имеет зависимостей от GUI или визуализации.
    """
    
    def __init__(self, pendulum: PendulumSystem, dt: float, goal_position_2d: np.array, 
                 initial_position_2d: np.array):
        """
        Инициализация логики споры.
        
        Args:
            pendulum: Система маятника для эволюции
            dt: Временной шаг
            goal_position_2d: Целевая позиция в 2D (x, z)
            initial_position_2d: Начальная позиция в 2D (x, z)
        """
        self.pendulum = pendulum
        self.dt = dt
        self.goal_position_2d = np.array(goal_position_2d, dtype=float)
        self.position_2d = np.array(initial_position_2d, dtype=float)
        
        # Проверяем размерности
        if len(self.goal_position_2d) != 2:
            raise ValueError(f"goal_position_2d должна быть 2D, получена {len(self.goal_position_2d)}D")
        if len(self.position_2d) != 2:
            raise ValueError(f"initial_position_2d должна быть 2D, получена {len(self.position_2d)}D")
        
        # Функция стоимости работает в 2D
        self.cost_function = lambda x=self.position_2d: np.sum((x - self.goal_position_2d) ** 2)
        self.cost = self.cost_function()
    
    def get_position_2d(self) -> np.array:
        """Возвращает текущую 2D позицию (x, z)"""
        return self.position_2d.copy()
    
    def get_cost(self) -> float:
        """Возвращает текущую стоимость"""
        return self.cost
    
    def set_position_2d(self, position_2d: np.array):
        """Устанавливает новую 2D позицию"""
        position_2d = np.array(position_2d, dtype=float)
        if len(position_2d) != 2:
            raise ValueError(f"position_2d должна быть 2D, получена {len(position_2d)}D")
        
        self.position_2d = position_2d
        self.cost = self.cost_function()
    
    def evolve(self, state: np.array = None, control: float = 0) -> np.array:
        """
        Эволюция в 2D пространстве состояний.
        
        Args:
            state: 2D состояние (x, z). Если None, используется текущая позиция
            control: Управляющее воздействие
            
        Returns:
            Новое 2D состояние после эволюции
        """
        if state is None:
            state = self.position_2d
        else:
            # Конвертируем в numpy array
            state = np.array(state, dtype=float)
            
            # Проверяем что это не скаляр
            if np.isscalar(state) or state.ndim == 0:
                raise ValueError(f"state не может быть скаляром: {state}")
            
            # Проверяем размерность
            if len(state) != 2:
                raise ValueError(f"state должно быть 2D, получено {len(state)}D: {state}")
        
        next_state = self.pendulum.discrete_step(state, control)
        return next_state
    
    def step(self, state: np.array = None, control: float = 0) -> 'SporeLogic':
        """
        Создает новую SporeLogic после эволюции.
        
        Args:
            state: 2D состояние для эволюции
            control: Управляющее воздействие
            
        Returns:
            Новый экземпляр SporeLogic с обновленной позицией
        """
        if state is None:
            state = self.position_2d
        
        new_position_2d = self.evolve(state=state, control=control)
        
        new_logic = SporeLogic(
            pendulum=self.pendulum,
            dt=self.dt,
            goal_position_2d=self.goal_position_2d,
            initial_position_2d=new_position_2d
        )
        return new_logic
    
    def clone(self) -> 'SporeLogic':
        """Клонирование логического состояния"""
        return SporeLogic(
            pendulum=self.pendulum,
            dt=self.dt,
            goal_position_2d=self.goal_position_2d,
            initial_position_2d=self.position_2d
        )
    
    def sample_random_controls(self, N: int) -> np.array:
        """Генерирует N случайных управляющих воздействий"""
        a, b = self.pendulum.get_control_bounds()
        return np.random.uniform(a, b, N)
    
    def sample_mesh_controls(self, N: int) -> np.array:
        """Генерирует N равномерно распределенных управляющих воздействий"""
        a, b = self.pendulum.get_control_bounds()
        return np.linspace(a, b, N)
    
    def sample_controls(self, N: int, method: str = 'random') -> np.array:
        """
        Универсальный метод генерации управлений.
        
        Args:
            N: Количество управлений
            method: 'random' или 'mesh'
            
        Returns:
            Массив управляющих воздействий
        """
        if method == 'random':
            return self.sample_random_controls(N)
        elif method == 'mesh':
            return self.sample_mesh_controls(N)
        else:
            raise ValueError(f"Неизвестный метод: {method}. Используйте 'random' или 'mesh'")
    
    def simulate_controls(self, controls: np.array) -> np.array:
        """
        Симулирует множественные управления от текущего состояния.
        
        Args:
            controls: Массив управляющих воздействий
            
        Returns:
            Массив 2D состояний после применения каждого управления
        """
        controls = np.array(controls)
        states = np.zeros((len(controls), 2))
        for i, control in enumerate(controls):
            states[i] = self.evolve(state=self.position_2d, control=control)
        return states
    
    def __str__(self):
        return f"SporeLogic(pos={self.position_2d}, cost={self.cost:.3f})"
    
    def __repr__(self):
        return self.__str__() 