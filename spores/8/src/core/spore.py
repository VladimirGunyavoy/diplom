from .spore_visual import SporeVisual
from .spore_logic import SporeLogic
from src.managers.color_manager import ColorManager
import numpy as np
from .pendulum import PendulumSystem

class Spore(SporeVisual):
    """
    Объединенный класс Spore, который:
    - Наследуется от SporeVisual (визуализация)
    - Агрегирует SporeLogic (математическая логика)
    - Обеспечивает обратную совместимость с существующим API
    """
    
    def __init__(self, dt, 
                 pendulum: PendulumSystem, 
                 goal_position,
                 model='sphere', 
                 color_manager=None,
                 is_goal=False,
                 *args, **kwargs):
        # Инициализируем визуальную часть
        super().__init__(model=model, color_manager=color_manager, 
                        is_goal=is_goal, *args, **kwargs)
        
        # Создаем логическую часть
        # Конвертируем goal_position в 2D если нужно
        if hasattr(goal_position, '__len__') and len(goal_position) == 3:
            goal_position_2d = np.array([goal_position[0], goal_position[2]])
        else:
            goal_position_2d = np.array(goal_position)
        
        # Определяем начальную позицию для логики
        if 'position' in kwargs:
            initial_pos = kwargs['position']
            if hasattr(initial_pos, '__len__') and len(initial_pos) >= 2:
                # Конвертируем 3D позицию в 2D для логики
                initial_pos_2d = np.array([initial_pos[0], initial_pos[2] if len(initial_pos) > 2 else initial_pos[1]])
            else:
                initial_pos_2d = np.array([0.0, 0.0])  # По умолчанию
        else:
            initial_pos_2d = np.array([0.0, 0.0])  # По умолчанию
            
        self.logic = SporeLogic(
            pendulum=pendulum, 
            dt=dt, 
            goal_position_2d=goal_position_2d,
            initial_position_2d=initial_pos_2d
        )
        
        # Сохраняем параметры для обратной совместимости
        self.dt = dt
        self.pendulum = pendulum
        self.goal_position = goal_position
        
        # Сохраняем параметры инициализации для создания новых спор
        self._initial_pendulum = pendulum
        self._initial_dt = dt
        self._initial_model = model
        self._initial_color_manager = color_manager
        self._initial_is_goal = is_goal
        self._initial_args = args
        self._initial_kwargs = kwargs
        

        
        # Обратная совместимость: cost_function и cost
        self.cost_function = lambda x=None: self.logic.calculate_cost(x) if x is not None else self.logic.get_cost()
        
    @property 
    def cost(self):
        """Обратная совместимость: возвращает текущую стоимость из логики"""
        return self.logic.get_cost()
    
    def apply_transform(self, a, b, **kwargs):
        """Применяет трансформацию к визуальной части"""
        super().apply_transform(a, b, **kwargs)
    
    def calc_2d_pos(self):
        """Обратная совместимость: возвращает 2D позицию из логики"""
        return self.logic.position_2d.copy()
    
    def calc_3d_pos(self, pos_2d):
        """Обратная совместимость: конвертирует 2D позицию в 3D"""
        pos_2d = np.array(pos_2d, dtype=float)
        return np.array([pos_2d[0], self.y_coordinate, pos_2d[1]])
    
    def evolve_3d(self, state=None, control=0):
        """
        Обратная совместимость: эволюция с возвратом 3D позиции
        """
        if state is None:
            # Используем текущую позицию логики
            new_pos_2d = self.logic.evolve(control=control)
        else:
            # Конвертируем state в 2D если нужно
            if hasattr(state, '__len__') and len(state) >= 2:
                if len(state) == 3:
                    state_2d = np.array([state[0], state[2]])
                else:
                    state_2d = np.array(state[:2])
            else:
                state_2d = self.logic.position_2d
            
            # Создаем временную логику для эволюции
            temp_logic = SporeLogic(dt=self.dt, pendulum=self.pendulum, 
                                   goal_position_2d=self.logic.goal_position_2d,
                                   initial_position_2d=state_2d)
            new_pos_2d = temp_logic.evolve(control=control)
        
        return np.array([new_pos_2d[0], self.y_coordinate, new_pos_2d[1]])
    
    def evolve_2d(self, state=None, control=0):
        """
        Обратная совместимость: эволюция с возвратом 2D позиции
        """
        if state is None or len(state) != 2:
            # Используем текущую позицию логики
            return self.logic.evolve(control=control)
        else:
            # Создаем временную логику для эволюции
            temp_logic = SporeLogic(dt=self.dt, pendulum=self.pendulum, 
                                   goal_position_2d=self.logic.goal_position_2d,
                                   initial_position_2d=np.array(state))
            return temp_logic.evolve(control=control)
    
    def step(self, state=None, control=0):
        """
        Обратная совместимость: создает новую спору с эволюционированной позицией
        """
        if state is None:
            state = self.real_position
        
        new_spore = self.clone()
        new_3d_position = self.evolve_3d(state=state, control=control)
        new_spore.real_position = new_3d_position
        
        # Синхронизируем логику с новой позицией
        new_spore.set_logic_position_2d(
            np.array([new_3d_position[0], new_3d_position[2]])
        )
        
        return new_spore
    
    def clone(self):
        """
        Обратная совместимость: создает копию споры
        """
        init_kwargs = self._initial_kwargs.copy()
        init_kwargs['position'] = self.real_position
        
        new_spore = Spore(
            pendulum=self._initial_pendulum,
            dt=self._initial_dt,
            model=self._initial_model,
            color_manager=self._initial_color_manager,
            goal_position=self.goal_position,
            is_goal=self._initial_is_goal,
            *self._initial_args,
            **init_kwargs
        )
        
        # Копируем состояние логики
        new_spore.logic.position_2d = self.logic.position_2d.copy()
        
        return new_spore
    
    def sample_random_controls(self, N):
        """Обратная совместимость: генерация случайных управлений"""
        return self.logic.sample_controls(N, method='random')
    
    def sample_mesh_controls(self, N):
        """Обратная совместимость: генерация сетки управлений"""
        return self.logic.sample_controls(N, method='mesh')
    
    def simulate_controls(self, controls):
        """Обратная совместимость: симуляция набора управлений"""
        return self.logic.simulate_controls(controls)
    
    def sync_position_with_logic(self):
        """Синхронизирует визуальную позицию с логической"""
        pos_2d = self.logic.position_2d
        self.real_position = np.array([pos_2d[0], self.y_coordinate, pos_2d[1]])
    
    def sync_logic_with_position(self):
        """Синхронизирует логическую позицию с визуальной"""
        logic_pos_2d = np.array([self.real_position[0], self.real_position[2]])
        self.logic.position_2d = logic_pos_2d
    
    def sync_with_logic(self):
        """Обёртка для синхронизации с внутренней логикой"""
        super().sync_with_logic(self.logic)
    
    def set_logic_position_2d(self, position_2d):
        """Устанавливает 2D позицию логики"""
        self.logic.set_position_2d(position_2d)
    
    def remove_self(self):
        """Обратная совместимость: удаление объекта"""
        self.destroy()
    
    def __str__(self):
        """Обратная совместимость: строковое представление"""
        return f'{self.position}'
    
    def __repr__(self):
        """Обратная совместимость: представление для отладки"""
        return f'{self.position}' 