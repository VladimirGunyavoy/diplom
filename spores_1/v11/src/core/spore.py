from __future__ import annotations
from ..visual.spore_visual import SporeVisual
from ..logic.spore_logic import SporeLogic
from ..managers.color_manager import ColorManager
import numpy as np
from ..logic.pendulum import PendulumSystem
from typing import Optional, Any, Tuple, List, Dict

class Spore(SporeVisual):
    """
    Объединенный класс Spore, который:
    - Наследуется от SporeVisual (визуализация)
    - Агрегирует SporeLogic (математическая логика)
    - Обеспечивает обратную совместимость с существующим API
    """
    
    def __init__(self, 
                 dt: float, 
                 pendulum: PendulumSystem, 
                 goal_position: np.ndarray | Tuple[float, float] | Tuple[float, float, float],
                 model: str = 'sphere', 
                 color_manager: Optional[ColorManager] = None,
                 is_goal: bool = False,
                 is_ghost: bool = False,
                 *args: Any, **kwargs: Any):
        # Инициализируем визуальную часть
        super().__init__(model=model, color_manager=color_manager, 
                        is_goal=is_goal, *args, **kwargs)
        
        self.is_ghost: bool = is_ghost # Сохраняем флаг

        # Создаем логическую часть
        # Конвертируем goal_position в 2D если нужно
        goal_position_2d = np.array(goal_position)[:2]
        
        # Определяем начальную позицию для логики
        if 'position' in kwargs:
            initial_pos = kwargs['position']
            if hasattr(initial_pos, '__len__') and len(initial_pos) >= 2:
                # Конвертируем 3D позицию в 2D для логики
                initial_pos_2d = np.array(initial_pos)[[0, 2]] if len(initial_pos) == 3 else np.array(initial_pos)
            else:
                initial_pos_2d = np.array([0.0, 0.0])  # По умолчанию
        else:
            initial_pos_2d = np.array([0.0, 0.0])  # По умолчанию
            
        self.logic: SporeLogic = SporeLogic(
            pendulum=pendulum, 
            dt=dt, 
            goal_position_2d=goal_position_2d,
            initial_position_2d=initial_pos_2d
        )
        
        # Сохраняем параметры для обратной совместимости
        self.dt: float = dt
        self.pendulum: PendulumSystem = pendulum
        self.goal_position: np.ndarray | Tuple[float, float] | Tuple[float, float, float] = goal_position
        
        # Сохраняем параметры инициализации для создания новых спор
        self._initial_pendulum: PendulumSystem = pendulum
        self._initial_dt: float = dt
        self._initial_model: str = model
        self._initial_color_manager: Optional[ColorManager] = color_manager
        self._initial_is_goal: bool = is_goal
        self._initial_is_ghost: bool = is_ghost
        
        init_kwargs = kwargs.copy()
        init_kwargs['model'] = model
        init_kwargs['color_manager'] = color_manager
        init_kwargs['is_goal'] = is_goal
        
        self._initial_kwargs: Dict[str, Any] = init_kwargs
        
        # Флаг завершения эволюции (для объединенных спор)
        self.evolution_completed = False
        
        # Обратная совместимость: cost_function и cost
        self.cost_function = self.logic.calculate_cost
        
        # Устанавливаем правильный цвет при создании
        self.update_visual_state()
        
    
    def apply_transform(self, a: float, b: np.ndarray, **kwargs: Any) -> None:
        """Применяет трансформацию к визуальной части"""
        super().apply_transform(a, b, **kwargs)
    
    def calc_2d_pos(self) -> np.ndarray:
        """Обратная совместимость: возвращает 2D позицию из логики"""
        return self.logic.position_2d.copy()
    
    def calc_3d_pos(self, pos_2d: np.ndarray) -> np.ndarray:
        """Обратная совместимость: конвертирует 2D позицию в 3D"""
        pos_2d = np.array(pos_2d, dtype=float)
        return np.array([pos_2d[0], self.y_coordinate, pos_2d[1]])
    
    def evolve_3d(self, state: Optional[np.ndarray] = None, control: float = 0, dt: Optional[float] = None) -> np.ndarray:
        """
        Обратная совместимость: эволюция с возвратом 3D позиции
        """
        if state is None:
            # Используем текущую позицию логики
            new_pos_2d = self.logic.evolve(control=control, dt=dt)
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
            new_pos_2d = temp_logic.evolve(control=control, dt=dt)
        
        return np.array([new_pos_2d[0], self.y_coordinate, new_pos_2d[1]])
    
    def evolve_2d(self, state: Optional[np.ndarray] = None, control: float = 0, dt: Optional[float] = None) -> np.ndarray:
        """
        Обратная совместимость: эволюция с возвратом 2D позиции
        """
        if state is None or len(state) != 2:
            # Используем текущую позицию логики
            return self.logic.evolve(control=control, dt=dt)
        else:
            # Создаем временную логику для эволюции
            temp_logic = SporeLogic(dt=self.dt, pendulum=self.pendulum, 
                                   goal_position_2d=self.logic.goal_position_2d,
                                   initial_position_2d=np.array(state))
            return temp_logic.evolve(control=control, dt=dt)
    
    def step(self, state: Optional[np.ndarray] = None, control: float = 0, dt: Optional[float] = None) -> Spore:
        """
        Обратная совместимость: создает новую спору с эволюционированной позицией
        """
        if state is None:
            state = self.real_position
        
        new_spore = self.clone()
        new_3d_position = self.evolve_3d(state=state, control=control, dt=dt)
        new_spore.real_position = new_3d_position
        
        # Синхронизируем логику с новой позицией
        new_spore.set_logic_position_2d(
            np.array([new_3d_position[0], new_3d_position[2]])
        )
        
        return new_spore
    
    def clone(self, new_position: Optional[np.ndarray | Tuple[float, float, float]] = None) -> Spore:
        """
        Обратная совместимость: создает копию споры.
        Можно передать новую позицию для клона.
        """
        init_kwargs = self._initial_kwargs.copy()
        init_kwargs['position'] = new_position if new_position is not None else self.real_position
        
        # Убираем дублирующиеся ключи, если они пришли через _initial_args
        for key in ['pendulum', 'dt', 'model', 'color_manager', 'goal_position', 'is_goal', 'is_ghost']:
            if key in init_kwargs:
                del init_kwargs[key]
        
        new_spore = Spore(
            pendulum=self._initial_pendulum,
            dt=self._initial_dt,
            model=self._initial_model,
            color_manager=self._initial_color_manager,
            goal_position=self.goal_position,
            is_goal=self._initial_is_goal,
            is_ghost=self._initial_is_ghost,
            **init_kwargs
        )
        
        # Копируем состояние логики
        new_spore.logic.position_2d = self.logic.position_2d.copy()
        
        return new_spore
    
    def sample_random_controls(self, N: int) -> np.ndarray:
        """Обратная совместимость: генерация случайных управлений"""
        return self.logic.sample_controls(N, method='random')
    
    def sample_mesh_controls(self, N: int) -> np.ndarray:
        """Обратная совместимость: генерация сетки управлений"""
        return self.logic.sample_controls(N, method='mesh')
    
    def simulate_controls(self, controls: np.ndarray) -> List[np.ndarray]:
        """Обратная совместимость: симуляция набора управлений"""
        return self.logic.simulate_controls(controls)
    
    def sync_with_logic(self, *args, **kwargs) -> None:
        """Обёртка для синхронизации с внутренней логикой"""
        super().sync_with_logic(self.logic)
    
    def set_logic_position_2d(self, position_2d: np.ndarray) -> None:
        """Устанавливает 2D позицию логики"""
        self.logic.set_position_2d(position_2d)
    
    def is_alive(self) -> bool:
        """Проверяет жива ли спора."""
        return self.logic.is_alive()
    
    def set_alive(self, alive: bool) -> None:
        """Устанавливает состояние жизни споры."""
        self.logic.set_alive(alive)
    
    def check_death(self) -> None:
        """Проверяет условия смерти споры и устанавливает alive=False при необходимости."""
        old_alive = self.logic.is_alive()
        self.logic.check_death()
        
        # Если состояние изменилось, обновляем визуальное представление
        if old_alive != self.logic.is_alive():
            self.update_visual_state()
    
    def mark_evolution_completed(self) -> None:
        """Помечает спору как завершившую эволюцию (например, после объединения)."""
        self.evolution_completed = True
        # Обновляем визуальное состояние для отображения завершения эволюции
        self.update_visual_state()
    
    def can_evolve(self) -> bool:
        """Проверяет может ли спора продолжать эволюцию."""
        return self.is_alive() and not self.evolution_completed
    
    def update_visual_state(self) -> None:
        """Обновляет визуальное состояние споры в зависимости от alive."""
        if hasattr(self, 'is_goal') and self.is_goal:
            # Целевая спора - всегда зеленый цвет (бессмертна)
            self.set_color_type('goal')
        elif hasattr(self, 'is_candidate') and self.is_candidate:
            # Кандидатская спора - белый полупрозрачный
            self.set_color_type('candidate')
        elif hasattr(self, 'is_ghost') and self.is_ghost:
            # Призрачная спора - полупрозрачный цвет
            self.set_color_type('ghost')
        elif self.evolution_completed:
            # Спора завершила эволюцию (объединилась) - синий цвет
            self.set_color_type('completed')
        elif not self.logic.is_alive():
            # Мертвая спора - серый цвет
            self.set_color_type('dead')
        else:
            # Живая обычная спора - стандартный цвет
            self.set_color_type('default')
    
    def remove_self(self) -> None:
        """Обратная совместимость: удаление объекта"""
        self.destroy()
    
    def __str__(self) -> str:
        """Обратная совместимость: строковое представление"""
        return f'{self.position}'
    
    def __repr__(self) -> str:
        """Обратная совместимость: представление для отладки"""
        return f'{self.position}' 