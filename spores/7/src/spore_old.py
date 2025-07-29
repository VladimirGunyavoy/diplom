from .scalable import Scalable
from .color_manager import ColorManager
import numpy as np
from ursina import *
from .pendulum import PendulumSystem

class Spore(Scalable):
    def __init__(self, dt, 
                 pendulum: PendulumSystem, 
                 goal_position,
                 model='sphere', 
                 color_manager=None,
                 is_goal=False,
                 *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        
        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        # Выбираем цвет в зависимости от того, является ли спора целевой
        color_type = 'goal' if is_goal else 'default'
        self.color = self.color_manager.get_color('spore', color_type)

        self.pendulum = pendulum
        self.dt = dt
        self.goal_position = goal_position

        # Сохраняем параметры инициализации для создания новых спор
        self._initial_pendulum = pendulum
        self._initial_dt = dt
        self._initial_model = model
        self._initial_color_manager = color_manager
        self._initial_is_goal = is_goal
        self._initial_args = args
        self._initial_kwargs = kwargs

        self.cost_function = lambda x=self.position: np.sum((x - self.pendulum.goal_position) ** 2)
        self.cost = self.cost_function()

    def apply_transform(self, a, b, **kwargs):
        spores_scale = kwargs.get('spores_scale', 1.0)
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a * spores_scale

    def __str__(self):
        return f'{self.position}'

    def __repr__(self):
        return f'{self.position}'

    def calc_2d_pos(self):
        return np.array([self.real_position[0], self.real_position[2]])
    
    def calc_3d_pos(self, pos_2d):
        return np.array([pos_2d[0], self.real_position[1], pos_2d[1]])

    def evolve_3d(self, state=None, control=0):
        if state is None:
            state = self.calc_2d_pos()
        next_state = self.pendulum.discrete_step(self.calc_2d_pos(), control)
        return self.calc_3d_pos(next_state)

    def clone(self):
        init_kwargs = self._initial_kwargs.copy()
        init_kwargs['position'] = self.real_position
        return Spore(
            pendulum=self._initial_pendulum,
            dt=self._initial_dt,
            model=self._initial_model,
            color_manager=self._initial_color_manager,
            goal_position=self.goal_position,
            is_goal=self._initial_is_goal,
            *self._initial_args,
            **init_kwargs
        )
    
    def evolve_2d(self, state=None, control=0):
        if state is None or len(state) != 2:
            state = self.calc_2d_pos()
        next_state = self.pendulum.discrete_step(state, control)
        return next_state
    
    def step(self, state=None, control=0):
        if state is None:
            state = self.real_position
        new_spore = self.clone()
        new_spore.real_position = self.evolve_3d(state=state, control=control)
        return new_spore
    
    def sample_random_controls(self, N):
        a, b = self.pendulum.get_control_bounds()
        return np.random.uniform(a, b, N)

    
    def sample_mesh_controls(self, N):
        a, b = self.pendulum.get_control_bounds()
        return np.linspace(a, b, N)
    
    def simulate_controls(self, controls):
        states = np.zeros((len(controls), 2))
        for i, control in enumerate(controls):
            states[i] = self.evolve_2d(state=self.real_position, control=control)
        return states