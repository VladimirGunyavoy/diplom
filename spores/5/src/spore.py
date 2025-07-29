from .scalable import Scalable
from .color_manager import ColorManager
import numpy as np
from ursina import *

class Spore(Scalable):
    def __init__(self, pendulum, dt, 
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

        # Сохраняем параметры инициализации для создания новых спор
        self._initial_pendulum = pendulum
        self._initial_dt = dt
        self._initial_model = model
        self._initial_color_manager = color_manager
        self._initial_is_goal = is_goal
        self._initial_args = args
        self._initial_kwargs = kwargs

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

    def predict_next_position(self, control=0):
        state = self.calc_2d_pos()
        next_state = self.pendulum.discrete_step(state, control)
        return self.calc_3d_pos(next_state)

    def clone(self):
        init_kwargs = self._initial_kwargs.copy()
        init_kwargs['position'] = self.real_position
        return Spore(
            pendulum=self._initial_pendulum,
            dt=self._initial_dt,
            model=self._initial_model,
            color_manager=self._initial_color_manager,
            is_goal=self._initial_is_goal,
            *self._initial_args,
            **init_kwargs
        )
    
    def step(self, control=0):
        new_spore = self.clone()
        new_spore.real_position = self.predict_next_position(control)
        return new_spore
