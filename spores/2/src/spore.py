from .scalable import Scalable
import numpy as np

class Spore(Scalable):
    def __init__(self, pendulum, dt, model='sphere', *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.pendulum = pendulum
        self.A_2 = pendulum.A_2
        self.B_2 = pendulum.B_2
        self.A_3 = self.calc_A_3()
        self.B_3 = self.calc_B_3()
        self.dt = dt

        # Сохраняем параметры инициализации для создания новых спор
        self._initial_pendulum = pendulum
        self._initial_dt = dt
        self._initial_model = model
        self._initial_args = args
        self._initial_kwargs = kwargs

    def __str__(self):
        return f'{self.position}'

    def __repr__(self):
        return f'{self.position}'

    def calc_2d_pos(self):
        return np.array([self.position[0], self.position[2]])
    
    def calc_3d_pos(self, pos_2d):
        return np.array([pos_2d[0], self.position[1], pos_2d[1]])

    def A_evolve(self):
        pos_2d = self.calc_2d_pos()
        next_pos_2d = self.A_2 @ pos_2d
        next_pos_3d = self.calc_3d_pos(next_pos_2d)

        # Создаем НОВЫЙ экземпляр Spore, используя сохраненные параметры инициализации,
        # а затем обновляем его позицию.
        init_kwargs = self._initial_kwargs.copy()

        new_spore = Spore(
            pendulum=self._initial_pendulum,
            dt=self._initial_dt,
            model=self._initial_model,
            *self._initial_args,
            **init_kwargs
        )
        new_spore.position = next_pos_3d
        return new_spore
    
    def calc_A_3(self):
        return np.array([[self.A_2[0, 0], 0, self.A_2[0, 1]],
                         [             0, 1,              0],
                         [self.A_2[1, 0], 0, self.A_2[1, 1]]])

    def calc_B_3(self):
        return np.array([self.B_2[0, 0], 
                                      0, 
                         self.B_2[1, 0]])
