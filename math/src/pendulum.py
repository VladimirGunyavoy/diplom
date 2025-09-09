import numpy as np
from .pendulum_jit import phase_acceleration_vector


class Pendulum:
    def __init__(self, 
                 mass=1, 
                 length=1, 
                 damping=0.1, 
                 gravity=9.81,
                 init_state=np.array([0, 0]),
                 dt=0.01):
        self.mass = mass
        self.length = length
        self.damping = damping
        self.gravity = gravity
        self._state = init_state
        self.dt = dt
        self.g = gravity
        self.l = length
        self.b = damping

    @property
    def state(self):
        return self._state
        
    @state.setter
    def state(self, state):
        self._state = state

    def get_state(self):
        return self.state
    
    def get_phase_acceleration_vector(self):
        """Возвращает вектор фазового ускорения маятника"""
        return phase_acceleration_vector(self.state, self.length, self.damping, self.gravity)
    