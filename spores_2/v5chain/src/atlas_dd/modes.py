"""Режимы дифдрайва (АТЛАС §8): to_chart(state) -> (label, s), from_chart(label, s) -> state, rate = ds/dt.
state = (x, y, θ). Все преобразования точные (замкнутая форма)."""
import numpy as np


class Straight:
    """v = sign·vmax, ω = 0. label = (n, θ): боковое смещение и курс; s — путь вдоль курса."""
    def __init__(self, sign, vmax=1.0):
        self.sign, self.vmax = sign, vmax
        self.rate = sign * vmax

    def to_chart(self, st):
        x, y, th = st
        return (-x * np.sin(th) + y * np.cos(th), th), x * np.cos(th) + y * np.sin(th)

    def from_chart(self, label, s):
        n, th = label
        return (s * np.cos(th) - n * np.sin(th), s * np.sin(th) + n * np.cos(th), th)


class Rotate:
    """v = 0, ω = sign·wmax. label = (x, y); s = θ."""
    def __init__(self, sign, wmax=1.0):
        self.sign, self.wmax = sign, wmax
        self.rate = sign * wmax

    def to_chart(self, st):
        x, y, th = st
        return (x, y), th

    def from_chart(self, label, s):
        return (label[0], label[1], s)


class Arc:
    """v = sv·vmax, ω = sw·wmax: окружность радиуса R = v/ω (со знаком). label = центр (xc, yc); s = θ."""
    def __init__(self, sv, sw, vmax=1.0, wmax=1.0):
        self.sv, self.sw, self.vmax, self.wmax = sv, sw, vmax, wmax
        self.R = sv * vmax / (sw * wmax)
        self.rate = sw * wmax

    def to_chart(self, st):
        x, y, th = st
        return (x - self.R * np.sin(th), y + self.R * np.cos(th)), th

    def from_chart(self, label, s):
        return (label[0] + self.R * np.sin(s), label[1] - self.R * np.cos(s), s)
