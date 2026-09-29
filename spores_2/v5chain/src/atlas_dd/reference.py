"""A3: эталон «поворот–прямая–поворот» (RSR) с непрерывным курсом, цель (0,0,0): время при v, ω по очереди с ОДНОЙ прямой.
НЕ граница оптимума: граф с двумя прямыми (R-S-R-S, напр. узел (−1,1,90°): назад 1, поворот, вперёд 1) бывает быстрее; дуги тоже.
Точного эталона для дифдрайва нет — проверка сходимости: сравнивать с более плотной решёткой (A3b)."""
import numpy as np
from math import atan2, hypot, pi


def _wrap(a):
    return abs((a + pi) % (2 * pi) - pi)


def T_rsr(x, y, th, vmax=1.0, wmax=1.0):
    d = hypot(x, y)
    if d < 1e-12:
        return _wrap(th) / wmax
    bearing = atan2(-y, -x)                                  # направление на цель
    best = np.inf
    for hd, sg in ((bearing, 1), (bearing + pi, -1)):        # вперёд / задним ходом (курс при хвосте на цель)
        best = min(best, (_wrap(hd - th) + _wrap(0.0 - hd)) / wmax + d / vmax)
    return best
