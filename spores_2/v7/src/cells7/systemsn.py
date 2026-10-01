"""Системы 0з-7 для CellN: нормированные координаты, flow(y,k,t) батч. Манипуляторы/дифдрайв берутся из v6 (src/atlas6), нормировка: углы/угл. скорости в рад и рад/с-собственных (масштаб 1), |ω|≤WM."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'v6', 'src', 'atlas6'))   # модули v6 по именам файлов (пакет src у v7 свой)
import numpy as np
from .celln import SysN
from .systems import di, pend


def from2(S2):
    return SysN(S2.name, 2, 2, lambda y, k, t: S2.rk4(y, k, t), ((-1, -1), (1, 1)), per=S2.per)


def dd(layers='rhomb'):
    from dd3 import flow
    LAYERS = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]; RECT = LAYERS + [(sv, sw) for sv in (1.0, -1.0) for sw in (1.0, -1.0)]
    L = LAYERS if layers == 'rhomb' else RECT; sc = np.array([2.0, 2.0, np.pi])
    return SysN('dd_' + layers, 3, len(L), lambda y, k, t: flow(y * sc, L[k], t) / sc, ((-1, -1, -1), (1, 1, 1)), per=(0, 0, 2.0))


def manip(n, g=0.0, WM=3.0, dt=0.05):
    from manip3dyn import flow
    lo = [-np.pi] * n + [-WM] * n; hi = [np.pi] * n + [WM] * n
    return SysN('manip%d_g%.1f' % (n, g), 2 * n, 2 ** n, lambda y, k, t: flow(y, k, t, dt_max=dt, n=n, g=g), (lo, hi), per=[2 * np.pi] * n + [0] * n)


SYSTEMS = {'di': lambda: from2(di()), 'pend': lambda: from2(pend(0.3)), 'dd': lambda: dd('rhomb'), 'dd_rect': lambda: dd('rect'),
           'm2': lambda: manip(2), 'm2g': lambda: manip(2, 0.3), 'm3': lambda: manip(3), 'm3g': lambda: manip(3, 0.3), 'm4g': lambda: manip(4, 0.3)}
