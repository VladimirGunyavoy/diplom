"""Динамика 3 звеньев: при τ=0 энергия сохраняется (дрейф < 1e-9), баланс мощности dE = ∫τ·w (< 1e-3). Запуск из v6: python3 tests/check_manip3dyn.py"""
import sys; sys.path.insert(0, '.')
import numpy as np
from src.atlas6 import manip3dyn as M
x = np.array([.3, -.5, .8, .4, -.6, .2]); h = .01
def rk(z, tau): k1 = M.f(z, tau); k2 = M.f(z + h / 2 * k1, tau); k3 = M.f(z + h / 2 * k2, tau); k4 = M.f(z + h * k3, tau); return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
z = x.copy()
for _ in range(300): z = rk(z, (0., 0., 0.))
d = abs(M.energy(z) - M.energy(x)); print('дрейф E', d); assert d < 1e-9
z = x.copy(); W = 0; tau = (1., -1., 1.)
for _ in range(300): zn = rk(z, tau); W += h * np.dot(tau, (z[3:] + zn[3:]) / 2); z = zn
r = abs(M.energy(z) - M.energy(x) - W) / abs(W); print('баланс мощности', r); assert r < 1e-3
z = x.copy(); fg = lambda z: M.f(z, (0., 0., 0.), g=9.8); h = .0025
for _ in range(1200): k1 = fg(z); k2 = fg(z + h / 2 * k1); k3 = fg(z + h / 2 * k2); k4 = fg(z + h * k3); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
d = abs(M.energy(z, g=9.8) - M.energy(x, g=9.8)); print('дрейф E с гравитацией', d); assert d < 1e-6
for n, g in ((3, 0.0), (3, 0.3)):
    a = M.flow(x, 5, 1.0, n=n, g=g); b = M.flow(x[None, :], 5, 1.0, n=n, g=g)[0]; e = np.abs(a - b).max(); print('скалярный flow vs векторный', e); assert e < 1e-12
print('OK')
