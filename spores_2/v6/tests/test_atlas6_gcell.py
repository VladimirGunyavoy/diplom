"""Общая клетка v6: RK4-поток против аналитики DI и физика маятника. Запуск: python3 tests/test_atlas6_gcell.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.cell import Cell
from src.atlas6.gcell import GCell, double_integrator, pendulum, rk4

def test_matches_analytic_DI():
    a = Cell((0, 2), +1, 0.2, 0.7, Lx=4, Lv=2); g = GCell(double_integrator(), (0, 2), +1, 0.2, 0.7, Lx=4, Lv=2)
    assert np.allclose(a.d, g.d) and np.allclose(a.exit(), g.exit(), atol=1e-9) and np.allclose(a.entry(), g.entry(), atol=1e-9)
    t = np.linspace(-0.7, 0.7, 7)
    assert np.allclose(a.wall(1, t), g.wall(1, t), atol=1e-9) and abs(a.stretch() - g.stretch()) < 1e-9

def test_pendulum_energy_and_hyperbolic_stretch():
    f = pendulum(1.0); x0 = np.array([1.0, 0.0])
    E = lambda x: x[1] ** 2 / 2 - np.cos(x[0])
    assert abs(E(rk4(f, x0, 0.0, 20.0)) - E(x0)) < 1e-6                       # u=0: энергия сохраняется
    up = [GCell(f, (np.pi - 0.05, 0.0), 0.2, 0.01, tau).stretch() for tau in (0.5, 1.0, 2.0)]   # у верхнего положения — экспонента e^{λt}, λ=1
    print("ρ у верхнего положения при τ=0.5/1/2:", [round(r, 2) for r in up])
    assert up[0] < up[1] < up[2] and up[2] > 3
    down = GCell(f, (0.0, 0.5), 0.2, 0.01, 2.0).stretch()
    assert down < up[2]

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
