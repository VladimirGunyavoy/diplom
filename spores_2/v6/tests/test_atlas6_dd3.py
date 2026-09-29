"""Клетка дифдрайва: поток, левая инвариантность (клетка в q = L_q шаблона из нуля). Запуск: python3 tests/test_atlas6_dd3.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.dd3 import Cell3, flow, compose

def test_flow_arc():
    p = flow(np.array([0.0, 0.0, 0.0]), (1.0, 1.0), np.pi / 2)      # четверть окружности радиуса 1
    assert np.allclose(p, [1.0, 1.0, np.pi / 2]); assert np.allclose(flow(np.array([0.0, 0.0, 0.3]), (1.0, 0.0), 2.0), [2 * np.cos(.3), 2 * np.sin(.3), .3])

def test_left_invariance():
    rng = np.random.default_rng(1); worst = 0.0
    for vw in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 0.5), (1.0, -1.0)):
        tmpl = Cell3((0, 0, 0), vw, 0.1, 0.8)
        for _ in range(5):
            q = np.array([*rng.uniform(-3, 3, 2), rng.uniform(-np.pi, np.pi)]); cq = Cell3(q, vw, 0.1, 0.8)
            for a, b in ((cq.corners(), compose(q, tmpl.corners())), (cq.exit(), compose(q, tmpl.exit())), (cq.entry(), compose(q, tmpl.entry())), (cq.wall(2, [0.4]), compose(q, tmpl.wall(2, [0.4])))):
                d = np.abs(a - b); d[..., 2] = np.abs((a[..., 2] - b[..., 2] + np.pi) % (2 * np.pi) - np.pi); worst = max(worst, d.max())
            assert abs(cq.stretch() - tmpl.stretch()) < 1e-10
    print(f"максимальное расхождение клетка(q) − L_q(шаблон): {worst:.1e}"); assert worst < 1e-9

def test_stretch_linear_not_exp():
    rho = [Cell3((0, 0, 0), (1.0, 0.0), 0.1, t).stretch() for t in (0.5, 1, 2, 4)]
    rot = Cell3((0, 0, 0), (0.0, 1.0), 0.1, 3.0).stretch()
    print("ρ прямой при τ=.5/1/2/4:", [round(r, 2) for r in rho], "; поворот на месте:", round(rot, 3))
    assert np.all(np.diff(rho) > 0) and abs(rot - 1) < 1e-9

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
