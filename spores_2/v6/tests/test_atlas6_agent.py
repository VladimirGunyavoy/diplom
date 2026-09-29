"""Агент v6 DI: T*, доходимость и время/T* с клеткой цели. Запуск: python3 tests/test_atlas6_agent.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.agent import T_star, make_cells, run_agent

def test_T_star():
    assert abs(T_star(-2, 0) - 2 * 2 ** 0.5) < 1e-12 and abs(T_star(1, 0) - 2) < 1e-12 and abs(T_star(0, 0)) < 1e-12
    assert abs(T_star(-0.5, 1) - 1) < 1e-12                 # на кривой: x = −v|v|/2

def test_agent_goal_cell():
    rng = np.random.default_rng(0); st = rng.uniform(-2, 2, (20, 2))
    for h in (0.4, 0.2):
        cells = make_cells(h)
        res = [run_agent(x, v, cells, R_goal=2 * h) for x, v in st]
        ok = [r for r in res if r[1]]
        print(f"h={h}: дошли {len(ok)}/20, время/T* mean {np.mean([r[0] for r in ok]):.3f} max {np.max([r[0] for r in ok]):.3f}, переключений mean {np.mean([r[2] for r in res]):.1f}")
        assert len(ok) == 20 and max(r[0] for r in ok) < 1.15

def test_agent_no_goal_cell_stalls():
    rng = np.random.default_rng(0); st = rng.uniform(-2, 2, (20, 2)); cells = make_cells(0.4)
    n = sum(run_agent(x, v, cells, R_goal=0.0)[1] for x, v in st)
    print(f"без клетки цели, h=0.4: дошли {n}/20"); assert n < 20

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
