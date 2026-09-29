"""V по клеткам против T*, затем агент с этим V. Запуск: python3 tests/test_atlas6_value.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.agent import T_star, make_cells, run_agent
from src.atlas6.value import solve_V, V_interp

def test_V_vs_Tstar():
    for h in (0.4, 0.2):
        xs, V, it = solve_V(h)
        rng = np.random.default_rng(0); st = rng.uniform(-2, 2, (200, 2))
        e = np.array([V_interp(xs, V, x, v) - T_star(x, v) for x, v in st])
        print(f"h={h}: {it} итераций, V−T*: mean {e.mean():+.3f} mean|e| {abs(e).mean():.3f} max|e| {abs(e).max():.3f}")
        assert abs(e).mean() < 1.5                            # V занижен интерполяцией вогнутой T* (болезнь v5, source_doc §9.1) — ориентир, не гарантия

def test_agent_with_graph_V():
    rng = np.random.default_rng(0); st = rng.uniform(-2, 2, (20, 2))
    for h in (0.4, 0.2):
        xs, V, _ = solve_V(h)
        Vf = lambda x, v: V_interp(xs, V, x, v)
        from src.atlas6.agent import CellV
        cells = make_cells(h, V=Vf)
        res = [run_agent(x, v, cells, R_goal=2 * h) for x, v in st]
        ok = [r for r in res if r[1]]
        print(f"агент с V из графа, h={h}: дошли {len(ok)}/20, время/T* mean {np.mean([r[0] for r in ok]):.3f} max {np.max([r[0] for r in ok]):.3f}")
        assert len(ok) >= 18

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
