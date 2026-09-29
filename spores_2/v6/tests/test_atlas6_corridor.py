"""Коридор v6 DI: из траектории агента (V из графа) → сжатие → оптимизатор; время против T*. Запуск: python3 tests/test_atlas6_corridor.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.agent import T_star, make_cells
from src.atlas6.value import solve_V, V_interp
from src.atlas6.corridor import agent_controls, compress, endpoint, optimize

def test_corridor():
    rng = np.random.default_rng(0); st = rng.uniform(-2, 2, (20, 2))
    for h, tau in ((0.4, 0.4 ** 0.5), (0.2, 0.4)):
        xs, V, _ = solve_V(h, tau=tau)
        cells = make_cells(h, V=lambda x, v: V_interp(xs, V, x, v))
        r_agent, r_corr, res, nseg = [], [], [], []
        for x, v in st:
            us, ok = agent_controls(x, v, cells, 2 * h)
            if not ok:
                continue
            cor = compress(us)
            e0 = float(np.hypot(*endpoint(x, v, cor)))
            cor2, T, res_ = optimize(x, v, cor)
            r_agent.append(0.01 * len(us) / T_star(x, v)); r_corr.append(T / T_star(x, v)); res.append((e0, res_))
        print(f"h={h}: {len(r_corr)}/20; время/T*: агент {np.mean(r_agent):.3f}, коридор после оптимизации {np.mean(r_corr):.4f} (max {np.max(r_corr):.4f}); невязка конца до/после: {np.mean([a for a, _ in res]):.3f} / {np.max([b for _, b in res]):.1e}")
        assert np.max([b for _, b in res]) < 1e-6 and np.max(r_corr) < 1.01

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
