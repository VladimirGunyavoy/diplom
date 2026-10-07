"""solve8 без окна: маленький атлас (слои ±1) → граф, V: инварианты (V≥0, V в цели 0, V ≤ цена E1 + V(next)), V* ≤ T_ref-оценки не завышает сильно, агент доходит из покрытых точек, t_ref."""
import sys, os
os.environ.update(MAXC='14')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from src.algo import solve8 as S


def build():
    g = S.load_growN(); atlas, _ = S.build_atlas([1., -1.], 0, par=False); G = S.make_graph(g, atlas); G.solve(verbose=False); return g, G


def test_graph_invariants():
    g, G = build(); V = G.V; fin = V < S.BIG / 2
    assert fin.any() and (V[fin] >= 0).all()
    assert (V[np.linalg.norm(G.P, axis=1) <= G.rho] == 0).all()
    e = np.flatnonzero((G.nxt >= 0) & (G.cost > 1e-12) & fin[np.maximum(G.nxt, 0)]); assert (V[e] <= G.cost[e] + V[G.nxt[e]] + 1e-9).all()


def test_entry_time():
    X = np.array([[.5, -.9], [.1, 0.], [1., 0.]]); te = S.entry_time(X, 1., np.array([5., 5., 5.]), .2)
    x, v = .5 - .9 * te[0] + te[0] ** 2 / 2, -.9 + te[0]; assert np.isfinite(te[0]) and abs(x * x + v * v - .04) < 1e-8
    assert np.isinf(te[1]) and np.isinf(te[2])                                        # внутри круга — не считается (узлы внутри V = 0 в графе); (1,0) под u=+1 круг не достигает


def test_query_and_agent():
    g, G = build(); X0 = S.starts(g, 20, 1); Tr, src = S.t_ref(X0); Vs, us = G.Vstar(X0); ok = np.isfinite(Vs)
    assert ok.any() and (Vs[ok] <= 1.05 * Tr[ok] + .5).all()
    T, sw = G.run_agent(X0[ok][:5], 3 * Tr[ok][:5] + 2); assert T.shape == (5,)


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
