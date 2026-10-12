"""solve8: расчёт по Q (Graph.set_calc 'q'/'qquad'), слои ±1, M=3, посев 0, ±1 — числа заметки v_kink_q.md (v5chain): узлы <−.05: 1.3%, проходов ~104. По умолчанию ('v') — побитно как раньше."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'algo'))
import numpy as np
import solve8 as S
from src.algo import ref_di_disk as R
_C = {}


def _graph():
    if 'G' not in _C:
        g = S.load_growN(); atlas, _ = S.build_atlas([1., -1.], 0, par=False); _C['g'], _C['atlas'], _C['G'] = g, atlas, S.make_graph(g, atlas, 'N')
    return _C['g'], _C['atlas'], _C['G']


def _err(G):
    T = R.T_ref(G.P, G.rho); out = np.linalg.norm(G.P, axis=1) > G.rho; fin = G.Vn < S.BIG / 2; e = (G.Vn - T)[out & fin]; return e


def test_default_unchanged():
    g, atlas, G = _graph(); G.set_calc('v'); G.solve(verbose=False); V1 = G.V.copy(); p1 = G.passes
    G2 = S.make_graph(g, atlas, 'N'); G2.solve(verbose=False); assert np.array_equal(V1, G2.V) and p1 == G2.passes and np.array_equal(G.Vn, G.V)      # set_calc('v') = как до изменений
    assert (_err(G) < -.05).mean() > .2, 'старый расчёт занижает у кривой переключения'


def test_q_lin():
    g, atlas, G = _graph(); G.set_calc('q'); G.solve(verbose=False); e = _err(G); u = float((e < -.05).mean()); print('q lin: под <−.05 %.4f, проходов %d, |ош| мед %.3f p90 %.3f, solve %.2f с' % (u, G.passes, np.median(np.abs(e)), np.percentile(np.abs(e), 90), G.t_solve))
    assert u < .03 and 80 <= G.passes <= 130 and np.median(np.abs(e)) < .08 and np.percentile(np.abs(e), 90) < .2
    Vs, _ = G.Vstar(S.starts(g, 50, 1)); assert np.isfinite(Vs).mean() > .7      # запросы по Q
    G.set_calc('v'); G.solve(verbose=False)


def test_q_quad():
    g, atlas, G = _graph(); G.set_calc('qquad'); G.solve(verbose=False); e = _err(G); print('q quad: под <−.05 %.4f, проходов %d, |ош| мед %.3f p90 %.3f, solve %.2f с' % (float((e < -.05).mean()), G.passes, np.median(np.abs(e)), np.percentile(np.abs(e), 90), G.t_solve))
    assert np.median(np.abs(e)) < .05 and np.percentile(np.abs(e), 90) < .25 and G.passes < 200 and G.quad
    G.set_calc('v'); assert not G.quad; G.solve(verbose=False); G2 = S.make_graph(g, atlas, 'N'); G2.solve(verbose=False); assert np.array_equal(G.V, G2.V)      # возврат к 'v' — снова побитно
