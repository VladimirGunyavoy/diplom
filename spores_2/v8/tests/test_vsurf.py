"""Поверхность V (src/stepper/vsurf.py) без Ursina: живой атлас ДИ (LiveStepper, слои 1/−1) → solve8 → меш. v8, слово пользователя 2026-10-12."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.stepper.live import LiveStepper
from src.stepper import vsurf as VS


def _atlas():
    L = LiveStepper(seed=None, layers=(1., -1.)); L.key('C'); assert L.snapshot['phase'] == 'done'
    return {u: c for u, c in L.cells_by_layer.items()}


def test_gradient():
    c = VS.gradient(np.array([0., .5, 1., 7., -3.])); assert c.shape == (5, 4) and np.allclose(c[:, 3], VS.ALPHA) and np.allclose(c[0], c[4]) and np.allclose(c[2], c[3]) and not np.allclose(c[0], c[2])


def test_solve_and_mesh():
    A = _atlas(); Vs, info = VS.solve_layers(A)
    assert info['n'] > 100 and info['finite'] > .3 and info['t'] < 30
    for u, cl in A.items():
        assert len(Vs[u]) == len(cl)
        for c, V in zip(cl, Vs[u]): assert V.shape == c['G'].shape[:2]
    vmax = VS.vscale(Vs); assert vmax > 1
    for u, cl in A.items():
        m = VS.build_mesh(cl, Vs[u], vmax=vmax)
        assert m['verts'].shape[1] == 3 and m['colors'].shape == (len(m['verts']), 4) and len(m['tris']) % 3 == 0 and m['tris'].max() < len(m['verts'])
        assert m['pts'].shape == m['pcol'].shape[:1] + (3,) and np.all(m['pts'][:, 1] >= 0)
        fin = np.isfinite(m['pts']).all(); assert fin
        # высота = V·VH на конечных узлах; ∞ — серые на y0, вне поверхности (треугольники с такой вершиной не построены)
        used = np.unique(m['tris']); assert (m['verts'][used, 1] <= vmax * 100 * VS.VH).all()
        assert np.allclose(m['inf'][:, 1], .01) if len(m['inf']) else True


def test_goal_zero_and_edge_e1():
    """узлы внутри цели (|x| ≤ RHO) имеют V=0; выполнено ребро E1 вдоль потока: V[k] ≤ (tau[k+1] − tau[k]) + V[k+1] там, где V[k+1] конечно"""
    A = _atlas(); Vs, _ = VS.solve_layers(A); from src.algo import growN as g; bad = 0; tot = 0
    for u, cl in A.items():
        for c, V in zip(cl, Vs[u]):
            P = np.asarray(c['G']); fin = V < VS.BIG / 2; inn = np.linalg.norm(P, axis=-1) <= g.RHOV[0] + 1e-9
            assert (V[inn] == 0).all()
            tau = np.asarray(c['tau']); cost = tau[1:] - tau[:-1]; ok = fin[1:] & (cost > 1e-12); tot += int(ok.sum()); bad += int((V[:-1][ok] > (cost + V[1:])[ok] + 1e-6).sum())
    assert tot > 0 and bad == 0, (bad, tot)


def test_inf_nodes_are_holes():
    A = _atlas(); Vs, _ = VS.solve_layers({1.: A[1.][:1]})    # один слой, одна клетка: часть узлов без пути
    cl = A[1.][:1]; m = VS.build_mesh(cl, Vs[1.], vmax=VS.vscale(Vs)); nt, M = Vs[1.][0].shape; fin = Vs[1.][0] < VS.BIG / 2
    assert len(m['inf']) == int((~fin).sum()) and len(m['pts']) == int(fin.sum())


def test_skeleton_edges():
    """рёбра скелета: вдоль (k,j)–(k+1,j) и поперёк (k,j)–(k,j+1) только между конечными узлами; число = числу конечных пар; концы лежат на узлах поверхности"""
    A = _atlas(); Vs, _ = VS.solve_layers(A); vmax = VS.vscale(Vs)
    for u, cl in A.items():
        m = VS.build_mesh(cl, Vs[u], vmax=vmax); nl = nc = 0
        for V in Vs[u]: f = V < VS.BIG / 2; nl += int((f[:-1] & f[1:]).sum()); nc += int((f[:, :-1] & f[:, 1:]).sum())
        assert len(m['long']) == 2 * nl and len(m['cross']) == 2 * nc and nl > 0 and nc > 0
        pts = {tuple(np.round(p, 6)) for p in m['pts']}; assert all(tuple(np.round(p, 6)) in pts for p in m['long'][:200]) and all(tuple(np.round(p, 6)) in pts for p in m['cross'][:200])
