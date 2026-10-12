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
        ncen = 0
        for V in Vs[u]: f = V < VS.BIG / 2; e = f[:-1] & f[1:]; ncen += int(e[:, V.shape[1] // 2].sum()); nl += int(e.sum()) - int(e[:, V.shape[1] // 2].sum()); nc += int((f[:, :-1] & f[:, 1:]).sum())
        assert len(m['long']) == 2 * nl and len(m['cross']) == 2 * nc and len(m['center']) == 2 * ncen and nl > 0 and nc > 0 and ncen > 0
        pts = {tuple(np.round(p, 6)) for p in m['pts']}; assert all(tuple(np.round(p, 6)) in pts for p in m['long'][:200]) and all(tuple(np.round(p, 6)) in pts for p in m['cross'][:200]) and all(tuple(np.round(p, 6)) in pts for p in m['center'][:200])


def test_common_layer_and_goal_columns():
    """общий слой V*: у каждого узла ≤1 победитель по слоям (в разных слоях узлы разные — проверяем долю), узлы с V=∞ не побеждают; столбцы, входящие в цель, есть в каждом слое; рёбра goal лежат на конечных узлах"""
    A = _atlas(); Vs, info = VS.solve_layers(A); vmax = VS.vscale(Vs); nw = nn = 0; ng = 0
    for u, cl in A.items():
        assert len(info['wins'][u]) == len(cl) and len(info['goalcols'][u]) == len(cl)
        for V, w in zip(Vs[u], info['wins'][u]): assert not (w & (V >= VS.BIG / 2)).any(); nw += int(w.sum()); nn += w.size
        m = VS.build_mesh(cl, Vs[u], vmax=vmax, win=info['wins'][u], goalcols=info['goalcols'][u]); mf = VS.build_mesh(cl, Vs[u], vmax=vmax, goalcols=info['goalcols'][u])
        assert len(m['pts']) == sum(int(w.sum()) for w in info['wins'][u]) and len(m['pts']) <= len(mf['pts']) and len(m['inf']) == 0 and len(m['tris']) <= len(mf['tris'])
        ng += len(mf['goal']); assert len(mf['goal']) + len(mf['goalc']) > 0 and len(mf['goalc']) > 0 and len(m['goal']) <= len(mf['goal']) and len(m['goalc']) <= len(mf['goalc'])
    assert 0 < nw < nn          # общий слой — часть узлов (оба слоя что-то выигрывают, но не всё)


def test_lift():
    A = _atlas(); Vs, _ = VS.solve_layers(A); u = next(iter(A)); m = VS.build_mesh(A[u], Vs[u], vmax=1., lift=.05); m0 = VS.build_mesh(A[u], Vs[u], vmax=1., lift=0.)
    assert np.allclose(m['pts'][:, 1] - m0['pts'][:, 1], .05) and np.allclose(m['long'][:, 1] - m0['long'][:, 1], .05)


def test_time_skeleton():
    """временной скелет на поверхности: V(t) по Gt — интерполяция вдоль столбца по tau; в узлах, где tt совпал с tau, V та же; рёбра есть, поверхность и узлы не меняются, высота в пределах [V]"""
    A = _atlas(); Vs, _ = VS.solve_layers(A); vmax = VS.vscale(Vs); nhit = 0; nl = 0
    for u, cl in A.items():
        for c, V in zip(cl, Vs[u]):
            fin = V < VS.BIG / 2; Gt, Vt, ok = VS.time_nodes(c, V, fin); tau = np.asarray(c['tau']); tt = np.asarray(c['tt']); assert Vt.shape == (len(tt), V.shape[1]) and Gt.shape[:2] == Vt.shape
            for j in range(V.shape[1]):
                for k in range(len(tau)):
                    m = np.abs(tt - tau[k, j]) < 1e-9
                    if m.any() and fin[k, j] and ok[m, j].all(): assert np.allclose(Vt[m, j], V[k, j], atol=1e-6); nhit += 1
            if not fin.any(): assert not ok.any(); continue
            assert (Vt[ok] >= V[fin].min() - 1e-9).all() and (Vt[ok] <= V[fin].max() + 1e-9).all()
        m0 = VS.build_mesh(cl, Vs[u], vmax=vmax); m1 = VS.build_mesh(cl, Vs[u], vmax=vmax, skel='time')
        assert np.array_equal(m0['verts'], m1['verts']) and np.array_equal(m0['pts'], m1['pts']) and len(m1['long']) > 0 and len(m1['cross']) > 0 and len(m1['center']) > 0
        assert (m1['long'][:, 1] <= vmax * 100 * VS.VH + 1).all() and not np.array_equal(m0['long'], m1['long']); nl += len(m1['long'])
    assert nhit > 0 and nl > 0
