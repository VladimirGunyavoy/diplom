"""research hub-v5chain-research-9: характеристики ПМП маятника ФРОНТОМ (выход из тупика pend_char_adapt.py: там вставка шла по параметру θ на
границе цели, а семейство расходится у седла как e^t ⇒ нужна точность θ ~ e^−T, «вечные» дыры). Здесь параметр — сам фронт: все характеристики
(x, p) идут назад одним временем t; после каждого шага между соседями по фронту, разошедшимися > dmax, вставляется середина (линейно по (x, p)) —
вставка в ТЕКУЩЕМ времени, точность по θ не нужна. Сближение < dmin — прореживание. Отсев: сетка лучшего времени G; точка, пришедшая в ячейку
позже G + margin, неоптимальна навсегда (принцип оптимальности) — удаляется. V_char(q) = время записей фронта рядом с q.
Проверка: V_char против реальных времён агентов (exact_switch P0/P2/P3, бабочки) — должно быть V_char ≤ T агента и близко."""
import numpy as np, json, sys, time, os
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from pend_cost_grid import wrap, UM
from pend_char_adapt import terminal, F
WMAX = 4.; KC = float(os.environ.get('KC', 3))

def march(S=16., h=.005, dmax=.03, dmin=.008, rec=.02, margin=.15, n0=4000, cell=.02, log=None):
    th = np.linspace(0, 8, n0, endpoint=False); z, ok = terminal(th); Z = z.T.copy()                    # (n, 4)
    link = ok & np.roll(ok, -1); Z, link = Z[ok], link[ok]                                               # link[i]: i и i+1 (по кольцу) — соседи
    nx, nw = int(2 * np.pi / cell), int(2 * WMAX / cell); G = np.full((nx, nw), np.inf)
    P, T = [np.c_[wrap(Z[:, 0]), Z[:, 1:]]], [np.zeros(len(Z))]; every = int(round(rec / h)); t = 0.; ins = 0
    for i in range(int(S / h)):
        zz = Z.T; k1 = -F(zz); k2 = -F(zz + h / 2 * k1); k3 = -F(zz + h / 2 * k2); k4 = -F(zz + h * k3); Z = (zz + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).T; t += h
        ix = ((wrap(Z[:, 0]) + np.pi) / cell).astype(int) % nx; iw = np.clip(((Z[:, 1] + WMAX) / cell).astype(int), 0, nw - 1)
        sp = np.hypot(Z[:, 1], np.sin(Z[:, 0]) - UM * np.sign(Z[:, 3]))                                 # скорость в фазовой плоскости: у равновесий V меняется на ячейке сильно
        dead = (np.abs(Z[:, 1]) > WMAX) | (t > G[ix, iw] + margin + KC * cell / np.maximum(sp, 1e-3)); np.minimum.at(G, (ix[~dead], iw[~dead]), t)
        if dead.any():
            link = link & ~dead & ~np.roll(dead, -1); pd = np.roll(dead, 1)                               # связь через удалённую точку рвётся
            Z, link = Z[~dead], link[~dead]
        if not len(Z): break
        nxt = np.roll(np.arange(len(Z)), -1); d = np.hypot(wrap(Z[nxt, 0] - Z[:, 0]), Z[nxt, 1] - Z[:, 1])
        far = link & (d > dmax)
        if far.any():
            j = np.flatnonzero(far); mid = Z[j].copy(); mid[:, 0] += wrap(Z[nxt[j], 0] - Z[j, 0]) / 2; mid[:, 1:] = (Z[j, 1:] + Z[nxt[j], 1:]) / 2
            s0 = np.sign(mid[:, 3]); mid[:, 3] = np.where(np.abs(mid[:, 3]) < 1e-12, 1e-12 * np.where(s0 == 0, 1, s0), mid[:, 3])
            Z = np.insert(Z, j + 1, mid, axis=0); link = np.insert(link, j + 1, True); ins += len(j)
        elif (i % 4) == 0:
            near = link & (d < dmin) & np.roll(link, -1); near[1::2] = False; near &= ~np.roll(near, 1)   # убрать i+1, если i и i+1 слиплись
            if near.any(): drop = np.roll(near, 1); Z, link = Z[~drop], link[~drop]
        if (i + 1) % every == 0: P.append(np.c_[wrap(Z[:, 0]), Z[:, 1:]]); T.append(np.full(len(Z), t))
        if log and (i + 1) % int(1 / h) == 0: print('t', round(t, 2), 'front', len(Z), 'links', int(link.sum()), 'inserted', ins, 'cells', round(float(np.isfinite(G).mean()), 3), flush=True)
    return np.concatenate(P), np.concatenate(T), G

class VChar:
    def __init__(s, P, T): s.Z = np.r_[P, P, P]; P = P[:, :2]; s.P, s.T = np.r_[P, P + [2 * np.pi, 0], P - [2 * np.pi, 0]], np.r_[T, T, T]; s.tree = cKDTree(s.P)
    def vmin(s, Q, r=.02, k=64):
        d, i = s.tree.query(Q, k=k, distance_upper_bound=r); return np.where(np.isfinite(d), s.T[np.minimum(i, len(s.T) - 1)], np.inf).min(1)
    def vnear(s, Q, k=8):                                                                               # среднее по k ближайшим (вес 1/d) среди записей с t ≤ min+0.1
        d, i = s.tree.query(Q, k=k); t = s.T[i]; w = (t <= t.min(1, keepdims=True) + .1) / (d + 1e-6); return (w * t).sum(1) / w.sum(1), d[:, 0]

def st(r): return dict(n=int(len(r)), mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), min=round(float(r.min()), 4), max=round(float(r.max()), 4))
if __name__ == '__main__':
    dmax = float(sys.argv[1]) if len(sys.argv) > 1 else .03
    t0 = time.time(); P, T, G = march(dmax=dmax, dmin=dmax / 4, log=True); V = VChar(P, T); print('samples', len(P), 'sec', round(time.time() - t0), flush=True)
    np.savez_compressed('pend_char_front_d%g.npz' % dmax, P=P.astype(np.float32), T=T.astype(np.float32), G=G.astype(np.float32))
    rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)
    vm, (vn, dn) = V.vmin(Q), V.vnear(Q); out = dict(dmax=dmax, covered_r02=round(float(np.isfinite(vm).mean()), 3), nearest_dist_max=round(float(dn.max()), 4), vmin_over_vnear=st(vm[np.isfinite(vm)] / vn[np.isfinite(vm)]))
    A = np.load('exact_switch_pend_P0_P2_P30.5.npy'); B = np.load('butterfly_pend_5000_tau0.3_r0.1_tl2.npy'); far = B[0] > .05
    best40 = np.minimum.reduce([A[0], A[1], A[2], B[1][:40], B[3][:40]]); m = np.isfinite(best40) & far[:40]
    out['best_known40_over_Vchar'] = st(best40[m] / vn[:40][m]); out['P3_over_Vchar'] = st((A[2] / vn[:40])[m & np.isfinite(A[2])]); out['P0_over_Vchar'] = st((A[0] / vn[:40])[m & np.isfinite(A[0])])
    out['butterfly5000_over_Vchar_100'] = st((B[3] / vn)[far & np.isfinite(B[3])]); out['grid_agent_over_Vchar_100'] = st((B[1] / vn)[far & np.isfinite(B[1])]); out['Vgrid_over_Vchar_100'] = st((B[0] / vn)[far])
    out['sec'] = round(time.time() - t0); print(json.dumps(out, ensure_ascii=False), flush=True)
