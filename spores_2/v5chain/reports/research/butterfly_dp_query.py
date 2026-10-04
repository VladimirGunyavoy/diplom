"""research hub-v5chain-research-10: запрос к выращенному атласу бабочек двойного маятника (butterfly_dp_grow.py, TR=1) через ПРЯМОЕ мини-дерево.
Из произвольной точки дуга с постоянным τ попадает в отрезок споры редко (трубка приходов 3D, отрезок 1D), поэтому запрос строит прямое
дерево глубины DEP: дуги τ ∈ {−1, 0, 1}², t ∈ TS (27 детей на уровень), каждая вершина пробует пары в атлас; V(y) = min [Σt + V_атлас].
Агент проходит первую дугу лучшей ветви целиком (модель точная) и перепланирует. Концепция «прямое + обратное дерево» (concepts.md)."""
import numpy as np, sys, os, time, json
sys.path.insert(0, '.')
from butterfly_dp import wrap, flow, ingoal, BIG, G
from butterfly_dp_atlas import Atlas, R, MN, SNAPA, snap
from butterfly_dp_grow import emb
from scipy.spatial import cKDTree
TRAJ = []; CHAIN = []; PEND = []; DK = []; DS = []; ONK = [-1, 0.]; WCHK = int(os.environ.get('WCHK', 0)); TS = np.array([float(x) for x in os.environ.get('TS', '.25,.5,.9').split(',')]); DEP = int(os.environ.get('DEP', 2)); TOPE = int(os.environ.get('TOPE', 5)); ALLN = int(os.environ.get('ALLN', 1)); PLANFB = int(os.environ.get('PLANFB', 1)); FORCEPLAN = int(os.environ.get('FORCEPLAN', 0))
UU = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], float)
def load(path):
    d = np.load(path, allow_pickle=True); A = Atlas.__new__(Atlas); A.C, A.n, A.V = d['C'], d['n'], d['V']; A.K = len(A.C)
    A.sn = np.linspace(-R, R, MN); A.X = emb(A.C); A.tree = cKDTree(A.X)
    A.e = [np.asarray(x, dt) for x, dt in zip(d['e'], (int, int, int, float, float))]; return A
def direct(A, Y):
    """Лучшее значение одной дугой в атлас для каждой строки Y (n, 4); и (t, u1, u2) этой дуги."""
    out = np.full(len(Y), BIG); arc = np.zeros((len(Y), 3))
    DK[:] = [-1] * len(Y); DS[:] = [0.] * len(Y)                                             # research-11: целевая спора и s лучшей дуги — агент после прямой дуги «стоит на споре»
    if not len(Y): return out, arc
    gi = ingoal(Y.T); out[gi] = 0.
    iy, k, sv, t = A.pairs(Y); iy, k = iy.astype(int), k.astype(int)
    if not len(iy): return out, arc
    f_ = (sv + R) / (2 * R) * (MN - 1); j0 = np.clip(np.floor(f_).astype(int), 0, MN - 2); a = f_ - j0
    V0, V1 = A.V[k, j0], A.V[k, j0 + 1]; a = snap(a); val = t + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - SNAPA)) | ((V1 >= BIG / 2) & (a > SNAPA))] = BIG
    o = np.lexsort((val, iy)); iy, k, val, t, sv = iy[o], k[o], val[o], t[o], sv[o]; first = np.r_[True, iy[1:] != iy[:-1]]
    from butterfly_dp import solve_arcs
    if WCHK:                                                                                 # research-11: |w| ≤ 3 ВДОЛЬ дуги (g = 2: FORCEPLAN давал wmax 4.16) — до 4 лучших кандидатов на точку
        rank = np.arange(len(iy)) - np.maximum.accumulate(np.where(first, np.arange(len(iy)), 0)); tries = (rank < 4) & (val < BIG / 2)
        for i, kk, v, tt, ss in zip(iy[tries], k[tries], val[tries], t[tries], sv[tries]):
            if out[i] < BIG / 2: continue
            _, u1, u2, _, ok = solve_arcs(Y[i][:, None], A.C[kk][:, None], A.n[kk][:, None]); z = Y[i][:, None].copy(); wm = 0.
            for _ in range(30): z = flow(z, u1, u2, tt / 30, n=1); wm = max(wm, float(np.abs(z[2:]).max()))
            if wm <= 3.: out[i] = v; arc[i] = (tt, u1[0], u2[0]); DK[i] = int(kk); DS[i] = float(ss)
        return out, arc
    for i, kk, v, tt, ss in zip(iy[first], k[first], val[first], t[first], sv[first]):
        if v < out[i]:
            DK[i] = int(kk); DS[i] = float(ss)
            _, u1, u2, _, ok = solve_arcs(Y[i][:, None], A.C[kk][:, None], A.n[kk][:, None]); out[i] = v; arc[i] = (tt, u1[0], u2[0])
    return out, arc
def children(Y):
    n = len(Y); U = np.repeat(UU, len(TS), 0); T = np.tile(TS, len(UU)); m = len(T)
    Yr = np.repeat(Y, m, 0); Z = flow(Yr.T, np.tile(U[:, 0], n), np.tile(U[:, 1], n), np.tile(T, n)).T; Z[:, :2] = wrap(Z[:, :2])
    return Z, np.tile(T, n), np.tile(U, (n, 1))
def plan(A, y):
    """V(y) и первая дуга (t, u1, u2): прямо в атлас или через дерево глубины DEP."""
    v, arc = direct(A, y[None]); best, barc = v[0], arc[0]
    ONK[:] = [DK[0], DS[0]]; CHAIN[:] = [tuple(arc[0])]                                                               # research-11: вся лучшая цепочка дуг — запас агенту, если следующий план не найдётся
    lv = [(y[None], None, None, None)]; Y = y[None]; par = None
    for d in range(DEP):
        Z, T, U = children(Y); ok = np.isfinite(Z).all(1) & (np.abs(Z[:, 2:]) <= 3).all(1)
        if WCHK:                                                                             # и у дуг мини-дерева — в середине
            for fr in (.2, .4, .6, .8):
                Zm = flow(np.repeat(Y, len(TS) * len(UU), 0).T, U[:, 0], U[:, 1], T * fr).T; ok &= np.isfinite(Zm).all(1) & (np.abs(Zm[:, 2:]) <= 3).all(1)
        root = np.repeat(np.arange(len(Y)), len(TS) * len(UU)) if par is None else np.repeat(par, len(TS) * len(UU))
        cost = np.repeat(np.zeros(len(Y)) if d == 0 else Ccost, len(TS) * len(UU)) + T
        first = np.c_[T, U] if d == 0 else np.repeat(Farc, len(TS) * len(UU), 0)
        H = np.c_[T, U][:, None, :] if d == 0 else np.concatenate([np.repeat(Hp, len(TS) * len(UU), 0), np.c_[T, U][:, None, :]], 1); H = H[ok]
        Z, cost, first = Z[ok], cost[ok], first[ok]; vz, az = direct(A, Z); tot = cost + vz; i = int(np.argmin(tot))
        if tot[i] < best: best, barc = tot[i], first[i]; CHAIN[:] = [tuple(h) for h in H[i]] + [tuple(az[i])]; ONK[:] = [-1, 0.]
        Y, Ccost, Farc, par, Hp = Z, cost, first, None, H
    return best, barc
def rollout(A, y0, tmax=25.):
    y = np.array(y0, float); t = 0.; wmax = 0.; nplan = 0; V0 = None
    while t < tmax:
        if ingoal(y[:, None])[0]: return t, wmax, nplan, V0
        J, (ta, u1, u2) = plan(A, y); nplan += 1; V0 = J if V0 is None else V0
        if J >= BIG / 2: return np.inf, wmax, nplan, V0
        for _ in range(max(1, int(round(ta / .01)))):
            y = flow(y[:, None], np.array([u1]), np.array([u2]), ta / max(1, int(round(ta / .01))), n=1)[:, 0]; t += ta / max(1, int(round(ta / .01))); wmax = max(wmax, np.abs(y[2:]).max())
            if ingoal(y[:, None])[0]: return t, wmax, nplan, V0
    return np.inf, wmax, nplan, V0
if __name__ == '__main__':
    A = load(sys.argv[1]); t0 = time.time(); fin = A.V[:, MN // 2] < BIG / 2
    q = np.array([-np.pi / 2, 0, 0, 0]); dd = np.linalg.norm(A.X[fin] - emb(q[None]), axis=1)
    print(json.dumps(dict(atlas=sys.argv[1], spores=A.K, finite_centers=round(float(fin.mean()), 3), near_start=round(float(dd.min()), 3), DEP=DEP, TS=TS.tolist())), flush=True)
    T, wm, npl, V0 = rollout(A, q)
    print(json.dumps(dict(query='висит→вверх', V=round(float(V0), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), v6=5.28, wmax=round(float(wm), 2), plans=npl, sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    rng = np.random.default_rng(3); Q = np.c_[rng.uniform(-np.pi, np.pi, (int(os.environ.get('NQ', 20)), 2)), rng.uniform(-1, 1, (int(os.environ.get('NQ', 20)), 2))]; res = []
    for qq in Q: T, wm, npl, V0 = rollout(A, qq); res.append((V0, T, wm, npl))
    R_ = np.array(res, float); f_ = np.isfinite(R_[:, 1]) & (R_[:, 0] < BIG / 2)
    print(json.dumps(dict(random_reach=round(float(f_.mean()), 2), V_finite=round(float((R_[:, 0] < BIG / 2).mean()), 2),
                          T_over_V=dict(mean=round(float((R_[f_, 1] / R_[f_, 0]).mean()), 3), max=round(float((R_[f_, 1] / R_[f_, 0]).max()), 3)) if f_.any() else None,
                          T_med=round(float(np.median(R_[f_, 1])), 2) if f_.any() else None, wmax_over3=int((R_[:, 2] > 3).sum()), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    np.save(sys.argv[1].replace('.npz', '_q_dep%d.npy' % DEP), R_)

# --- research-10: агент «по рёбрам» — тёплый Ньютон из точки на отрезке к цели ребра ближайшего узла ---
from butterfly_dp import f as fdyn
def newton_warm(y, c, nn, t, u1, u2, s, it=12):
    for _ in range(it):
        Z = flow(y[:, None], np.array([u1]), np.array([u2]), t)[:, 0]; Rz = Z - (c + s * nn); Rz[:2] = wrap(Rz[:2])
        e = 1e-6; Z1 = flow(y[:, None], np.array([u1 + e]), np.array([u2]), t)[:, 0]; Z2 = flow(y[:, None], np.array([u1]), np.array([u2 + e]), t)[:, 0]
        J = np.c_[fdyn(Z[:, None], np.array([u1]), np.array([u2]))[:, 0], (Z1 - Z) / e, (Z2 - Z) / e, -nn]
        try: d = np.linalg.solve(J, Rz)
        except np.linalg.LinAlgError: return None
        t, u1, u2, s = t - d[0], u1 - d[1], u2 - d[2], s - d[3]
        if not np.isfinite([t, u1, u2, s]).all() or t <= 1e-3 or t > 3: return None
    Z = flow(y[:, None], np.array([u1]), np.array([u2]), t)[:, 0]; Rz = Z - (c + s * nn); Rz[:2] = wrap(Rz[:2])
    if np.abs(Rz).max() > 1e-7 or abs(u1) > 1 + 1e-9 or abs(u2) > 1 + 1e-9 or abs(s) > R: return None
    return t, u1, u2, s
def arc_wmax(y, u1, u2, t, n=30):
    z = np.array(y, float)[:, None]; wm = 0.
    for _ in range(n): z = flow(z, np.array([u1]), np.array([u2]), t / n, n=1); wm = max(wm, float(np.abs(z[2:]).max()))
    return wm
def node_edges(A):
    """Для каждого узла (k, j): лучшее ребро (k2, t, u1, u2, s2) по V атласа."""
    iy, k, j0, a, t = A.e; V = A.V.reshape(-1); val = t + (1 - a) * V[k * MN + j0] + a * V[k * MN + j0 + 1]
    o = np.lexsort((val, iy)); E = {}
    for i in o:                                                                          # research-10: до TOPE рёбер на узел — запасные, если тёплый Ньютон не сошёлся
        if val[i] < BIG / 2 and len(E.setdefault(int(iy[i]), [])) < TOPE: E[int(iy[i])].append((int(k[i]), float(t[i]), float(((j0[i] + a[i]) / (MN - 1)) * 2 * R - R)))
    return E
def edge_value(A, k2, s2):
    f_ = (s2 + R) / (2 * R) * (MN - 1); j0 = min(max(int(np.floor(f_)), 0), MN - 2); a = float(snap(f_ - j0)); return (1 - a) * A.V[k2, j0] + a * A.V[k2, j0 + 1]
def rollout_edges(A, k, s, tmax=40.):
    """Старт на отрезке споры k в точке s. Возвращает T, число дуг, wmax."""
    from butterfly_dp import solve_arcs
    E = node_edges(A) if not hasattr(A, 'E') else A.E; A.E = E; y = A.C[k] + s * A.n[k]; T = 0.; arcs = 0; wmax = 0.
    while T < tmax:
        if ingoal(y[:, None])[0]: return T, arcs, wmax
        kp = k
        if FORCEPLAN: k = None                                                               # FORCEPLAN: всегда мини-дерево из точки (проверка: даёт ли оно короче графа)
        if k is None:                                                                        # вне отрезка: прямое мини-дерево из точки (plan), первая дуга целиком
            J, (tt, u1, u2) = plan(A, y)
            fail = J >= BIG / 2 or tt <= 0
            onk = (-1, 0.)
            if fail and PEND: tt, u1, u2 = PEND.pop(0); fail = tt <= 0; A.npend = getattr(A, 'npend', 0) + 1     # план не найден — следующая дуга прошлого плана (открыто)
            elif not fail: PEND[:] = CHAIN[1:]; onk = tuple(ONK)
            if fail and kp is None: return np.inf, arcs, wmax
            if not fail:
                nst = max(6, int(tt / .01))
                TRAJ.append((y.copy(), float(u1), float(u2), float(tt)))                             # research-11: дуги пути — для пула траекторий (dp_g2_pool.py)
                for _ in range(nst):
                    y = flow(y[:, None], np.array([u1]), np.array([u2]), tt / nst, n=1)[:, 0]; wmax = max(wmax, np.abs(y[2:]).max())
                    if ingoal(y[:, None])[0]: return T + tt * (_ + 1) / nst, arcs + 1, wmax
                T += tt; arcs += 1; y[:2] = wrap(y[:2]); A.nplan = getattr(A, 'nplan', 0) + 1
                if onk[0] >= 0: k, s = int(onk[0]), float(onk[1])                                    # прямая дуга в атлас: стоим на споре onk — при неудаче плана пойдём по её рёбрам
                continue
            k = kp                                                                          # research-11: мини-дерево не нашло пути — шаг по рёбрам споры, на которой стоим
        f_ = (s + R) / (2 * R) * (MN - 1); js = sorted({min(max(int(np.floor(f_)), 0), MN - 1), min(max(int(np.ceil(f_)), 0), MN - 1)}); best = None
        if ALLN: js = list(np.argsort(A.V[k]))                                               # ALLN: рёбра всех узлов споры по возрастанию V узла
        for j in js:
            for k2, t0, s20 in E.get(k * MN + j, []):
                P = A.C[k] + A.sn[j] * A.n[k]
                tt, u1, u2, s2, ok = solve_arcs(P[:, None], A.C[k2][:, None], A.n[k2][:, None])
                if not ok[0]: continue
                r = newton_warm(y, A.C[k2], A.n[k2], tt[0], u1[0], u2[0], s2[0])
                if r is None: continue
                if WCHK and arc_wmax(y, r[1], r[2], r[0]) > 3.: continue                      # research-11: |w| ≤ 3 вдоль дуги и при ходьбе по рёбрам
                v = r[0] + edge_value(A, k2, r[3])
                if v < BIG / 2 and (best is None or v < best[0]): best = (v, k2, r)
                break                                                                    # первое сошедшееся ребро узла (они по возрастанию цены)
        if best is None and ALLN:                                                            # запас: прямые пары из точки (Ньютон из формулы)
            iy, kk, sv, t_ = A.pairs(y[None])
            for kk_, sv_ in zip(kk.astype(int), sv):
                tt, u1, u2, s2, ok = solve_arcs(y[:, None], A.C[kk_][:, None], A.n[kk_][:, None])
                if ok[0] and abs(s2[0]) <= R and not (WCHK and arc_wmax(y, u1[0], u2[0], tt[0]) > 3.):
                    v = tt[0] + edge_value(A, kk_, s2[0])
                    if v < BIG / 2 and (best is None or v < best[0]): best = (v, kk_, (tt[0], u1[0], u2[0], s2[0]))
        if best is None:
            if not PLANFB: return np.inf, arcs, wmax
            k = None; continue
        _, k2, (tt, u1, u2, s2) = best; nst = max(6, int(tt / .01)); yy = y.copy()
        for _ in range(nst):
            yy = flow(yy[:, None], np.array([u1]), np.array([u2]), tt / nst, n=1)[:, 0]; wmax = max(wmax, np.abs(yy[2:]).max())
            if ingoal(yy[:, None])[0]: return T + tt * (_ + 1) / nst, arcs + 1, wmax
        T += tt; arcs += 1; k, s = k2, s2; y = A.C[k] + s * A.n[k]
    return np.inf, arcs, wmax
