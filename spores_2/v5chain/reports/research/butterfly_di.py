"""research hub-v5chain-research-8: споры-«бабочки» на ДИ (идея пользователя 2026-10-02): спора = точка c + нормальный отрезок S_c(s) = c + s·e_v
(нормаль к дрейфу f(c,0) = (v,0)), |s| ≤ r, узлы V на отрезке. Все управления сразу: из точки y в точку p отрезка ведёт ровно одна дуга
с постоянным u — t = 2(x_p − x_y)/(v_y + v_p), u = (v_p − v_y)/t (годна при 0 < t ≤ τ, |u| ≤ 1). Беллман внутри карты:
V(y) = min по спорам c и s ∈ [−r, r] [t(y → S_c(s)) + V_c(s)], V_c — линейно по узлам. Без формулы T* внутри метода (T* — только оценка).
Агент: в y — лучшая (c, s) ⇒ (t, u); режим 'dt' — держать u шаг dt и перепланировать; режим 'arc' — держать u всю дугу до отрезка."""
import numpy as np, json, sys, time
from scipy.spatial import cKDTree
sys.path.insert(0, '.'); from v7_faces_di import tstar_box
L, RHO, BIG = 2.5, .1, 1e3

class Butterfly:
    def __init__(s, N=800, r=.1, m=5, tau=.5, ns=21, seed=0, nodes_only=False):
        rng = np.random.default_rng(seed)
        g = np.linspace(-RHO, RHO, 3); gc = np.array([(a, b) for a in g for b in g])          # споры в цели (V = 0)
        s.C = np.r_[gc, rng.uniform(-L, L, (N, 2))]; s.K = len(s.C); s.ngoal = len(gc)
        s.r, s.m, s.tau = r, m, tau; s.sn = np.linspace(-r, r, m); s.sq = s.sn.copy() if nodes_only else np.linspace(-r, r, ns)   # nodes_only: дуги только в узлы (без интерполяции ⇒ V ≥ T*)
        s.V = np.full((s.K, m), BIG)
        s.ingoal = (np.abs(s.C[:, None, 0]) <= RHO) & (np.abs(s.C[:, None, 1] + s.sn[None, :]) <= RHO); s.V[s.ingoal] = 0.   # V = 0 только у узлов внутри цели
        s.tree = cKDTree(s.C)
    def arcs(s, y, c, sq):
        """время/управление дуги из y (2,) в точки c + sq·e_v (векторно по c (k,2), sq (q,))."""
        xp, vp = c[:, None, 0] + 0 * sq, c[:, None, 1] + sq[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            den = y[1] + vp; t = np.where(np.abs(den) > 1e-9, 2 * (xp - y[0]) / np.where(np.abs(den) > 1e-9, den, 1), np.inf); u = (vp - y[1]) / t
        ok = (t > 1e-6) & (t <= s.tau) & (np.abs(u) <= 1 + 1e-12) & np.isfinite(u)
        return np.where(ok, t, np.inf), u
    def Vc(s, k, sq): return np.stack([np.interp(sq, s.sn, s.V[i]) for i in k])                # V на отрезке споры
    def best(s, y, exclude=-1):
        k = np.array(s.tree.query_ball_point(y, s.tau * (abs(y[1]) + 1) + s.r + .05))
        k = k[k != exclude] if len(k) else k
        if not len(k): return BIG, None
        t, u = s.arcs(y, s.C[k], s.sq); J = t + s.Vc(k, s.sq); i = np.unravel_index(np.argmin(J), J.shape)
        return J[i], (k[i[0]], s.sq[i[1]], t[i], u[i])
    def solve(s, it=200, tol=1e-6):
        for n in range(it):
            Vold = s.V.copy()
            for i in range(s.K):
                for j, sv in enumerate(s.sn):
                    if s.ingoal[i, j]: continue
                    J, _ = s.best(s.C[i] + np.r_[0, sv], exclude=i); s.V[i, j] = min(s.V[i, j], J)
            d = np.max(np.abs(np.minimum(s.V, BIG) - np.minimum(Vold, BIG)))
            if d < tol: break
        s.n_it = n; return s
    def Vq(s, Y): return np.array([0. if (abs(y[0]) <= RHO + 1e-9 and abs(y[1]) <= RHO + 1e-9) else s.best(y)[0] for y in Y])
    def rollout(s, y0, mode='dt', dt=.06, tmax=20.):
        y = np.array(y0, float); t = 0.; pu = None; sw = 0
        while t < tmax:
            if abs(y[0]) <= RHO + 1e-9 and abs(y[1]) <= RHO + 1e-9: return t, sw
            J, b = s.best(y)
            if b is None or J >= BIG / 2: return np.inf, sw
            _, _, ta, u = b; h = min(dt, ta) if mode == 'dt' else ta
            if pu is not None and abs(u - pu) > .5: sw += 1
            pu = u
            for _ in range(8):                                                                   # точное движение, проверка входа в цель
                hh = h / 8; y = np.r_[y[0] + y[1] * hh + u * hh * hh / 2, y[1] + u * hh]; t += hh
                if abs(y[0]) <= RHO + 1e-9 and abs(y[1]) <= RHO + 1e-9: return t, sw
        return np.inf, sw

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 800; NO = len(sys.argv) > 2 and sys.argv[2] == 'nodes'
    t0 = time.time(); B = Butterfly(N=N, m=7, nodes_only=NO).solve(); print('nodes_only', NO); print('spores', B.K, 'iters', B.n_it, 'sec', round(time.time() - t0), 'BIG nodes', round(float((B.V >= BIG / 2).mean()), 3), flush=True)
    rng = np.random.default_rng(1); Q = rng.uniform(-1.5, 1.5, (300, 2)); Ts = tstar_box(Q[:, 0], Q[:, 1], RHO); ok = Ts > .05
    Vq = B.Vq(Q); m = ok & (Vq < BIG / 2); r = Vq[m] / Ts[m]
    print(json.dumps(dict(N=N, V_cover=round(float(m.sum() / ok.sum()), 3), V_over_Tstar=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4),
                          min=round(float(r.min()), 3), max=round(float(r.max()), 3))), ensure_ascii=False), flush=True)
    for mode in ('dt', 'arc'):
        R = [B.rollout(q, mode) for q in Q[ok][:150]]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T); rr = T[f] / Ts[ok][:150][f] if f.any() else np.array([np.nan])
        print(json.dumps(dict(N=N, agent=mode, reach=round(float(f.mean()), 3), T_over_Tstar=dict(mean=round(float(np.nanmean(rr)), 4), med=round(float(np.nanmedian(rr)), 4),
                              max=round(float(np.nanmax(rr)), 3)), sw_med=float(np.median(S))), ensure_ascii=False), flush=True)
