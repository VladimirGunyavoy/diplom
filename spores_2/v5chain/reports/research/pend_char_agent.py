"""research hub-v5chain-research-9: честная проверка V характеристик (pend_char_front.py) замкнутым агентом: в точке — запись фронта с min t в радиусе r,
u = −UM·sign(p_ω) этой записи, держать dt, шаг rk4. Реальное время агента — верхняя оценка T*; если T_агента ≈ V_char, то V_char — эталон."""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
import pend_char_front as M
from pend_cost_grid import step, wrap, UM
def agent(V, Q, r=.03, dt=.02, tmax=30., k=256):
    Y = np.array(Q, float); n = len(Y); T = np.full(n, np.inf); act = np.ones(n, bool); t = 0.; sw = np.zeros(n, int); pu = np.zeros(n); V0 = None
    while act.any() and t < tmax:
        a = np.flatnonzero(act); q = np.c_[wrap(Y[a, 0]), Y[a, 1]]; d, i = V.tree.query(q, k=k, distance_upper_bound=r); tt = np.where(np.isfinite(d), V.T[np.minimum(i, len(V.T) - 1)], np.inf)
        j = i[np.arange(len(a)), tt.argmin(1)]; lost = ~np.isfinite(tt.min(1)); jn = V.tree.query(q)[1]; j = np.where(lost, jn, j)       # нет записей рядом — ближайшая
        if V0 is None: V0 = tt.min(1)
        u = -UM * np.sign(V.Z[j, 3]); sw[a] += (pu[a] != 0) & (u != pu[a]); pu[a] = u
        for _ in range(2):
            x, w = step(Y[a, 0], Y[a, 1], u, dt / 2); Y[a, 0], Y[a, 1] = x, w
            g = (np.abs(wrap(x)) <= .1 + 1e-9) & (np.abs(w) <= .1 + 1e-9) & act[a]; T[a[g]] = t + dt / 2 * (_ + 1); act[a[g]] = False
        t += dt
    return T, sw, V0
if __name__ == '__main__':
    dmax = float(sys.argv[1]); margin = float(sys.argv[2]); M.KC = 10
    t0 = time.time(); P, T, G = M.march(dmax=dmax, dmin=dmax / 4, margin=margin, S=13.); V = M.VChar(P, T); tm = time.time() - t0
    rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)
    Ta, sw, V0 = agent(V, Q); B = np.load('butterfly_pend_5000_tau0.3_r0.1_tl2.npy'); A = np.load('exact_switch_pend_P0_P2_P30.5.npy'); best = np.minimum(B[1], B[3]); best[:40] = np.minimum.reduce([best[:40], A[0], A[1], A[2]])
    f = np.isfinite(Ta) & (B[0] > .05); r = Ta[f] / V0[f]; rb = Ta[f] / best[f]; st = M.st
    print(json.dumps(dict(dmax=dmax, margin=margin, samples=len(P), sec_march=round(tm, 1), reach=round(float(f.mean()), 3), Tagent_over_Vchar=st(r), Tagent_over_best_known=st(rb), sw_mean=round(float(sw[f].mean()), 2), sw_max=int(sw[f].max()),
                          best_known_over_Vchar=st(best[f] / V0[f]), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    np.save('pend_char_agent_d%g_m%g.npy' % (dmax, margin), np.array([V0, Ta, sw, best]))
    w = np.argsort(-(Ta / V0))[:5]; print('худшие T/V:', [(int(i), np.round(Q[i], 2).tolist(), round(float(V0[i]), 2), round(float(Ta[i]), 2), round(float(best[i]), 2)) for i in w])
