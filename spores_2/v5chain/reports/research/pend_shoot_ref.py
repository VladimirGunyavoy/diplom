"""research hub-v5chain-research-9: эталон T* маятника прямой стрельбой — перебор bang-bang из ≤ 3 дуг (знак первой дуги, τ1, τ2 по сетке dtau),
последняя дуга — «полоса»: множество точек, из которых постоянное u ведёт в цель (назад от границы цели, KD-дерево, допуск eps), затем
кандидаты по возрастанию времени проверяются ЧЕСТНОЙ симуляцией (rk4, вход в коробку ±.1). Результат — реальная траектория ⇒ строгая верхняя
оценка T*; близка к T*, если оптимум имеет ≤ 2 переключений (точность ~dtau)."""
import numpy as np, sys, time, json
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from pend_cost_grid import step, wrap, UM
R0 = .1
def band(u, S=9., h=.005, nb=300):
    e = np.linspace(-R0, R0, nb); b = np.r_[np.c_[e, 0 * e + R0], np.c_[e, 0 * e - R0], np.c_[0 * e + R0, e], np.c_[0 * e - R0, e]]; x, w = b[:, 0].copy(), b[:, 1].copy(); P, T = [], []
    for i in range(int(S / h)):
        x, w = step(x, w, u, -h); m = (np.abs(w) <= 4) & ~((np.abs(wrap(x)) < R0) & (np.abs(w) < R0)); P.append(np.c_[wrap(x), w][m]); T.append(np.full(m.sum(), (i + 1) * h))
    P, T = np.concatenate(P), np.concatenate(T); P3 = np.r_[P, P + [2 * np.pi, 0], P - [2 * np.pi, 0]]; return cKDTree(P3), np.r_[T, T, T]
def ingoal(x, w): return (np.abs(wrap(x)) <= R0 + 1e-9) & (np.abs(w) <= R0 + 1e-9)
def verify(q, s, t1, t2, tmax3, h=.005):
    """честно: дуга s·UM время t1, −s·UM время t2, s·UM до цели (≤ tmax3). Время входа в цель или inf."""
    x, w = np.array([q[0]]), np.array([q[1]]); t = 0.
    for u, d in ((s * UM, t1), (-s * UM, t2), (s * UM, tmax3)):
        n = int(np.ceil(d / h - 1e-9)); hh = d / n if n else 0.
        for _ in range(n):
            x, w = step(x, w, u, hh); t += hh
            if ingoal(x, w)[0]: return t
    return np.inf
def shoot(q, B, dtau=.02, S1=9., S2=9., eps=.004, ntry=300):
    if ingoal(q[0], q[1]): return 0., None
    cand = []
    for s in (1., -1.):
        n1 = int(S1 / dtau) + 1; x, w = np.array([q[0]]), np.array([q[1]]); X1, W1 = [x[0]], [w[0]]
        for _ in range(n1 - 1): x, w = step(x, w, s * UM, dtau); X1.append(x[0]); W1.append(w[0])
        x, w = np.array(X1), np.array(W1); ok = np.abs(w) <= 4; X, Wv = [x.copy()], [w.copy()]
        for _ in range(int(S2 / dtau)): x, w = step(x, w, -s * UM, dtau); X.append(x.copy()); Wv.append(w.copy())
        X, Wv = np.array(X), np.array(Wv)                                                    # (n2, n1): после τ1 = j·dtau, τ2 = i·dtau
        d, i = B[s][0].query(np.c_[wrap(X.ravel()), Wv.ravel()], distance_upper_bound=eps); T3 = np.where(np.isfinite(d), B[s][1][np.minimum(i, len(B[s][1]) - 1)], np.inf).reshape(X.shape)
        i2, i1 = np.nonzero(np.isfinite(T3)); tot = i1 * dtau + i2 * dtau + T3[i2, i1]; cand += [(tot[k], s, i1[k] * dtau, i2[k] * dtau, T3[i2[k], i1[k]]) for k in np.argsort(tot)[:ntry]]
    cand.sort()
    for tot, s, t1, t2, t3 in cand[:ntry]:
        T = verify(q, s, t1, t2, t3 + .15)
        if np.isfinite(T): return T, (s, t1, t2, T - t1 - t2)
    return np.inf, None
if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t0 = time.time(); B = {s: band(s * UM) for s in (1., -1.)}; print('bands', [len(B[s][1]) for s in B], 'sec', round(time.time() - t0), flush=True)
    rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)[:N]; R = []
    for i, q in enumerate(Q):
        T, pl = shoot(q, B); R.append((T,) + (pl if pl else (0, 0, 0, 0)))
        if i % 10 == 9: print(i + 1, 'sec', round(time.time() - t0), flush=True)
    R = np.array(R); np.save('pend_shoot_ref.npy', R)
    Bf = np.load('butterfly_pend_5000_tau0.3_r0.1_tl2.npy'); A = np.load('exact_switch_pend_P0_P2_P30.5.npy'); C = np.load('pend_char_agent_d0.015_m0.5.npy')
    best = np.minimum(Bf[1], Bf[3])[:N]; n4 = min(N, 40); best[:n4] = np.minimum.reduce([best[:n4], A[0][:n4], A[1][:n4], A[2][:n4]]); best = np.minimum(best, C[1][:N]); T = R[:, 0]; f = np.isfinite(T) & (Bf[0][:N] > .05)
    def st(r): return dict(n=int(len(r)), mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), min=round(float(r.min()), 4), max=round(float(r.max()), 4))
    print(json.dumps(dict(found=round(float(np.isfinite(T).mean()), 3), shoot_over_best_known=st(T[f] / best[f]), shoot_better=int((T[f] < best[f] - 1e-3).sum()), shoot_over_Vchar=st(T[f] / C[0][:N][f]),
                          ref=dict(note='ref = min(shoot, best_known)', butterfly5000_over_ref=st((Bf[3][:N] / np.minimum(T, best))[Bf[0][:N] > .05]), Vchar_over_ref=st((C[0][:N] / np.minimum(T, best))[Bf[0][:N] > .05])), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    for i in (5, 48, 61, 62, 65, 84):
        if i < N: print(i, np.round(Q[i], 2), 'shoot', np.round(R[i], 3), 'best', round(float(best[i]), 3), 'Vchar', round(float(C[0][i]), 3))
