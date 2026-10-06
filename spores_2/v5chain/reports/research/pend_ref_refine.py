"""research-15: эталон маятника для слабого мотора — доводка пути агента по сетке (pend_ref_grid.py, *_ctl.npz) стрельбой по длительностям дуг.
Управление пути → дуги bang-bang (u = ±UM; дребезг короче MERGE и u = 0 сливаются с соседями по большинству) → SLSQP: min Σ τ_k при τ ≥ 0,
конец в коробке цели ±RHO (с запасом .98). Плюс энергия + LQR (раскачка u = UM·sign(ω·(1 − E)), у верха LQR) — вторая верхняя оценка.
Эталон = min(сетка, доводка, энергия+LQR) — всё реальные траектории (верхние оценки T*). Запуск: UM=.15 python3 pend_ref_refine.py"""
import numpy as np, os, json, time
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.linalg import solve_continuous_are
UM = float(os.environ.get('UM', .3)); RHO = float(os.environ.get('RHO', .1)); MERGE = float(os.environ.get('MERGE', .1)); SUB = .01
suf = '' if RHO == .1 else '_R%g' % RHO; Z = np.load('pend_ref_grid_u%g%s_ctl.npz' % (UM, suf)); Q, TG, U, H = Z['Q'], Z['T'], Z['U'], float(Z['H'])
def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
def f(y, u): return np.array([y[1], np.sin(y[0]) + u])
def rk4(y, u, h):
    k1 = f(y, u); k2 = f(y + h / 2 * k1, u); k3 = f(y + h / 2 * k2, u); k4 = f(y + h * k3, u); return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def run(y, signs, taus):
    for s_, t_ in zip(signs, taus):
        n = max(1, int(np.ceil(t_ / SUB))); h = t_ / n
        for _ in range(n): y = rk4(y, s_ * UM, h)
    return y
def ingoal(y): return abs(wrap(y[0])) <= RHO + 1e-9 and abs(y[1]) <= RHO + 1e-9
def arcs(u):
    u = u[~np.isnan(u)]; s = np.sign(u); idx = np.flatnonzero(s != 0)
    if not len(idx): return [], []
    for i in range(len(s)):                                                                    # u = 0 — к ближайшему ненулевому
        if s[i] == 0: s[i] = s[idx[np.argmin(np.abs(idx - i))]]
    sg, tau = [s[0]], [H]
    for v in s[1:]:
        if v == sg[-1]: tau[-1] += H
        else: sg.append(v); tau.append(H)
    ch = True
    while ch and len(tau) > 1:                                                                 # дребезг: самую короткую дугу < MERGE вливаем в соседей
        ch = False; k = int(np.argmin(tau))
        if tau[k] < MERGE:
            ch = True; t_ = tau.pop(k); sg.pop(k)
            if 0 < k < len(tau) + 0 and k - 1 >= 0 and k < len(tau) and sg[k - 1] == sg[k]: tau[k - 1] += t_ + tau.pop(k); sg.pop(k)
            elif k - 1 >= 0: tau[k - 1] += t_
            else: tau[0] += t_
    return sg, tau
def refine(y0, sg, tau):
    if not sg: return np.inf
    cons = [{'type': 'ineq', 'fun': lambda z: np.array([.98 * RHO - abs(wrap(run(y0, sg, z)[0])), .98 * RHO - abs(run(y0, sg, z)[1])])}]
    r = minimize(lambda z: z.sum(), np.array(tau), jac=lambda z: np.ones_like(z), constraints=cons, bounds=[(0, None)] * len(tau), method='SLSQP', options=dict(maxiter=200))
    return float(r.x.sum()) if ingoal(run(y0, sg, r.x)) else np.inf
def energy_lqr(y0, tmax=60.):
    A = np.array([[0, 1], [1, 0]]); B = np.array([[0], [1]]); P = solve_continuous_are(A, B, np.diag([10, 1]), np.array([[.1]])); K = (B.T @ P / .1)[0]
    y = y0.copy(); t = 0.; h = .005
    while t < tmax:
        if ingoal(y): return t
        e = np.array([wrap(y[0]), y[1]]); E = y[1] ** 2 / 2 + np.cos(y[0])                        # верх: E = 1
        u = np.clip(-K @ e, -UM, UM) if e @ P @ e < .3 else UM * np.sign(y[1] * (1 - E) + 1e-12)
        y = rk4(y, u, h); t += h
    return np.inf
def one(i):
    q = Q[i].astype(float); sg, tau = arcs(U[:, i]); tr = refine(q, sg, tau) if np.isfinite(TG[i]) else np.inf; return (TG[i], tr, energy_lqr(q), len(sg))
if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(int(os.environ.get('NP', 30))) as pool: R = np.array(list(tqdm(pool.imap(one, range(len(Q))), total=len(Q), desc='refine u=%g' % UM, mininterval=float(os.environ.get('TQDM_MI', 10)))))
    best = np.nanmin(R[:, :3], 1); np.save('pend_ref_best_u%g%s.npy' % (UM, suf), best)
    out = dict(UM=UM, sec=round(time.time() - t0), arcs_med=float(np.median(R[:, 3])), refine_ok=int(np.isfinite(R[:, 1]).sum()), elqr_ok=int(np.isfinite(R[:, 2]).sum()),
               refine_vs_grid=round(float(np.nanmean(np.where(np.isfinite(R[:, 1]), R[:, 1] / R[:, 0], np.nan))), 4), elqr_vs_best=round(float(np.nanmean(np.where(np.isfinite(R[:, 2]), R[:, 2] / best, np.nan))), 4))
    if UM == .3 and RHO == .1 and os.path.exists('pend_ref_T.npy'):
        S = np.load('pend_ref_T.npy'); ok = np.isfinite(S) & (S > .05); r = best[ok] / S[ok]; out.update(best_vs_shoot_mean=round(float(r.mean()), 4), best_vs_shoot_min=round(float(r.min()), 4), best_vs_shoot_max=round(float(r.max()), 4))
    print(json.dumps(out), flush=True)
