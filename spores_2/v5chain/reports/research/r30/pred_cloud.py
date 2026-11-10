"""ENV3 п.1 (research-30): ошибка прогноза положения дисков journal-2026 и покрытие гауссова облака.
Закон (агенту неизвестен): c = base + (Ax sin(W t+phi), Ay cos(.73 W t+1.7 phi)), dt .08, W = 2pi/(180 dt).
Прогнозы по наблюдениям p(t), p(t-dt): snap m=p; vel m=p+w*tau. Облако: sigma^2 = q tau^3/3 (белое ускорение),
q калибруется онлайн по прошлому окну (остатки vel-прогноза на лаге tc). Альтернатива: sigma = s2*tau^2 (ограниченное ускорение)."""
import numpy as np
rng = np.random.default_rng(0)
dt = .08; W = 2*np.pi/(180*dt)
taus = np.array([.5, 1., 2., 3.]); kt = np.round(taus/dt).astype(int)
def traj(mode, n_ep=400, T=60.):
    t = np.arange(0, T, dt); P = []
    for e in range(n_ep):
        for i in range(3):
            phi = .35+2*np.pi*i/3; sg = 1 if i % 2 == 0 else -1; A = .42*sg*np.array([1, .65])
            if mode == 'random': phi = rng.uniform(0, 2*np.pi); A = A*rng.uniform(.65, 1.2)
            P.append(np.stack([A[0]*np.sin(W*t+phi), A[1]*np.cos(.73*W*t+1.7*phi)], 1))
    return np.array(P)  # (M, n, 2)
def run(mode, Wc=4., tc=1., kap=2.45):
    P = traj(mode); M, n, _ = P.shape; kc = int(round(tc/dt)); kw = int(round(Wc/dt))
    t0 = np.arange(kw+kc+1, n-kt.max())  # моменты, где есть окно калибровки и будущее
    p = P[:, t0]; w = (P[:, t0]-P[:, t0-1])/dt
    # онлайн q: остатки vel-прогноза на лаге kc, взятые в прошлом окне [t-kw-kc, t-kc]
    q = np.zeros((M, len(t0), 2))
    for j, tt in enumerate(t0):
        s = np.arange(tt-kw-kc, tt-kc+1); ws = (P[:, s]-P[:, s-1])/dt
        r = P[:, s+kc]-P[:, s]-ws*tc
        q[:, j] = 3*np.mean(r**2, 1)/tc**3
    out = []
    for tau, k in zip(taus, kt):
        f = P[:, t0+k]
        e_s = np.sqrt(np.mean(np.sum((f-p)**2, -1))); ev = f-p-w*tau; e_v = np.sqrt(np.mean(np.sum(ev**2, -1)))
        sig = np.sqrt(q*tau**3/3)  # по осям
        z = np.sqrt(np.sum((ev/np.maximum(sig, 1e-9))**2, -1))   # анизотропное облако
        sig_iso = np.sqrt(q.max(-1)*tau**3/3); ziso = np.linalg.norm(ev, axis=-1)/np.maximum(sig_iso, 1e-9)
        out.append((tau, e_s, e_v, np.mean(z <= kap), np.mean(ziso <= kap), np.percentile(np.linalg.norm(ev, axis=-1), 95), np.median(sig_iso)*kap))
    return out
for mode in ['fixed', 'random']:
    for tc in [.5, 1., 2.]:
        print(f'\n{mode}, калибровка на лаге tc={tc} с, окно 4 с, kappa 2.45 (цель покрытия 95%)')
        print('tau | СКО снимок | СКО по скор. | покрытие аниз. | покрытие изотр. | p95 ошибки vel | мед. kappa*sigma изотр.')
        for r in run(mode, tc=tc): print(' | '.join(f'{x:.3f}' for x in r))
