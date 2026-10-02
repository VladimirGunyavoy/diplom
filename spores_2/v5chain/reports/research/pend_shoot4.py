"""research hub-v5chain-research-9: стрельба из 4 дуг для стартов, где 3 дуг не хватило: τ1 по сетке .05 (оба знака), дальше pend_shoot_ref.shoot с
противоположным знаком (сетка .03). Проверка — честная симуляция внутри shoot (первая дуга — rk4 шагом .005)."""
import numpy as np, sys, time
sys.path.insert(0, '.')
from pend_shoot_ref import band, shoot, step, UM, ingoal
B = {s: band(s * UM) for s in (1., -1.)}; rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)
R = np.load('pend_shoot_ref.npy'); ref = np.load('pend_ref_T.npy'); idx = [int(a) for a in sys.argv[1:]] or np.flatnonzero(~np.isfinite(R[:, 0])).tolist(); t0 = time.time()
for i in idx:
    best = (np.inf,)
    for s in (1., -1.):
        x, w = np.array([Q[i, 0]]), np.array([Q[i, 1]]); t1 = 0.
        while t1 < 8. and abs(w[0]) <= 4 and t1 < best[0]:
            T, pl = shoot(np.array([x[0], w[0]]), B, dtau=.03, S1=min(9., best[0] - t1) if np.isfinite(best[0]) else 9., ntry=60, signs=(-s,))
            if t1 + T < best[0]: best = (t1 + T, s, t1) + tuple(pl[1:])
            for _ in range(10): x, w = step(x, w, s * UM, .005)
            t1 += .05
    print(i, np.round(Q[i], 3), '4 дуги:', np.round(best, 3), 'прежний эталон', round(float(ref[i]), 3), 'отношение', round(float(best[0] / ref[i]), 4), 'sec', round(time.time() - t0), flush=True)
    if best[0] < ref[i]: ref[i] = best[0]
np.save('pend_ref_T.npy', ref)
