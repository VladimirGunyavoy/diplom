"""research hub-v5chain-research-9: бабочки на маятнике + финиш стрельбой. Агент бабочек («dt») идёт, пока V бабочек > VF; как только V ≤ VF —
стрельба ≤ 3 дуг из текущей точки (pend_shoot_ref.shoot, честная симуляция) и исполнение открыто (модель точная). T = t_до + T_стрельбы.
Сравнение с эталоном pend_ref_T.npy на 100 стартах."""
import numpy as np, sys, time, json, os
sys.path.insert(0, '.')
from butterfly_pend import ButterflyPend, BIG, RHO, stats
from pend_shoot_ref import band, shoot
from pend_cost_grid import step, wrap, UM
def run(B, Bd, Q, VF, dt=.05, h=.01, tmax=40.):
    T = np.full(len(Q), np.inf); SW = np.zeros(len(Q), int)
    for i, q in enumerate(Q):
        y = np.array(q, float); t = 0.; pu = None; sw = 0
        while t < tmax:
            if abs(wrap(y[0])) <= RHO + 1e-9 and abs(y[1]) <= RHO + 1e-9: T[i] = t; break
            J, ta, u = B.best(y[None]); J, ta, u = J[0], ta[0], u[0]
            if J <= VF:
                Ts, pl = shoot(np.array([wrap(y[0]), y[1]]), Bd, S1=VF + 1.5, S2=VF + 1.5, ntry=100)
                if np.isfinite(Ts) and Ts <= J + .3:
                    arcs = [(pl[0], pl[1]), (-pl[0], pl[2]), (pl[0], pl[3])]; us = [a for a, d in arcs if d > 1e-9]
                    sw += sum(1 for a, b in zip(([np.sign(pu)] if pu is not None else []) + us[:-1], us) if a != b); T[i] = t + Ts; break
            if J >= BIG / 2: break
            if pu is not None and abs(u - pu) > UM: sw += 1
            pu = u; hold = min(dt, ta)
            while hold > 1e-12:
                hh = min(h, hold); x, w = step(np.array([y[0]]), np.array([y[1]]), u, hh); y = np.array([x[0], w[0]]); t += hh; hold -= hh
                if abs(wrap(y[0])) <= RHO + 1e-9 and abs(y[1]) <= RHO + 1e-9: break
        SW[i] = sw
    return T, SW
if __name__ == '__main__':
    N = int(sys.argv[1]); t0 = time.time(); B = ButterflyPend(N=N, tau=.3, r=.1).build().solve(); Bd = {s: band(s * UM) for s in (1., -1.)}; print('готово', round(time.time() - t0), flush=True)
    rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1); ref = np.load('pend_ref_T.npy'); far = ref > .05; out = {}
    for VF in (0., 1.5, 3., 5.):
        t1 = time.time(); T, SW = run(B, Bd, Q, VF); f = far & np.isfinite(T); out[VF] = T
        print(json.dumps(dict(N=N, VF=VF, reach=round(float(f.sum() / far.sum()), 3), T_over_ref=stats(T[f] / ref[f]), sw_mean=round(float(SW[f].mean()), 2), sec=round(time.time() - t1)), ensure_ascii=False), flush=True)
    np.save('butterfly_pend_finish_%d.npy' % N, np.array([out[k] for k in out]))
