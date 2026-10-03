"""Гибрид: бабочки v7 (ButterflyPend, rollout) + финиш стрельбой ≤3 дуг, когда V бабочек ≤ VF (PLAN п.1, hub-worker-14).
Эталон — pend_ref_T.npy (v5chain/reports/research), те же 100 стартов. Запуск из spores_2/v7."""
import sys, os, time, json
sys.path.insert(0, '.'); sys.path.insert(0, '../v5chain/reports/research')
import numpy as np
from src.cells7.butterfly_pend import ButterflyPend, ingoal, wrap, rk4, UM, BIG
from pend_shoot_ref import band, shoot

GAIN = float(os.environ.get('GAIN', 1.))
def run(B, Bd, Q, VF, dt=.06, tmax=40., sub=4):
    T = np.full(len(Q), np.inf); SW = np.zeros(len(Q), int); tf = np.zeros(len(Q))
    for i, q in enumerate(Q):
        y = np.array(q, float); y[0] = wrap(y[0]); t = 0.; pu = None; sw = 0
        while t < tmax:
            if ingoal(y): T[i] = t; break
            J, b = B.best(y)
            if b is None or J >= BIG / 2: break
            if J <= VF:
                Ts, pl = shoot(y.copy(), Bd, S1=VF + 1.5, S2=VF + 1.5, ntry=100)
                if np.isfinite(Ts) and Ts <= J + .3:
                    if pl is None: T[i] = t; break                   # уже в коробке ±.1 — shoot вернул (0, None)
                    arcs = [(pl[0], pl[1]), (-pl[0], pl[2]), (pl[0], pl[3])]; us = [a for a, d in arcs if d > 1e-9]; z = y[None, :].copy(); te = 0.
                    hit = False
                    for a, d in arcs:                                  # открытое исполнение плана с ошибкой модели; цель плана — коробка ±.1 (как в стрельбе)
                        n = max(1, int(np.ceil(d / .005)))
                        for _ in range(n):
                            z = rk4(z, np.array([a * UM * GAIN]), np.array([d / n])); te += d / n
                            if abs(wrap(z[0, 0])) <= .1 and abs(z[0, 1]) <= .1: hit = True; break
                        if hit: break
                    if hit or GAIN == 1.:
                        sw += sum(1 for a, c in zip(([np.sign(pu)] if pu is not None else []) + us[:-1], us) if a != c); T[i] = t + (te if GAIN != 1. else Ts); tf[i] = Ts; break
                    # план промахнулся: продолжаем агентом (замкнутая обратная связь)
            _, _, ta, u = b; h = min(dt, ta)
            if pu is not None and abs(u - pu) > .5 * UM: sw += 1
            pu = u; z = y[None, :]
            for _ in range(sub): z = rk4(z, np.array([u * GAIN]), np.array([h / sub])); t += h / sub
            y = z[0].copy(); y[0] = wrap(y[0])
        SW[i] = sw
    return T, SW

if __name__ == '__main__':
    t0 = time.time(); FILL = bool(os.environ.get('FILL'))                                     # FILL=1: V с заполненными дырами (bp_fill.py)
    B = ButterflyPend(N=5000, m=7, tau=2.0, win=.3, extra=np.load('/tmp/claude-1000/bp_extra.npy') if FILL else None); B.V = np.load('/tmp/claude-1000/bp_V5.npy' if FILL else '/tmp/claude-1000/bp_V4.npy')
    Bd = {s: band(s * UM) for s in (1., -1.)}; print('готово', round(time.time() - t0), flush=True)
    rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 100), rng.uniform(-2, 2, 100)], 1)
    ref = np.load('../v5chain/reports/research/pend_ref_T.npy'); ok0 = ref > .05
    for VF in [float(a) for a in sys.argv[1:]] or [0., 1.5, 3.]:
        t1 = time.time(); T, SW = run(B, Bd, Q, VF); f = ok0 & np.isfinite(T) & np.isfinite(ref); r = T[f] / ref[f]; near = f & (ref < 2)
        print(json.dumps(dict(VF=VF, reach=round(float(f.sum() / ok0.sum()), 3), mean=round(float(r.mean()), 3), med=round(float(np.median(r)), 3), p90=round(float(np.percentile(r, 90)), 3), max=round(float(r.max()), 3),
              sw=round(float(SW[f].mean()), 2), near_n=int(near.sum()), near_dT=round(float(np.mean((T - ref)[near])), 3) if near.any() else None, sec=round(time.time() - t1)), ensure_ascii=False), flush=True)
