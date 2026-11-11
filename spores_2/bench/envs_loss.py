"""b15 (2026-11-11): разбивка потери T/T_ref ≈ 1.09 кара на ИХ шаге (замкнутый контур envs2026.closed_loop, c1600) против эталона OCP r21 (ref_car_N80/car_q*.npy = [T, U(80×2)]).
T/T_ref = (L/L_ref)·(v̄_ref/v̄), L — длина пути в xy, v̄ = L/T. Метрики: мин. зазор до дисков, число смен знака a, доля шагов a<0, время до v ≥ 1.9, доля времени a<0.
Запуск (aida 2, из bench/): LOAD=~/spore_v5/car/c1600.pkl REF=~/spore_v5/ref_car_N80 NP=12 python envs_loss.py → data/env_loss.json"""
import os, sys, json, time
os.environ.setdefault('SYS', 'car'); os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np
import envs2026 as E
from envs2026 import their_step, plan, dclear, DT, GOAL, GR
K = .625
def rec_loop(G, A, x0, tmax=12.):
    y = np.array(x0, float); Y = [y.copy()]; U = []; last = None; pen = 1e9; n = 0
    while n * DT < tmax:
        if np.linalg.norm(y[:2] - GOAL) <= GR: break
        r = plan(G, A, y)
        if r is None:
            if last is None: break
            us, dts = last; cum = np.cumsum(dts) - DT; keep = cum > 1e-9
            if not keep.any(): break
            us = [u for u, kk in zip(us, keep) if kk]; dts = [float(d) for d, kk in zip(dts, keep) if kk]; dts[0] = float(cum[keep][0]); r = (us, dts, 0.)
        us, dts, _ = r; last = (us, dts); cum = np.cumsum(dts); k = min(int(np.searchsorted(cum, .5 * E.DT)), len(us) - 1); u = np.array(us[k], float)
        yn = their_step(y, u); n += 1
        for s in np.linspace(0, 1, 21)[1:]: pen = min(pen, dclear(y[:2] * (1 - s) + yn[:2] * s))
        y = yn; Y.append(y.copy()); U.append(u)
    return np.array(Y), np.array(U), pen, n * DT

def one(args):
    i, x = args; Y, U, pen, T = rec_loop(E._G, E._A, x); return dict(i=i, Y=Y.tolist(), U=U.tolist(), clr=float(pen), T=T, ok=bool(np.linalg.norm(Y[-1, :2] - GOAL) <= GR))

def ref_roll(x0, T, U, h=.002):
    n = int(np.ceil(T / h)); h = T / n; x = np.array(x0, float); N = len(U); Y = [x.copy()]; Ua = []; clr = 1e9
    f = lambda x, u: np.array([x[3] * np.cos(x[2]), x[3] * np.sin(x[2]), u[1], u[0] - K * x[3]])
    for k in range(n):
        u = U[min(int(k * h / (T / N)), N - 1)]; k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); Y.append(x.copy()); Ua.append(u); clr = min(clr, dclear(x[:2]))
    return np.array(Y), np.array(Ua), clr, h

def metrics(Y, U, dt, clr, T):
    d = np.diff(Y[:, :2], axis=0); L = float(np.hypot(d[:, 0], d[:, 1]).sum()); v = Y[:, 3]; i19 = np.where(v >= 1.9)[0]
    a = U[:, 0]; sg = np.sign(a); sw = int(np.sum(sg[1:] * sg[:-1] < 0))
    return dict(T=float(T), L=L, vbar=L / T, clr=float(clr), t19=float(i19[0] * dt) if len(i19) else None, fa_neg=float(np.mean(a < -1e-6)), switches=sw, a_m15=float(np.mean(a <= -1.499)), vmax=float(v.max()), w_sat=float(np.mean(np.abs(U[:, 1]) >= 1.799)))

if __name__ == '__main__':
    import pickle, multiprocessing as mp, growCar as G, common, inspect, __main__
    for nm, o in vars(G).items():
        if inspect.isclass(o): setattr(__main__, nm, o)
    E._G = G; d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; E._A = A
    e = common.make('car'); X0 = e.X0; Tref = e.Tref; REF = os.path.expanduser(os.environ['REF'])
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(int(os.environ.get('NP', 12))) as p: res = p.map(one, list(enumerate(X0)))
    out = []
    for r in sorted(res, key=lambda r: r['i']):
        i = r['i']; Y = np.array(r['Y']); U = np.array(r['U']); ma = metrics(Y, U, DT, r['clr'], r['T']) if len(U) else None
        a = np.load(os.path.join(REF, 'car_q%d.npy' % i)); Tr = float(a[0]); Ur = a[1:].reshape(-1, 2); Yr, Uar, cr, h = ref_roll(X0[i], Tr, Ur); mr = metrics(Yr, Uar, h, cr, Tr)
        out.append(dict(i=i, ok=r['ok'], Tref_car_ref20=float(Tref[i]), agent=ma, ref=mr, Ya=r['Y'], Ua=r['U'], Yr=Yr[::10].tolist(), Ur=Ur.tolist()))
    json.dump(out, open(os.path.join(common.DATA, os.environ.get('OUTJ', 'env_loss.json')), 'w'), default=float); print('wall', time.perf_counter() - t0)
