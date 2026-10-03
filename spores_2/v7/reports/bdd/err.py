"""Дифдрайв-бабочки v7 при ошибке модели (PLAN «Открыто» п.2, hub-v5chain-worker-15): реальный привод v·GV, ω·GW (план — по номиналу).
Агент без привязки к центру споры (в run() v7 позиция «защёлкивается» на центр — идеализация): после каждого сегмента пересчёт из РЕАЛЬНОГО состояния.
Цель — коробка |xy| ≤ TOL, |θ| ≤ THT (точный ноль при ошибке недостижим). Запуск из spores_2/v7: RW=.9 GV=.95 GW=.95 python3 reports/bdd/err.py N."""
import sys, os, time, json
sys.path.insert(0, '.'); here = os.getcwd()
from src.cells7 import butterfly_dd as BD          # chdir в research внутри
import numpy as np
M, wrap, BIG, move, tgt, tgtgt = BD.M, BD.wrap, BD.BIG, BD.move, BD.tgt, BD.tgtgt
GV, GW = float(os.environ.get('GV', 1)), float(os.environ.get('GW', 1)); TOL, THT = float(os.environ.get('TOL', .03)), float(os.environ.get('THT', .05)); SNAP = int(os.environ.get('SNAP', 0))
def ingoal(y): return np.hypot(y[0], y[1]) <= TOL and abs(wrap(y[2])) <= THT
def run(B, q, smax=400):
    y = np.array(q, float); t = 0.; at = -1; segs = 0
    for _ in range(smax):
        if ingoal(y): return t, segs
        J, act = B.best(y, at)
        if act is None or J >= BIG / 2: return np.inf, segs
        kind, phi, L, k = act
        if kind == 'turn': y = np.array([y[0], y[1], y[2] + GW * phi]); t += abs(phi)
        else:
            y = move(y, GW * phi, GV * L); t += abs(L) + abs(phi)
            if SNAP: y[:2] = B.C[k]
        y[2] = wrap(y[2]); at = k if np.hypot(*(y[:2] - B.C[k])) < 1e-3 else -1; segs += 1
    return np.inf, segs
if __name__ == '__main__':
    N = int(sys.argv[1]); os.chdir(BD.RES); ref = np.load('dd_ref_60.npy'); rng = np.random.default_rng(1)
    Q = np.c_[rng.uniform(-2, 2, (60, 2)), rng.uniform(-np.pi, np.pi, 60)]; t0 = time.time(); B = BD.ButterflyDDObs(N=N, obs=[]).solve(); print('готово', B.K, round(time.time() - t0), flush=True)
    R = [run(B, q) for q in Q]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T); r = T[f] / ref[f] if f.any() else np.array([np.nan])
    print(json.dumps(dict(N=N, GV=GV, GW=GW, TOL=TOL, THT=THT, SNAP=SNAP, n=len(Q), reach=round(float(f.mean()), 3), mean=round(float(r.mean()), 3), med=round(float(np.median(r)), 3), p90=round(float(np.percentile(r, 90)), 3), max=round(float(r.max()), 3), segs_med=float(np.median(S[f])), sec=round(time.time() - t0))), flush=True)
