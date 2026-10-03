"""research hub-v5chain-research-9: бабочки дифдрайва + финиш стрельбой: агент бабочек, как только V ≤ VF — точный путь min(TGT, TGTGT) из
текущей позы (`dd_rhombus_ref.py`; это реальные пути поворот/прямая, время аддитивно). T = t_до + T_финиша. Сравнение с эталоном на тех же 60 стартах."""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
import butterfly_dd as M
from dd_rhombus_ref import tgt, tgtgt
def run(B, q, VF, smax=300):
    y = np.array(q, float); t = 0.; at = -1; segs = 0
    for _ in range(smax):
        if np.hypot(y[0], y[1]) < 1e-9 and abs(M.wrap(y[2])) < 1e-9: return t, segs
        J, act = B.best(y, at)
        if J <= VF:
            a, b = tgt(*y), tgtgt(*y, starts=20)
            if min(a, b) <= J + 1e-6: return t + min(a, b), segs + (3 if a <= b else 5)
        if act is None or J >= M.BIG / 2: return np.inf, segs
        kind, phi, L, k = act
        if kind == 'turn': y = np.array([y[0], y[1], y[2] + phi]); t += abs(phi)
        else: y = M.move(y, phi, L); y[:2] = B.C[k]; t += abs(L) + abs(phi)
        at = k; segs += 1
    return np.inf, segs
if __name__ == '__main__':
    N = int(sys.argv[1]); ref = np.load('dd_ref_60.npy'); rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-2, 2, (60, 2)), rng.uniform(-np.pi, np.pi, 60)]
    t0 = time.time(); B = M.ButterflyDD(N=N).solve(); print('готово', round(time.time() - t0), flush=True)
    for VF in (0., 1., 2., 3.):
        t1 = time.time(); R = [run(B, q, VF) for q in Q]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T); r = T[f] / ref[f]
        print(json.dumps(dict(N=N, VF=VF, reach=round(float(f.mean()), 3), T_over_ref=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3)),
                              segs_med=float(np.median(S[f])), sec=round(time.time() - t1)), ensure_ascii=False), flush=True)
