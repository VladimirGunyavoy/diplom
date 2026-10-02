"""п.3 PLAN (hub-worker-14): бабочки ДИ на ТЕХ ЖЕ 60 стартах, что spore_v (s10.py: rng(0).uniform(-.375,.375,(200,2))[:60]*4). Запуск из spores_2/v7."""
import sys, time; sys.path.insert(0, '.'); sys.argv = ['x']
from src.cells7.butterfly_di import *
from v7_faces_di import tstar_box
t0 = time.time(); B = ButterflyFast(N=1500, m=7).solve(); print('споров', B.K, 'пар', B.npairs, 'сек', round(time.time() - t0), flush=True)
Q = np.random.default_rng(0).uniform(-.375, .375, (200, 2))[:60] * 4; Ts = tstar_box(Q[:, 0], Q[:, 1], .1); ok = Ts > .05
Vq = B.Vq(Q); m = ok & (Vq < BIG / 2); print('V cover', round(m.sum() / ok.sum(), 3), 'V/T* med', np.median(Vq[m] / Ts[m]).round(4))
R = [B.rollout(q, 'dt') for q in Q[ok]]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T); r = T[f] / Ts[ok][f]
print('n', ok.sum(), 'дошли', f.mean().round(3), 'T/T* mean', r.mean().round(3), 'med', np.median(r).round(3), 'p90', np.percentile(r, 90).round(3), 'max', r.max().round(2), 'sw med', np.median(S), 'max', S.max(), 'сек', round(time.time() - t0))
