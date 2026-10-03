"""Разбор выбросов агента бабочек 4D (hub-worker-14): худшие старты T/T* при V/T* в старте. Нужен кэш пар /tmp/claude-1000/b4pairs_50081_7.npz. Из spores_2/v7."""
import sys, time; sys.path.insert(0, '.'); sys.argv = ['x']
from src.cells7.butterfly_4d import *
S = B4Fast(N=50000).solve(); rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-1, 1, (300, 2)), rng.uniform(-.7, .7, (300, 2))]
Ts = np.maximum(tstar_box(Q[:, 0], Q[:, 2], RHO), tstar_box(Q[:, 1], Q[:, 3], RHO)); ok = Ts > .05; Q, Ts = Q[ok], Ts[ok]; Q, Ts = Q[:150], Ts[:150]
V, _, _ = S.best(Q); T, sw = S.rollout(Q); r = T / Ts; o = np.argsort(-np.where(np.isfinite(r), r, 0))[:8]
for i in o: print('старт', i, Q[i].round(2), 'T*', round(Ts[i], 2), 'V/T*', round(V[i] / Ts[i], 3), 'T/T*', round(r[i], 3), 'перекл.', int(sw[i]))
print('V/T* у выбросов (T/T*>1.5): ', np.round(V[r > 1.5] / Ts[r > 1.5], 2), ' n выбросов', int((r > 1.5).sum()), ' T* у них', np.round(Ts[r > 1.5], 2))
