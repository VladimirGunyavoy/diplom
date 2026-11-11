"""w39: хвост — по стартам T/T_ref агента и V(x0)/T_ref (без lookahead, KV=неподвижная точка). Запуск как tvg.py (KVAUTO=1)"""
import sys, numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:5]
import tvg
from tvg import *
S = mkS(TS); back = build_back(S, TAU, NB, RB); ag = Agent(S, back); kv = tvg.KV
X = env.X0; dd, j = ag.tree.query(S.wrap(X) / S.scale, k=min(KN * 3, len(ag.tree.data))); j = j % ag.nP
V = (ag.W[j] + kv * dd).min(1); rows = []
for i in range(len(X)):
    if not env.keep[i]: continue
    T, why, _ = episode(ag, X[i]); rows.append((i, (T / env.Tref[i]) if T else np.inf, V[i] / env.Tref[i], dd[i, 0], why))
rows.sort(key=lambda r: -r[1]); print('KV', kv); print('i  T/Tref  V/Tref  d_near why')
for r in rows[:10]: print('%2d %6.3f %6.3f %6.3f %s' % r)
a = np.array([[r[1], r[2]] for r in rows]); print('мед V/Tref всех %.3f; у хвоста (топ-8) %.3f; у остальных %.3f' % (np.median(a[:, 1]), np.median(a[:8, 1]), np.median(a[8:, 1])))
