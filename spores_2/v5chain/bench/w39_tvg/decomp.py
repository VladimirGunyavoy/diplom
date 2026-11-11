"""w39: разложение для pend — V(x0) без lookahead (min_k W_b+KV|x0-b|) против T_ref на тех же стартах. Запуск как tvg.py: ENV NB K TS"""
import sys, numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:5]
import tvg
from tvg import *
S = mkS(TS); back = build_back(S, TAU, NB, RB); ag = Agent(S, back)
X = env.X0[:20]; k = env.keep[:20]; dd, j = ag.tree.query(S.wrap(X) / S.scale, k=min(KN * 3, len(ag.tree.data))); j = j % ag.nP
for kv in (0., .05, .5, 2., 8.):
    V = (ag.W[j] + kv * dd).min(1); r = V[k] / env.Tref[:20][k]
    print('KV=%g V(x0)/T_ref мед %.3f p90 %.3f max %.3f min %.3f; ближ. узел d мед %.3f' % (kv, np.median(r), np.percentile(r, 90), r.max(), r.min(), np.median(dd[:, 0])))
print('окно цели бенча half', env.half, 'seeds ±', env.half - .02, 'per', env.period)
