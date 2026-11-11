"""b14: причина провала открытого плана на их шаге — расхождение их Эйлера с нашей непрерывной моделью: для плана агента (атлас c1600) положение через t = n·.08 в (а) нашей модели (RK4 h .01, v насыщается) и (б) их шаге (u по середине шага). Печатает ошибку положения (м) в конце плана: мед / p90 / max, и ошибку по шагам."""
import os, sys, pickle, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import envs2026 as E, growCar as G, common, __main__, inspect
for nm, o in vars(G).items():
    if inspect.isclass(o): setattr(__main__, nm, o)
d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
e = common.make('car'); K = .625
def cont_pos(x0, us, dts, tend, h=.01):
    def fl(y, a, w):
        dv = a - K * y[3]
        if (y[3] >= 2. and dv > 0) or (y[3] <= -1. and dv < 0): dv = 0.
        return np.array([y[3] * np.cos(y[2]), y[3] * np.sin(y[2]), w, dv])
    y = np.array(x0, float); t = 0.; cum = np.cumsum(dts)
    while t < tend - 1e-9:
        k = min(int(np.searchsorted(cum, t + 1e-9)), len(us) - 1); a, w = us[k]; hh = min(h, tend - t, cum[k] - t if cum[k] - t > 1e-9 else h)
        k1 = fl(y, a, w); k2 = fl(y + hh / 2 * k1, a, w); k3 = fl(y + hh / 2 * k2, a, w); k4 = fl(y + hh * k3, a, w); y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += hh
    return y
errs = []; e_end = []
for i, x in enumerate(e.X0):
    r = E.plan(G, A, x)
    if r is None: continue
    us, dts, Ta = r; cum = np.cumsum(dts); n = int(np.floor(cum[-1] / E.DT)); y = np.array(x, float); ei = []
    for j in range(n):
        k = min(int(np.searchsorted(cum, (j + .5) * E.DT)), len(us) - 1); y = E.their_step(y, us[k])
        if (j + 1) % 10 == 0 or j == n - 1: ei.append(float(np.linalg.norm(y[:2] - cont_pos(x, us, dts, (j + 1) * E.DT)[:2])))
    e_end.append(ei[-1]); errs.append(ei)
e_end = np.array(e_end); print('ошибка конца плана на их шаге, м: мед %.3f p90 %.3f max %.3f (n=%d); цель r=.45' % (np.median(e_end), np.percentile(e_end, 90), e_end.max(), len(e_end)))
print('по времени (каждые 10 шагов), среднее по стартам:', [round(float(np.mean([r[k] for r in errs if len(r) > k])), 3) for k in range(max(len(r) for r in errs))])
