"""b26: разбор первого шага з.19 ст.58 TIGHT: V̂ по 9 действиям, зазоры, число дисков (6 в TIGHT, код считает 3)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import car_dyn_astar as da
cd = da.cd; np = da.np
from car_dyn6 import ACT, TAU, HR, cand, HB, FM
import envs_dyn as D
K = cd.K0; I = int(os.environ.get('DIAGI', 58)); S0 = np.load(os.environ['STARTS'])[:, :4]
S = cd.mkS(K); back = cd.build_back(S, TAU, cd.NB, cd.RB); ag = da.AgentDynA(S, back); H = D.Hazards('random', K); x = np.array(S0[I], float)
P0 = H.pos(0.); Wv = np.zeros((6, 2)); print('диски в среде:', P0.shape, ' в коде агента (cand/plan): 3 первых'); ag.plan(x, P0, Wv)
print('nflag', ag.nflag, 'W0x %.2f Wx(после плана) ' % ag.W0x)
for s in range(9):
    xe, ye, the, ve, mn, hit = cand(x[0], x[1], x[2], x[3], ACT[s, 0], ACT[s, 1], P0, Wv, cd.KAPC, cd.QC, cd.SMAX, TAU, HB)
    q = np.array([[xe, ye, the, ve]]); print('a%d u=%s: конец %s, зазор(3 диска) %.2f, V̂(конец) %.2f' % (s, ACT[s].tolist(), np.round(q[0], 2).tolist(), mn, float(ag.hh(q)[0])))
# зазор до ВСЕХ 6 дисков вдоль пути каждого действия
from car_dyn6 import rk
for s in range(9):
    xx = x.copy(); m6 = 9.; m3 = 9.
    for k in range(10):
        xx = np.array(rk(xx[0], xx[1], xx[2], xx[3], ACT[s, 0], ACT[s, 1], TAU / 10, 1)); d = np.hypot(*(P0 - xx[:2]).T) - HR; m6 = min(m6, d.min()); m3 = min(m3, d[:3].min())
    print('a%d: мин зазор 3 диска %.2f, 6 дисков %.2f' % (s, m3, m6))
print('центры дисков', np.round(P0, 2).tolist(), 'старт', np.round(x, 2).tolist())
