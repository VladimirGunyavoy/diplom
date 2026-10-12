"""C2/CA2 статик кар: коридор v6 (research-21: CARL=6, NB 30000, NF 7500, K 5, зазор CM .002 → 20/20, 1.0467). Запуск на aida из ~/spore_v5/r21/v6 (там src.atlas6, car.py): CARL=6 CM=.002 python3 ~/spore_v5/c1/spores_2/bench/spores_car.py → data/car_spores.json.
Управления (слой s, длительность d) проверяет единый проверяльщик common.verify (RK4 h=.01)."""
import sys, time, os, json
sys.path.insert(0, '.'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('CARL', '6'); os.environ.setdefault('CM', '.002')
import numpy as np, common
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
import car as C
NB, NF, K = 30000, 7500, 5; tau, rho = .4, .15
g = lambda x: np.array([C.GR ** 2 - ((x[0] - C.GOAL[0]) ** 2 + (x[1] - C.GOAL[1]) ** 2)])
miss = lambda X: np.maximum(np.linalg.norm(np.asarray(X)[..., :2] - C.GOAL, axis=-1) - C.GR, 0)
ing = lambda P: np.linalg.norm(np.asarray(P)[..., :2] - C.GOAL, axis=-1) < C.GR
clear = lambda P: C.clearance(P) - float(os.environ['CM'])
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    r, a = C.GR * .95 * np.sqrt(rng.uniform()), rng.uniform(-np.pi, np.pi); seeds.append(np.array([C.GOAL[0] + r * np.cos(a), C.GOAL[1] + r * np.sin(a), rng.uniform(-np.pi, np.pi), rng.uniform(-.5, 2.)]))
fl = lambda P, s, t: C.flow(P, s, t, dt_max=.05)
S = SysN(fl, len(C.US), (C.XL, C.XL, np.pi, 1.5), ing, seeds, per=(0, 0, 2 * np.pi, 0), ok=lambda p: C.clearance(p) >= 0, blocked=lambda P: C.clearance(P) < 0)
e = common.make('car'); t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); tb = time.perf_counter() - t0; print('обратное дерево: %.1f с, узлов %s' % (tb, getattr(back, 'N', '?')), flush=True)
res = []; Ts = []; tq = []
for i, x in enumerate(e.X0):
    t1 = time.perf_counter(); b = corridor_query(S, fl, x, back, tau, NF, rho, g, miss, K=K, clear=clear, tries=3, kn=8); dq = time.perf_counter() - t1; tq.append(dq)
    if b is None: Ts.append(None); res.append(dict(i=i, T_verify=None, why='no_plan', t_query=dq)); continue
    us = [list(C.US[s]) for s in b[1]]; dts = [float(d) for d in b[2]]; Tv, why = common.verify(e, x, us, dts); Ts.append(Tv); res.append(dict(i=i, T_agent=float(b[0]), T_verify=Tv, why=why, t_query=dq)); print(res[-1], flush=True)
s = common.summarize(e, Ts); s.update(env='car', method='spores v6 коридор (CARL 6, NB 30000, NF 7500, K 5, CM .002; research-21)', t_grow=tb, t_solve=0., t_build=tb, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)),
                                  hw='CPU 1 proc (numpy+scipy SLSQP)', per_start=res, Tref=e.Tref.tolist(), nodes=int(getattr(back, 'N', 0) or 0)); s['weight'] = common.deep_weight(back)
json.dump(s, open(os.path.join(common.DATA, 'car_spores.json'), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
