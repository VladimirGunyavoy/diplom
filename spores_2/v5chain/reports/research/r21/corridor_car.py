"""research-21: статик кар — коридор v6 (адаптивное дерево + SLSQP с зазором до дисков вдоль пути). Запуск из v6: python3 corridor_car.py NB NF K"""
import sys, time, os, json, glob; sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.corridor_nd import corridor_query
import car as C; from ref_car import starts
NB, NF, K = [int(a) for a in sys.argv[1:4]]; tau, rho = float(os.environ.get('TAU', .4)), float(os.environ.get('RHO', .15))
ing = lambda P: np.linalg.norm(np.asarray(P)[..., :2] - C.GOAL, axis=-1) < C.GR
g = lambda x: np.array([C.GR ** 2 - ((x[0] - C.GOAL[0]) ** 2 + (x[1] - C.GOAL[1]) ** 2)])
miss = lambda X: np.maximum(np.linalg.norm(np.asarray(X)[..., :2] - C.GOAL, axis=-1) - C.GR, 0)
clear = lambda P: C.clearance(P) - float(os.environ.get('CM', .005))   # запас зазора в SLSQP (проверка — в 16 точках дуги)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    r, a = C.GR * .95 * np.sqrt(rng.uniform()), rng.uniform(-np.pi, np.pi); seeds.append(np.array([C.GOAL[0] + r * np.cos(a), C.GOAL[1] + r * np.sin(a), rng.uniform(-np.pi, np.pi), rng.uniform(-.5, 2.)]))
fl = lambda P, s, t: C.flow(P, s, t, dt_max=float(os.environ.get('DT', .05)))
S = SysN(fl, len(C.US), (C.XL, C.XL, np.pi, 1.5), ing, seeds, per=(0, 0, 2 * np.pi, 0), ok=lambda p: C.clearance(p) >= 0, blocked=lambda P: C.clearance(P) < 0)
Q = starts(); ref = {}
for f_ in glob.glob(os.path.expanduser('~/spore_v5/r21/car_N*.jsonl')):
    for l in open(f_): d = json.loads(l); ref[d['q']] = min(ref.get(d['q'], 1e9), d['T_ref'] or 1e9)
t0 = time.time(); back = build_back(S, tau, NB, rho); tb = time.time() - t0; print('обратное дерево NB %d: %.1f с' % (NB, tb), flush=True); res = []
from tqdm import tqdm
for i, x in tqdm(list(enumerate(Q)), desc='запросы коридора', mininterval=float(os.environ.get('TQDM_MI', 10))):
    t0 = time.time(); b = corridor_query(S, fl, x, back, tau, NF, rho, g, miss, K=K, clear=clear, tries=3, kn=8)
    ok = False; dmin = None
    if b is not None:
        xe = np.array(x, float); P = []
        for s, d in zip(b[1], b[2]):
            for _ in range(40): xe = C.flow(xe, s, d / 40, dt_max=.005); P.append(xe)
        dmin = float(C.clearance(np.array(P)).min()); ok = bool(g(xe)[0] > -1e-4 and dmin > -1e-3)
    res.append(dict(q=i, T=None if b is None else round(float(b[0]), 3), ref=ref.get(i), r=None if not ok else round(float(b[0]) / ref[i], 4), ok=ok, dmin=dmin and round(dmin, 4), sec=round(time.time() - t0, 1)))
    print(res[-1], flush=True)
r = np.array([x['r'] for x in res if x['ok']]); s_ = [x['sec'] for x in res]
print(json.dumps(dict(NB=NB, NF=NF, t_back=round(tb, 1), ok=int(len(r)), n=len(res), mean=round(float(r.mean()), 4) if len(r) else None, med=round(float(np.median(r)), 4) if len(r) else None,
                      max=round(float(r.max()), 4) if len(r) else None, q_med=float(np.median(s_)), q_p90=float(np.percentile(s_, 90)))), flush=True)
