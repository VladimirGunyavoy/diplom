"""Г4.2 рука (в): обратное дерево НА КАЖДУЮ цель + прямое дерево + SLSQP (коридор v6 SPD6, c13/spd6/spores_di.py со сдвигом цели). NB NF K из argv.
python g42_corr.py NB NF K tag ; те же цели/старты/эталон, что у руки (б) (g42_common)."""
import os, sys, time, json, itertools, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
BENCH = os.path.expanduser('~/spore_v5/a31/spores_2/bench'); sys.path.insert(0, BENCH)
import numpy as np, common
sys.path.insert(0, os.path.join(BENCH, '..', 'v6'))
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import g42_common
NB, NF, K = [int(a) for a in sys.argv[1:4]]; tag = sys.argv[4]; k = 2; dim = 4; tau = .4; rho = .15; R = .35; MG = .004; Rg = R - MG
e = common.make('di4'); US_ = np.array(list(itertools.product((-1., 1.), repeat=k))); hi = np.array(e.hi)
def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); t = np.asarray(t, float)[..., None]; a = US_[s]; return np.concatenate([P[..., :k] + P[..., k:] * t + a * t * t / 2, P[..., k:] + a * t], -1)
print('HEADER: рука (в) коридор v6 ДИ4D NB=%d NF=%d K=%d tau=%g rho=%g NEIGH=0 SLSQP-refine (не IPOPT), запас у цели %.3f; цели/старты g42_common seed 42; проверяльщик common.verify h .01 строгий; окно ±.35; коробка %s/%s; код md5=%s' % (NB, NF, K, tau, rho, MG, e.lo.tolist(), e.hi.tolist(), hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:10]), flush=True)
rows = []; allr = []; tb = []; tqs = []
for gi, (C, Q, ref, keep) in enumerate(g42_common.make()):
    ing = lambda P, C=C: np.max(np.abs(P - C), -1) < R; g = lambda x, C=C: np.array([Rg ** 2 - (xi - ci) ** 2 for xi, ci in zip(x, C)]); miss = lambda X, C=C: np.maximum(np.max(np.abs(X - C), -1) - Rg, 0)
    rng = np.random.default_rng(0); seeds = [C + v / np.max(np.abs(v)) * .99 * R for v in (rng.uniform(-1, 1, dim) for _ in range(32))]
    t0 = time.perf_counter(); S = SysN(flow, 2 ** k, hi, ing, seeds, per=None, ok=lambda p: bool(np.all(np.abs(p) <= hi))); back = build_back(S, tau, NB, rho); t_b = time.perf_counter() - t0; tb.append(t_b); res = []
    for i in range(len(Q)):
        if not keep[i]: continue
        t1 = time.perf_counter(); b = corridor_query(S, flow, Q[i], back, tau, NF, rho, g, miss, K=K, tries=3, kn=8, neigh=0, nbest=1); dt = time.perf_counter() - t1; tqs.append(dt)
        if b is None: res.append((None, 'no_path', ref[i])); continue
        Tv, why = common.verify(type('E', (), dict(in_goal=lambda s, y, C=C: (np.abs(y - C) < R).all(-1), clip_u=e.clip_u, step=e.step, out_of_box=e.out_of_box))(), Q[i], [US_[s] for s in b[1]], list(b[2])); res.append((Tv, why, ref[i]))
    ok = [(t / r) for t, w, r in res if t is not None and t <= 3 * r]; allr.append(res)
    rows.append(dict(goal=C[:2].tolist(), n=len(res), reach=len(ok), t_build=t_b, med=float(np.median(ok)) if ok else None, p90=float(np.percentile(ok, 90)) if ok else None, max=float(max(ok)) if ok else None, why={w: sum(1 for x in res if x[1] == w) for w in set(x[1] for x in res)}))
    print('цель %d %s: подготовка на цель %.2f с: %s' % (gi, np.round(C[:2], 2).tolist(), t_b, rows[-1]), flush=True)
R_ = np.array([t / r for res in allr for t, w, r in res if t is not None and t <= 3 * r]); nk = sum(r['n'] for r in rows); tq = np.array(tqs)
s = dict(env='di4', NB=NB, NF=NF, K=K, n=nk, reach=int(len(R_)), med=float(np.median(R_)), p90=float(np.percentile(R_, 90)), max=float(R_.max()), t_prep_goal_med=float(np.median(tb)), t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), weight=common.deep_weight(back), rows=rows)
print('ИТОГ (в): дошли %d/%d, T/T_ref мед %.3f p90 %.3f max %.3f; подготовка на цель мед %.2f с; запрос мед %.3f / p90 %.3f с; вес на цель %s' % (s['reach'], nk, s['med'], s['p90'], s['max'], s['t_prep_goal_med'], s['t_query_med'], s['t_query_p90'], s['weight']), flush=True)
o = os.path.expanduser('~/spore_v5/c13/out_g42'); os.makedirs(o, exist_ok=True); json.dump(s, open('%s/arm_v_%s.json' % (o, tag), 'w'), default=float)
