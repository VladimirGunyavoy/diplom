"""b3: разбор качества pend: по каждому старту V*(старт) vs T_ref vs T агента; 10 худших + 6 медианных. env как bench_pend.py"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); PI = np.pi; rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-PI, PI, 100), rq.uniform(-2, 2, 100)], 1)
ref = np.load(os.path.join(os.environ.get('PEND_REF', '.'), 'pend_ref_best_u0.3.npy')); tag = os.environ.get('TAG', '-')
atlas, tg = S.build_atlas(list(g.US), 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
pol = os.environ.get('POL', 'commit'); T, sw = G.run_agent(Q, 3 * np.where(np.isfinite(ref), ref, 5) + 2, **S.POLICIES[pol])
Vs = G.Vlayers(Q).min(1); r = T / ref; ok = np.isfinite(T) & np.isfinite(ref)
E = Q[:, 1] ** 2 / 2 + np.cos(Q[:, 0])
print('RES[%s] %s дошли %.2f мед %.3f p90 %.3f max %.2f | V*/T_ref мед %.3f p10 %.3f p90 %.3f | T/V* мед %.3f p90 %.3f max %.2f' % (tag, pol, ok.mean(), np.nanmedian(r), np.nanpercentile(r, 90), np.nanmax(r), np.nanmedian(Vs / ref), *np.nanpercentile(Vs / ref, [10, 90]), np.nanmedian(T / Vs), np.nanpercentile(T / Vs, 90), np.nanmax(T / Vs)))
o = np.argsort(-np.where(ok, r, 0)); med = np.argsort(np.abs(np.where(ok, r, 9) - np.nanmedian(r)))[:6]
print('RES  i   th      w     E-1   T_ref   V*     T    T/Tref V*/Tref T/V*  sw')
for grp, idx in (('худшие', o[:10]), ('медиана', med)):
    print('RES --', grp)
    for i in idx: print('RES %3d %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %3d' % (i, *Q[i], E[i] - 1, ref[i], Vs[i], T[i], r[i], Vs[i] / ref[i], T[i] / Vs[i], sw[i]))
# где теряется: доля потерь T-T_ref, объяснённая V* (граф) vs исполнением (T-V*)
d1 = (Vs - ref)[ok]; d2 = (T - Vs)[ok]; print('RES[%s] сумма потерь: граф (V*-T_ref) %.2f, исполнение (T-V*) %.2f; V*<T_ref в %.0f%% стартов, V*>1.03 T_ref в %.0f%%' % (tag, d1.sum(), d2.sum(), 100 * (Vs[ok] < ref[ok] * .995).mean(), 100 * (Vs[ok] > ref[ok] * 1.03).mean()))
