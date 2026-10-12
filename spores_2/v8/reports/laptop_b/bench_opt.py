"""laptop-B b2: бенч скорости SOLVE8_PLUS (3 споры) — время build/refine/solve(_solve_q, bisect)/агент commit на 200 стартов + качество (горб, занижение узлов, T/T_ref мед/p90, дошли).
MAXC=3 SOLVE8_PLUS=1 [WARM=1 TOL=1e-6 COMMIT_EVERY=3] python3 reports/laptop_b/bench_opt.py   (нужны hump.py и qsolve*.py в src/algo, как у hump_k.py). TAG=имя для строки RES."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
import hump as H
TG = H.TG; tag = os.environ.get('TAG', '-'); TM = {}
def timed(cls, name):
    f = getattr(cls, name)
    def w(*a, **k):
        t0 = time.perf_counter(); r = f(*a, **k); TM[name] = TM.get(name, 0) + time.perf_counter() - t0; TM[name + '#'] = TM.get(name + '#', 0) + 1; return r
    setattr(cls, name, w)
t0 = time.perf_counter(); g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); tb = time.perf_counter() - t0
for nm in ('_solve_q', '_bisect_pass', 'locate'): timed(type(G), nm)
rho = G.rho; t0 = time.perf_counter(); G.solve(verbose=False); ts = time.perf_counter() - t0
X0 = S.starts(g, 600, 1); mx = S.path_extent(X0, rho); inw = mx <= float(g.XLV[0]); Xz = X0[inw]; U0, T1, TT = H.opt_switch(Xz, rho)
ids = np.flatnonzero(np.isfinite(T1) & (T1 >= .8) & (TT - T1 >= .3)); PTS = np.zeros((len(ids), len(TG), 2)); TRF = np.zeros((len(ids), len(TG)))
for r, i in enumerate(ids):
    t = T1[i] + TG; a = np.minimum(t, T1[i]); b = np.maximum(t - T1[i], 0.); u = U0[i]; x1 = Xz[i, 0] + Xz[i, 1] * a + u * a * a / 2; v1 = Xz[i, 1] + u * a; PTS[r, :, 0] = x1 + v1 * b - u * b * b / 2; PTS[r, :, 1] = v1 - u * b; TRF[r] = TT[i] - t
E = np.full((len(ids), len(TG)), np.nan)
for r in range(len(ids)): E[r] = G.Vlayers(PTS[r]).min(1) - TRF[r]
E[~np.isfinite(E)] = np.nan; win = (TG >= -.5) & (TG <= 0); full = np.isfinite(E[:, 0]) & np.isfinite(E[:, win]).all(1); Ef = E[full]
hump = Ef[:, win].max(1) - Ef[:, 0]; out = np.linalg.norm(G.P, axis=1) > rho; Tn = R.T_ref(G.P, rho); fin = G.Vn < S.BIG / 2; e = (G.Vn - Tn)[out & fin]
print('load', os.getloadavg()[0], flush=True); print('RES[%s] build %.2f solve %.2f (_solve_q %.2f x%d, bisect_pass %.2f x%d, locate %.2f x%d) узлов %d проходов(посл) %d раундов %d подточек %s (границ рёбер %d→%d) | горб мед %.3f p90 %.3f | занижение<-.05: %.1f%% | Vsum %.4f' % (tag, tb, ts, TM['_solve_q'], TM['_solve_q#'], TM.get('_bisect_pass', 0), TM.get('_bisect_pass#', 0), TM['locate'], TM['locate#'], G.n, G.passes, getattr(G, 'nrounds', 0), str(getattr(G, 'npts', [])), G.bstat[0], G.bstat[1], np.median(hump), np.percentile(hump, 90), 100 * (e < -.05).mean(), np.minimum(G.Vn, 1e3).sum()), flush=True)
X2 = S.starts(g, 200, 1); Tr = R.T_ref(X2, rho); TM.clear()
for pol in ('commit', 'look'):
    t0 = time.perf_counter(); T, sw = G.run_agent(X2, 3 * Tr + 2, **S.POLICIES[pol]); ta = time.perf_counter() - t0; ok = np.isfinite(T); r = T[ok] / Tr[ok]
    print('RES[%s]   %-6s агент 200 стартов %.2f с (locate %.2f с x%d) | дошли %.3f T/T_ref мед %.4f p90 %.4f max %.2f' % (tag, pol, ta, TM.get('locate', 0), TM.get('locate#', 0), ok.mean(), np.median(r), np.percentile(r, 90), r.max()), flush=True); TM.clear()
