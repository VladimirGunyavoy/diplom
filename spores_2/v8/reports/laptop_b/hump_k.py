"""laptop-B: горб V* перед переключением и подточки (SUBK) в solve8. Из v8 (aida; нужны hump.py, qsolve*.py из v5chain/reports/research в src/algo):
MAXC=3 [GOALFRONT=1] python3 reports/laptop_b/hump_k.py   (env KS='0,1,3', N=600, OUT). Для каждого K: горб (мед/p90/max, доля V* растёт), доля узлов <−.05 (вне круга), commit/look в зоне."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
import hump as H
OUT = os.environ.get('OUT', 'hump_k'); TG = H.TG
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.] if not os.environ.get('NO0') else [1., -1.], 0, par=False); ins = 0
if os.environ.get('REFINE'): atlas, G, ins = S.refine_atlas(g, atlas, float(os.environ['REFINE']), int(os.environ.get('RDEPTH', 2)))
else: G = S.make_graph(g, atlas, 'N')
rho = G.rho; G.set_calc('q')
print('RES MAXC=%s GOALFRONT=%s REFINE=%s клеток %s (вставлено полос %d) узлов %d' % (os.environ.get('MAXC', '-'), os.environ.get('GOALFRONT', '0'), os.environ.get('REFINE', '-'), {u: len(atlas[u]) for u in atlas}, ins, G.n))
X0 = S.starts(g, int(os.environ.get('N', '600')), 1); mx = S.path_extent(X0, rho); inw = mx <= float(g.XLV[0]); Xz = X0[inw]; U0, T1, TT = H.opt_switch(Xz, rho)
ids = np.flatnonzero(np.isfinite(T1) & (T1 >= .8) & (TT - T1 >= .3)); PTS = np.zeros((len(ids), len(TG), 2)); TRF = np.zeros((len(ids), len(TG)))
for r, i in enumerate(ids):
    t = T1[i] + TG; a = np.minimum(t, T1[i]); b = np.maximum(t - T1[i], 0.); u = U0[i]; x1 = Xz[i, 0] + Xz[i, 1] * a + u * a * a / 2; v1 = Xz[i, 1] + u * a; PTS[r, :, 0] = x1 + v1 * b - u * b * b / 2; PTS[r, :, 1] = v1 - u * b; TRF[r] = TT[i] - t
Tr = R.T_ref(X0, rho); tmax = 3 * Tr + 2; prof = {}
for K in [int(k) for k in os.environ.get('KS', '0,1,3').split(',')]:
    G.subk = K; t0 = time.perf_counter(); G.solve(verbose=False); ts = time.perf_counter() - t0
    E = np.full((len(ids), len(TG)), np.nan)
    for r in range(len(ids)): E[r] = G.Vlayers(PTS[r]).min(1) - TRF[r]
    E[~np.isfinite(E)] = np.nan; win = (TG >= -.5) & (TG <= 0); full = np.isfinite(E[:, 0]) & np.isfinite(E[:, win]).all(1); Ef = E[full]
    if len(Ef): hump = Ef[:, win].max(1) - Ef[:, 0]; dV = np.diff(Ef + (-TG)[None, :], axis=1); up = (dV[:, (TG[1:] >= -.5) & (TG[1:] <= 0)] > 0).any(1)
    out = np.linalg.norm(G.P, axis=1) > rho; Tn = R.T_ref(G.P, rho); fin = G.Vn < S.BIG / 2; e = (G.Vn - Tn)[out & fin]
    line = 'RES K=%d solve %.1f с проходов %d | трасс %d из %d: ' % (K, ts, G.passes, len(Ef), len(ids))
    if len(Ef): line += 'горб мед %.3f p90 %.3f max %.3f, V* растёт у %.0f%% | ' % (np.median(hump), np.percentile(hump, 90), hump.max(), 100 * up.mean()); prof[K] = np.nanmedian(Ef, 0)
    line += 'узлы V<T_ref−.05: %.1f%% (мед ош %.3f)' % (100 * (e < -.05).mean(), np.median(e)); print(line, flush=True)
    for name, pol in (('commit', 'commit'), ('look', 'look')):
        T, sw = G.run_agent(X0, tmax, **S.POLICIES[pol]); ok = np.isfinite(T); r = T[ok & inw] / Tr[ok & inw]
        print('RES   K=%d %-6s в зоне дошли %.3f | T/T_ref мед %.3f p90 %.3f max %.2f' % (K, name, ok[inw].mean(), np.median(r), np.percentile(r, 90), r.max()), flush=True)
np.save(OUT + '_prof.npy', np.array([TG] + [prof[k] for k in sorted(prof)]))
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for K, p in prof.items(): plt.plot(TG, p, label='K=%d' % K)
    plt.axvline(0, color='k', lw=.6); plt.axhline(0, color='k', lw=.6); plt.xlabel('t − t_sw, с'); plt.ylabel('V* − T_ref (мед)'); plt.legend(); plt.grid(alpha=.3); plt.title('профиль вдоль оптимальной трассы, MAXC=%s' % os.environ.get('MAXC', 'полный')); plt.tight_layout(); plt.savefig(OUT + '.png', dpi=100)
except Exception as ex: print('нет картинки', ex)
