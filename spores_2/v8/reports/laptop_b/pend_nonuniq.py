"""b3: диагностика неединственности Q (Якоби vs вёдра): где |ΔQ|>.05, кто ближе к эталону, EVALMIN. env — pend_env.sh (+EVALMIN, MODE). Картинка: PNG=<путь>"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); PI = np.pi; rq = np.random.default_rng(0); Q0 = np.stack([rq.uniform(-PI, PI, 100), rq.uniform(-2, 2, 100)], 1)
ref = np.load('pend_ref_best_u0.3.npy'); tag = os.environ.get('TAG', '-'); atlas, tg = S.build_atlas(list(g.US), 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); res = {}
for nm, sb in (('якоби', 0), ('вёдра', 1)):
    S.SOLVEB = sb; t0 = time.perf_counter(); G._solve_q(3000, 1e-6, False); dt = time.perf_counter() - t0; Vs = G.Vlayers(Q0).min(1); res[nm] = (G.V.copy(), G.Vn.copy(), Vs, dt, G.passes)
(Qa, Va, Sa, ta, pa), (Qb, Vb, Sb, tb, pb) = res['якоби'], res['вёдра']; f = (Qa < 500) & (Qb < 500); d = np.abs(Qa - Qb); big = f & (d > .05); dv = np.abs(Va - Vb); bigv = (Va < 500) & (Vb < 500) & (dv > .05)
th = g.wrapy(G.P)[:, 0]; w = G.P[:, 1]; gd = np.hypot(th, w); lay = G.layer
# число ячеек-кандидатов на узел (нахлёст): строки пар по слою
ov = np.zeros(G.n)
for li, pr in enumerate(G.e3): pi, v_, ww, rk = pr; ov += np.bincount(pi, minlength=G.n) / 1.0
k = (ov > 4 * np.median(ov[ov > 0]) * 1.5)
print('RES[%s] Якоби %.1f с (%d пр.), вёдра %.1f с | узлов %d: |ΔQ|>.05: %d (%.1f%%), |ΔV|>.05: %d; >.2: %d; max %.2f' % (tag, ta, pa, tb, G.n, big.sum(), 100 * big.sum() / G.n, bigv.sum(), (f & (d > .2)).sum(), d[f].max()))
for nm, m in (('шов |θ|>2.8', np.abs(th) > 2.8), ('у цели |(θ,ω)|<.6', gd < .6), ('|ω|>3', np.abs(w) > 3), ('все с u=-.3', lay == 0), ('u=0', lay == 1), ('u=+.3', lay == 2)): print('RES   %-18s узлов %5d, из них |ΔQ|>.05: %5d (%.1f%% против %.1f%% в среднем)' % (nm, m.sum(), (big & m).sum(), 100 * (big & m).sum() / max(m.sum(), 1), 100 * big.sum() / G.n))
for nm, m in (('θ∈[-3.2,-2.2]', (th > -3.2) & (th < -2.2)), ('θ∈[-1,1]', np.abs(th) < 1), ('θ∈[2.2,3.2]', th > 2.2)): print('RES   %-18s |ΔQ|>.05: %5d' % (nm, (big & m).sum()))
ok = np.isfinite(ref); 
for nm, Sx in (('якоби', Sa), ('вёдра', Sb)): e = (Sx / ref - 1)[ok]; print('RES  ошибка V*(старт)/T_ref−1 [%s]: мед %+.3f, |мед| %.3f, p90|.| %.3f, max|.| %.2f; V*<T_ref в %.0f%%' % (nm, np.median(e), np.median(np.abs(e)), np.percentile(np.abs(e), 90), np.abs(e).max(), 100 * (e < -.005).mean()))
if os.environ.get('PNG'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5)); a = ax[0]; a.scatter(th[f], w[f], s=1, c='#cccccc'); a.scatter(th[big], w[big], s=2, c=d[big], cmap='plasma', vmin=.05, vmax=.5); a.set_title('|Q_вёдра − Q_Якоби| > .05: %d из %d узлов' % (big.sum(), G.n)); a.set_xlabel('θ'); a.set_ylabel('ω'); a.set_xlim(-PI, PI)
    a = ax[1]; sg = np.sign(Qb - Qa); a.scatter(th[big & (sg < 0)], w[big & (sg < 0)], s=2, c='#1b6ca8', label='вёдра ниже'); a.scatter(th[big & (sg > 0)], w[big & (sg > 0)], s=2, c='#d1495b', label='вёдра выше'); a.legend(markerscale=5); a.set_title('знак расхождения'); a.set_xlabel('θ'); a.set_xlim(-PI, PI)
    fig.tight_layout(); fig.savefig(os.environ['PNG'], dpi=110); print('RES картинка', os.environ['PNG'])
