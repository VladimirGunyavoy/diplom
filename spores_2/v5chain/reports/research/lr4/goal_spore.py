"""lr4: две споры, идущие от цели (слова пользователя: «в них не может быть ошибки, там вообще нет решений, всё линейно»).
Точки на траекториях своего u, которые САМИ входят в круг: y = поток u назад на время s от точки (x_c, 0) внутри круга, x_c ∈ (−ρ, ρ), s ∈ [0, 1.6].
Точное время входа в круг своим u (без решений) t_in — аналитика; сравнение с запросом Q_u(y) (Q lin). Ошибка по x_c/ρ (центр → край полосы) и по s. env M. Запуск из v8."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS, hump as H
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho; uvec = np.array(G.us); V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q
xc = np.linspace(-.995, .995, 200) * rho; ss = np.arange(.02, 1.6001, .02); XC, SS = np.meshgrid(xc, ss); Sg = np.arange(0, 2.2, .0005); fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, u in zip(ax, (1., -1.)):
    li = int(np.flatnonzero(uvec == u)[0]); x = XC + u * SS * SS / 2; v = -u * SS; P = np.stack([x.ravel(), v.ravel()], 1); out = np.linalg.norm(P, axis=1) > rho          # назад на s: x = x_c + u s²/2, v = −u s
    tin = np.concatenate([H.hit_time(P[i:i + 400, 0], P[i:i + 400, 1], u, rho, Sg) for i in range(0, len(P), 400)]); q = G.Vlayers(P)[:, li]; e = np.where(out & np.isfinite(q), q - tin, np.nan); E = e.reshape(XC.shape); r = np.abs(XC.ravel()) / rho
    print('RES M=%d слой u=%+g: точек вне круга %d, запрос определён у %.3f | Q_u − t_входа: мед %+.4f, |.| мед %.4f p90 %.4f p99 %.4f max %.3f; <−.02: %.3f, >+.02: %.3f' % (G.M, u, out.sum(), np.isfinite(e[out]).mean(), np.nanmedian(e), np.nanmedian(np.abs(e)), *np.nanpercentile(np.abs(e), [90, 99]), np.nanmax(np.abs(e)), np.nanmean(e < -.02), np.nanmean(e > .02)), flush=True)
    for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.)):
        m = (r >= lo) & (r < hi) & np.isfinite(e)
        if m.any(): print('RES    |x_c|/ρ ∈ [%.2f, %.2f): |ош| мед %.4f p90 %.4f max %.3f, знак мед %+.4f' % (lo, hi, np.median(np.abs(e[m])), np.percentile(np.abs(e[m]), 90), np.abs(e[m]).max(), np.median(e[m])), flush=True)
    for lo, hi in ((0, .2), (.2, .5), (.5, 1.), (1., 1.6)):
        m = (SS.ravel() >= lo) & (SS.ravel() < hi) & np.isfinite(e)
        if m.any(): print('RES    s ∈ [%.1f, %.1f) с назад от цели: |ош| мед %.4f p90 %.4f max %.3f' % (lo, hi, np.median(np.abs(e[m])), np.percentile(np.abs(e[m]), 90), np.abs(e[m]).max()), flush=True)
    im = a.pcolormesh(XC / rho, SS, E, cmap='RdBu_r', vmin=-.1, vmax=.1, shading='auto'); a.set_xlabel('x_c / ρ — где траектория пересекает ось v=0 внутри цели (±1 — касательные края полосы)'); a.set_ylabel('s — время назад от цели, с'); a.set_title('Спора от цели, слой u=%+g, M=%d: запрос Q_u − точное время входа' % (u, G.M), fontsize=9); fig.colorbar(im, ax=a)
fig.tight_layout(); fig.savefig('goal_spore_M%d.png' % G.M, dpi=110); print('RES картинка', flush=True)
