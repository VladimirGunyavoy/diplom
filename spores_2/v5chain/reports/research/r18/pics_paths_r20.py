# research-20 (слово пользователя): пути агента из 100 стартов — цвет отрезка = управление на шаге, полупрозрачно; фон — V* градиентом; θ ∈ [−XR, XR] с повторами
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.collections import LineCollection
plt.rcParams.update({'font.size': 13})
Z = np.load(sys.argv[1]); OUT = sys.argv[2]; TITLE = os.environ.get('TITLE', ''); XR = float(os.environ.get('XR', 6.)); RHO = .1
gx, gw, VG, P, U, T = Z['gx'], Z['gw'], Z['VG'], Z['P'], Z['U'], Z['T']; VG = np.where(VG > 500, np.nan, VG)
SH = [k * 2 * np.pi for k in range(-2, 3)]; UC = {-0.3: '#2166ac', 0.0: '#4d4d4d', 0.3: '#d6604d'}
fig, ax = plt.subplots(figsize=(22, 9.5))
for sh in SH:
    im = ax.pcolormesh(gx + sh, gw, VG.T, cmap='YlGnBu', shading='auto', vmin=0, vmax=np.nanquantile(VG, .99), alpha=.85, rasterized=True)
    ax.contour(gx + sh, gw, VG.T, levels=np.arange(1, np.nanmax(VG), 1.), colors='w', linewidths=.6, alpha=.7)
segs = {u: [] for u in UC}
for i in range(P.shape[1]):
    p = P[:, i]; mv = np.r_[True, np.any(np.diff(p, axis=0) != 0, 1)]; k = np.flatnonzero(mv)[-1] + 1; p = p[:k].copy(); p[:, 0] = np.unwrap(p[:, 0]); u = U[:k - 1, i]
    for j in range(k - 1):
        if np.isnan(u[j]): continue
        for sh in SH: segs[round(float(u[j]), 1)].append(p[j:j + 2] + [sh, 0])
for u, c in UC.items(): ax.add_collection(LineCollection(segs[u], colors=c, linewidths=1.6, alpha=.45))
for sh in SH: ax.plot(P[0, :, 0] + sh, P[0, :, 1], 'o', ms=3.5, color='k', alpha=.6); ax.add_patch(plt.Rectangle((-RHO + sh, -RHO), 2 * RHO, 2 * RHO, fill=False, ec='k', lw=2))
ax.set_xlim(-XR, XR); ax.set_ylim(gw[0], gw[-1]); ax.set_xlabel('θ  (повтор через 2π; пунктир — ±π)'); ax.set_ylabel('ω')
for v in (-np.pi, np.pi): ax.axvline(v, color='0.3', ls=':', lw=1)
fig.colorbar(im, ax=ax, pad=.01, label='V* — время до цели по атласу, с (белые линии — через 1 с)')
h = [plt.Line2D([], [], color=c, lw=4, alpha=.7) for c in UC.values()] + [plt.Line2D([], [], marker='o', ls='', color='k')]
ax.legend(h, ['u = −0.3', 'u = 0', 'u = +0.3', 'старт'], loc='upper center', bbox_to_anchor=(.5, -.07), ncol=4, frameon=False)
ax.set_title(TITLE + '   дошли %d/%d' % (np.isfinite(T).sum(), len(T)))
fig.savefig(OUT, dpi=80, bbox_inches='tight'); print('готово', OUT)
