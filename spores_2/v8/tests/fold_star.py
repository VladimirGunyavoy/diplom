"""W2: картинка V*−T_ref вокруг круга из npz fold_map (имена через пробел в env FILES: подпись=файл), M=3/5, XENTRY. Из v8."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fs = [t.split('=') for t in os.environ['FILES'].split()]; fig, ax = plt.subplots(1, len(fs), figsize=(6.2 * len(fs), 5))
for a, (name, f) in zip(np.atleast_1d(ax), fs):
    d = np.load(f); e = d['err_star']; p = a.pcolormesh(d['xs'], d['vs'], e, cmap='RdBu_r', vmin=-.3, vmax=.3, shading='auto'); plt.colorbar(p, ax=a); a.add_patch(plt.Circle((0, 0), .2, fill=False, color='k')); a.set_aspect('equal'); a.set_title('V* − T_ref: ' + name, fontsize=10); a.set_xlabel('x'); a.set_ylabel('v')
    X, V = np.meshgrid(d['xs'], d['vs']); r = np.hypot(X, V); ok = np.isfinite(e) & (r > .2)
    for lo, hi in ((.2, .3), (.3, .45), (.45, .7)):
        m = ok & (r > lo) & (r <= hi); print('RES %s r∈(%.2f,%.2f]: покрыто %.0f%%, мед %.3f p10 %.3f p90 %.3f, занижено<−.03: %.1f%%' % (name, lo, hi, 100 * m.sum() / ((r > lo) & (r <= hi)).sum(), np.median(e[m]), np.percentile(e[m], 10), np.percentile(e[m], 90), 100 * (e[m] < -.03).mean()))
plt.tight_layout(); plt.savefig(os.environ.get('OUT', 'fold_star.png'), dpi=75)
