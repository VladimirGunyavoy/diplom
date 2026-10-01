import sys, time, json; sys.path.insert(0, '.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.cells7.systems import di, pend
from src.cells7.cover import cover_layer, metrics
name = sys.argv[1]; m = int(sys.argv[2]) if len(sys.argv) > 2 else 100
S = di() if name == 'di' else pend(0.3)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5)); out = {}
for k in (0, 1):
    t0 = time.time(); cells, P, cov = cover_layer(S, k, m=m, log=lambda n, c: print(name, k, n, round(c, 3), flush=True)); dt = time.time() - t0
    M = metrics(S, cells, P, cov); M['sec'] = dt; out['layer%d' % k] = M; print(name, k, M, flush=True)
    for C in cells:
        a, b = C.outline(); ax[k].fill(np.r_[a[:, 0], b[::-1, 0]], np.r_[a[:, 1], b[::-1, 1]], alpha=0.25, lw=0.3, ec='k')
    ax[k].set_title('%s слой %d (u=%+.2f): %d клеток, перекрытие ядер %.1f%% гало %.1f%%' % (S.name, k, S.u(k), M['cells'], 100 * M['kernel_overlap'], 100 * M['halo_overlap']), fontsize=9)
    ax[k].set_xlim(*S.box[0]); ax[k].set_ylim(*S.box[1])
plt.tight_layout(); plt.savefig('reports/figures/cells_%s.png' % name, dpi=100); json.dump(out, open('reports/cells_%s.json' % name, 'w'), indent=1)
