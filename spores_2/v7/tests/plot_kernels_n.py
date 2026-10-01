"""Картинки ЯДЕР nD-клеток: проекция ядра (выпуклая оболочка образов точек сегмента-диска × t) на выбранные координаты + центральный путь. py tests/plot_kernels_n.py <система> [N] [пары координат: 01,23]"""
import sys; sys.path.insert(0, '.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from src.cells7.systemsn import SYSTEMS
from src.cells7.covern import cover_layer_n
name = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 120; pairs = [(int(p[0]), int(p[1])) for p in (sys.argv[3] if len(sys.argv) > 3 else '01').split(',')]
S = SYSTEMS[name](); rng = np.random.default_rng(0); fig, ax = plt.subplots(len(pairs), 2, figsize=(12, 5.5 * len(pairs)), squeeze=False); sizes = []
for k in (0, 1):
    cells = cover_layer_n(S, k, 2000, N, 400)[0]
    for C in cells:
        U = rng.normal(size=(60, C.m)); U /= np.linalg.norm(U, axis=1)[:, None]; U = np.vstack([U * C.r, np.zeros((1, C.m))]); tt = np.linspace(0, C.tau, 9)
        Y = np.concatenate([C.phi(U, np.full(len(U), t)) for t in tt]); sizes.append((C.r, C.tau))
        for row, (a, b) in enumerate(pairs):
            Z = Y[:, [a, b]]
            try: h = ConvexHull(Z); ax[row, k].fill(Z[h.vertices, 0], Z[h.vertices, 1], alpha=.18, lw=.3)
            except Exception: pass
            L = C.phi(np.zeros((9, C.m)), tt); ax[row, k].plot(L[:, a], L[:, b], 'k-', lw=.5); ax[row, k].plot(C.c[a], C.c[b], 'r.', ms=2)
            ax[row, k].set_title('%s слой %d: %d клеток, ядра в проекции (%d,%d)' % (name, k, len(cells), a, b), fontsize=9)
sz = np.array(sizes); print(name, 'клеток', len(sz), 'r мед %.3f τ мед %.3f' % (np.median(sz[:, 0]), np.median(sz[:, 1])))
plt.tight_layout(); plt.savefig('reports/figures/kernels_%s.png' % name, dpi=80)
