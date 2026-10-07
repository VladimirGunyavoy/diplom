"""Время build_layer по слоям + сверка узлов с эталоном (утилита). python tests/bench_layers.py [--save f.npz | --cmp f.npz] [--rng 0,1,2]
Env как live.py (MAXC 60, NFAIL 400). Сверка: число клеток и max|ΔG|, max|Δtau| по принятым клеткам."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
for k, v in dict(SYS='di', M='3', KF='21', MAXC='60', NFAIL='400', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000', GS='0', GLIM='0', GSEED='1', GOALB='0', GOALSHAPE='ball', RHO='.2').items(): os.environ.setdefault(k, v)
import numpy as np
from src.algo import growN as g
a = sys.argv[1:]; rngs = [int(x) for x in a[a.index('--rng') + 1].split(',')] if '--rng' in a else [0]
ref = np.load(a[a.index('--cmp') + 1]) if '--cmp' in a else None; out = {}; tot = 0.
for u in g.ULAYERS:
    for sd in rngs:
        t = time.perf_counter(); cells, idx = g.build_layer(u, np.random.default_rng(sd)); t = time.perf_counter() - t; tot += t; msg = ''
        for j, c in enumerate(cells): out['%g_%d_%d_G' % (u, sd, j)] = c.G; out['%g_%d_%d_tau' % (u, sd, j)] = c.tau
        if ref is not None:
            n0 = len([k for k in ref.files if k.startswith('%g_%d_' % (u, sd)) and k.endswith('_G')])
            d = [(np.abs(c.G - ref['%g_%d_%d_G' % (u, sd, j)]).max() if c.G.shape == ref['%g_%d_%d_G' % (u, sd, j)].shape else np.inf) for j, c in enumerate(cells) if j < n0]
            msg = ' | ref cells %d, max|dG| %.1e' % (n0, max(d) if d else 0.)
        print('u=%g rng=%d cells=%d %.2fs%s' % (u, sd, len(cells), t, msg), flush=True)
print('total %.2fs' % tot)
if '--save' in a: np.savez(a[a.index('--save') + 1], **out)
