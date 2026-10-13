"""офскрин: рост обоих слоёв pend с env main_pend (SEEDELL, FMIN, SELFOV…), картинка клеток целиком + зум у низа (θ=π) и сепаратрисы: python tests/ell_look.py out.png [MAXC]"""
import os, sys, re, ast, time
V8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, V8)
src = open(os.path.join(V8, "main_pend.py"), encoding="utf-8").read()
m = re.search(r"for _k, _v in dict\((.*?)\)\.items\(\)", src, re.S).group(1)
env = dict(re.findall(r"(\w+)='([^']*)'", m))
for k, v in env.items(): os.environ.setdefault(k, v)
os.environ['V8_MAXCELLS'] = '0'; os.environ['MAXC'] = sys.argv[2] if len(sys.argv) > 2 else '60'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.algo import growN as g
fig, axs = plt.subplots(2, 3, figsize=(18, 11)); t0 = time.time()
for r, u in enumerate([u for u in g.US if u != 0]):
    cl = [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(r), seeds=None, only_queue=False)[0]]
    print('u=%+g cells %d, %.0f s, ELLS %s' % (u, len(cl), time.time() - t0, [round(x, 2) for x in g.ELLS.get(u, [])]), flush=True)
    for c_, (xl, yl, ttl) in enumerate([((-8, 8), (-3.6, 3.6), 'all'), ((np.pi - 1.5, np.pi + 1.5), (-1.5, 1.5), 'bottom θ=π'), ((-1.2, 1.2), (-1.5, 1.5), 'goal/separatrix')]):
        ax = axs[r, c_]
        for k, c in enumerate(cl):
            G = np.asarray(c['G'], float)
            for sh in (-2 * np.pi, 0, 2 * np.pi):
                ring = np.concatenate([G[0], G[:, -1], G[-1][::-1], G[:, 0][::-1]]); ax.plot(ring[:, 0] + sh, ring[:, 1], color=plt.cm.tab10(k % 10), lw=.7)
                for row in G[::4]: ax.plot(row[:, 0] + sh, row[:, 1], color='gray', lw=.3)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_title('u=%+g %s (%d cells)' % (u, ttl, len(cl))); ax.set_aspect('equal') if c_ else None
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=70)
