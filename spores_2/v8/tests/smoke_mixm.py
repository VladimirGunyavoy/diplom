import os, sys; sys.path.insert(0, '.')
os.environ.update(SYS='pend', V8_WINDOW='offscreen')
import numpy as np
from src.stepper import growview as GV, vsurf as VS
def cell(M, nt=12, x0=0.):
    s = np.linspace(-.1, .1, M); t = np.linspace(0, 1, nt)
    G = np.zeros((nt, M, 2)); G[..., 0] = x0 + t[:, None] * .5; G[..., 1] = s[None, :] + .1 * t[:, None]
    return dict(G=G, c=G[nt // 2, M // 2], r=[.1], e=[[0, 1.]], nb=5, nf=5, tau=np.tile(t[:, None], (1, M)), tt=t, Gt=None)
cs = [cell(5), cell(9, x0=.6), cell(17, x0=1.2)]
snap = dict(cells=cs, cell=None, queue=np.zeros((0, 2)), seed=np.zeros(2), phase='done', u=.3)
for skel in ('normal', 'time'):
    g = GV.build_strips(snap, skel=skel); print(skel, {k: len(v) for k, v in g.items()})
Vs = {0.3: [np.random.rand(len(c['G']), c['G'].shape[1]) for c in cs]}
m = VS.build_mesh(cs, Vs[0.3]); print('mesh', {k: np.shape(v) for k, v in m.items()})
print('starts', len(VS.starts_grid_cells({0.3: cs}, 4)))
