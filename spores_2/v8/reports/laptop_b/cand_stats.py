import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
X = S.starts(g, 200, 1); idx = G.idx[1.0]; HI = g.HexIdx; print('hypercells', idx.n, 'bins', len(idx.bins))
cnt = {}
o = HI._test
def t(s, Y, pi, vv, tolc=1e-10):
    cnt['cand'] = cnt.get('cand', 0) + len(pi); r = o(s, Y, pi, vv, tolc); cnt['out'] = cnt.get('out', 0) + len(r[0]); return r
HI._test = t
for u in G.us: idx = G.idx[u]; idx.query(X); 
print('pts', len(X) * 3, cnt)
