import sys, os
sys.argv = ['x', 'm3d', '12000', '3000', '3']; os.environ['CAREF'] = '1'; os.environ['CAPRE'] = '1'
__file__ = os.path.abspath('spores_6d.py'); exec(open('spores_6d.py').read().split("if __name__ == '__main__':")[0])
from src.atlas6 import corridor_nd as C
cand, nh = C.candidates(S, e.X0[0], back, tau, NF, rho, miss, 8)
print('кандидатов', len(cand), 'попаданий', nh)
import numpy as np
for hit, sc, p in cand[:4]:
    seq, d0 = C.merge(p); print('n сегм', len(seq), 'T0 %.3f' % d0.sum(), 'hit', hit)
    c = CAR; S_, lbg, ubg = c.solver(len(seq), False)
    tau_ = c.US[list(seq)].T
    xe0 = C.endpoint(fl, e.X0[0], seq, d0); tgt = c.C3 + 2 * np.pi * np.round((xe0[:3] - c.C3) / (2 * np.pi)); print('xe0', xe0.round(2), 'tgt', tgt)
    P = np.r_[e.X0[0], tau_.ravel(order='F'), tgt]; r = S_(x0=d0, p=P, lbx=0., ubx=1.5 * d0.sum() + 1, lbg=lbg, ubg=ubg); st = S_.stats()
    d = np.array(r['x']).ravel(); xe = C.endpoint(fl, e.X0[0], seq, d); print(st['return_status'], st['iter_count'], 'T', d.sum().round(3), 'g', g(xe).round(3), 'xe', xe.round(2))
