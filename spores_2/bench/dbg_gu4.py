import sys, os, time
sys.argv = ['x', 'm3gu', '12000', '3000', '3', '', 'wlim']; os.environ['CAREF'] = '1'; os.environ['CAPRE'] = '1'
__file__ = os.path.abspath('spores_6d.py'); exec(open('spores_6d.py').read().split("if __name__ == '__main__':")[0])
from src.atlas6 import corridor_nd as C
seq = [5, 2, 5, 1, 6, 2, 7]; d0 = np.array([.35, 0, 0, 1.57, .8, .4, .3]); x0 = e.X0[5]
n = 7; Sv, lbg, ubg = CAR.solver(n, True); tau_ = CAR.US[seq].T
xe0 = C.endpoint(fl, x0, seq, d0); tgt = CAR.C3 + 2*np.pi*np.round((xe0[:3]-CAR.C3)/(2*np.pi)); P = np.r_[x0, tau_.ravel(order='F'), tgt]
ref = np.array([.35, 0, 0, 1.57, .8, .4, 0.]); print('g(ref)', g(C.endpoint(fl, x0, seq, ref)).round(3), 'xe0', xe0.round(2), 'tgt', tgt.round(2))
for st in (d0, ref+1e-3, ref+.05):
    r = Sv(x0=st, p=P, lbx=0., ubx=1.5*d0.sum()+1, lbg=lbg, ubg=ubg); d = np.array(r['x']).ravel(); xe = C.endpoint(fl, x0, seq, d)
    print(Sv.stats()['return_status'], Sv.stats()['iter_count'], 'T %.3f' % d.sum(), d.round(3), 'g', g(xe).round(4))
