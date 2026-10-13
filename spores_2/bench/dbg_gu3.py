"""w28: m3gu старт i: найти путь, на котором SLSQP выигрывает, и повторить на нём IPOPT"""
import sys, os, time
ids = [int(a) for a in sys.argv[2:]]; sys.argv = ['x', sys.argv[1], '12000', '3000', '3', '', 'wlim']; os.environ['CAREF'] = '1'; os.environ['CAPRE'] = '1'
__file__ = os.path.abspath('spores_6d.py'); exec(open('spores_6d.py').read().split("if __name__ == '__main__':")[0])
from src.atlas6 import corridor_nd as C
wl = ((3, 4, 5), WM); sl = CAR.orig; rec = []
def tr(flow, x0, path, g_, **kw):
    r = sl(flow, x0, path, g_, **kw); rec.append((path, r)); return r
for i in ids:
    C.refine = tr; rec.clear(); b = C.corridor_query(S, fl, e.X0[i], back, tau, NF, rho, g, miss, K=3, tries=3, kn=8, wlim=wl)
    print('== старт', i, 'SLSQP итог T=%.3f' % b[0], 'всего refine', len(rec), flush=True)
    good = [(r[0], p, r) for p, r in rec if r[3]]; good.sort(key=lambda z: z[0])
    for T, p, r in good[:3]:
        seq, d0 = C.merge(p); CAR.st.clear()
        a = CAR.__class__.refine(CAR, fl, e.X0[i], p, g, tries=3, wlim=wl)
        print(' SLSQP T=%.3f n=%d seq=%s d0=%s -> d=%s | IPOPT ok=%s T=%.3f' % (T, len(seq), list(seq), d0.round(2).tolist(), np.round(r[2], 2).tolist(), a[3], a[0]), CAR.st, flush=True)
