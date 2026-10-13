"""w28: m3gu — почему CAREF хуже: на стартах из argv сравнить IPOPT и SLSQP refine по топ-кандидатам (запуск: CAREF=1 CAPRE=1 python dbg_gu.py m3gu i1 i2 ...)"""
import sys, os, time
ids = [int(a) for a in sys.argv[2:]]; sys.argv = ['x', sys.argv[1], '12000', '3000', '3', '', 'wlim']; os.environ['CAREF'] = '1'; os.environ['CAPRE'] = '1'
__file__ = os.path.abspath('spores_6d.py'); exec(open('spores_6d.py').read().split("if __name__ == '__main__':")[0])
from src.atlas6 import corridor_nd as C
wl = ((3, 4, 5), WM)
for i in ids:
    cand, nh = C.candidates(S, e.X0[i], back, tau, NF, rho, miss, 8)
    print('== старт', i, 'кандидатов', len(cand), 'попаданий', nh, flush=True); seen = set(); k = 0
    for hit, sc, p in cand[:nh]:
        sq = tuple(C.merge(p)[0])
        if sq in seen: continue
        seen.add(sq); k += 1
        t0 = time.perf_counter(); a = CAR.refine(fl, e.X0[i], p, g, tries=3, wlim=wl); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); b = CAR.orig(fl, e.X0[i], p, g, tries=3, wlim=wl); tb = time.perf_counter() - t0
        print(' n=%d T0=%.3f | IPOPT ok=%s T=%.3f %.1fс | SLSQP ok=%s T=%.3f %.1fс' % (len(sq), sum(C.merge(p)[1]), a[3], a[0], ta, b[3], b[0], tb), flush=True)
        if k >= 3: break
