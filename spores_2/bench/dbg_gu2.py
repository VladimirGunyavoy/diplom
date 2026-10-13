"""w28: m3gu — трасса всех refine в corridor_query: IPOPT(+TREEFAST) vs штатный (запуск: python dbg_gu2.py m3gu i...; env TREEFAST/BATCHFAST как в spdA)"""
import sys, os, time
ids = [int(a) for a in sys.argv[2:]]; sys.argv = ['x', sys.argv[1], '12000', '3000', '3', '', 'wlim']; os.environ['CAREF'] = '1'; os.environ['CAPRE'] = '1'
__file__ = os.path.abspath('spores_6d.py'); exec(open('spores_6d.py').read().split("if __name__ == '__main__':")[0])
from src.atlas6 import corridor_nd as C
wl = ((3, 4, 5), WM); ipo = C.refine; log = []
def tr(flow, x0, path, g_, **kw):
    t0 = time.perf_counter(); r = ipo(flow, x0, path, g_, **kw); log.append((len(r[1]), round(r[0], 3), r[3], round(time.perf_counter() - t0, 2))); return r
for i in ids:
    C.refine = tr; log.clear(); t0 = time.perf_counter()
    b = C.corridor_query(S, fl, e.X0[i], back, tau, NF, rho, g, miss, K=3, tries=3, kn=8, wlim=wl)
    print('== старт', i, 'IPOPT: T=%.3f n=%d %.1fс' % (b[0], len(b[1]), time.perf_counter() - t0), 'refine(n,T,ok,t):', log, flush=True)
    C.refine = CAR.orig; log.clear(); t0 = time.perf_counter()
    b = C.corridor_query(S, fl, e.X0[i], back, tau, NF, rho, g, miss, K=3, tries=3, kn=8, wlim=wl)
    print('   статусы IPOPT:', CAR.st); CAR.st.clear()
    print('   SLSQP: T=%.3f n=%d %.1fс' % (b[0], len(b[1]), time.perf_counter() - t0), flush=True)
