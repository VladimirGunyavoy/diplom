"""w28: 8D — почему теряются старты (tf1): кандидаты дерева и исходы IPOPT/SLSQP-refine. python dbg_m8.py i..."""
import sys, os, time
ids = [int(a) for a in sys.argv[1:]]; sys.argv = ['x', 'm8', os.environ.get('NBP', '60000'), os.environ.get('NFP', '15000'), '3', '_', 'wlim']
__file__ = os.path.abspath('spores_8d.py'); exec(open('spores_8d.py').read().split("if __name__ == '__main__':")[0])
from src.atlas6 import corridor_nd as CN
wl = (tuple(range(n, 2 * n)), WM)
for i in ids:
    t0 = time.perf_counter(); cand, nh = CN.candidates(S, e.X0[i], back, tau, NF, rho, miss, 8, hfun=HF)
    print('== старт', i, 'кандидатов', len(cand), 'попаданий', nh, 't %.0f с' % (time.perf_counter() - t0), flush=True); seen = set(); k = 0
    for hit, sc, p in cand[:nh]:
        sq = tuple(CN.merge(p)[0])
        if sq in seen: continue
        seen.add(sq); k += 1; seq, d0 = CN.merge(p); CAR.st.clear()
        t0 = time.perf_counter(); a = CAR.refine(fl, e.X0[i], p, g, tries=3, wlim=wl); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); b = CAR.orig(fl, e.X0[i], p, g, tries=3, wlim=wl); tb = time.perf_counter() - t0
        print(' n=%d T0=%.2f | IPOPT ok=%s T=%.2f %.1fс %s | SLSQP ok=%s T=%.2f %.0fс' % (len(seq), d0.sum(), a[3], a[0], ta, dict(CAR.st), b[3], b[0], tb), flush=True)
        if k >= 3: break
    if not nh:
        for hit, sc, p in cand[:3]:
            seq, d0 = CN.merge(p); a = CAR.refine(fl, e.X0[i], p, g, tries=3, wlim=wl); print(' miss n=%d T0=%.2f score %.2f | IPOPT ok=%s' % (len(seq), d0.sum(), sc, a[3]), flush=True)
