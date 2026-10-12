"""b3: приёмка solve по вёдрам: одно решение Q (Якоби+ACTIVE vs вёдра SOLVEB) на том же графе: время, max|ΔQ|, max|ΔV|, число проходов/раундов. env — pend_env.sh"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); atlas, tg = S.build_atlas(list(g.US), 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q')
tol = float(os.environ.get('TOLX', 1e-6)); res = {}
for nm, sb in (('якоби', 0), ('вёдра', 1)):
    if sb == 0 and os.environ.get('SKIPJ') and 'ref' in globals(): continue
    S.SOLVEB = sb; t0 = time.perf_counter(); G._solve_q(3000, tol, False); dt = time.perf_counter() - t0; res[nm] = (G.V.copy(), G.Vn.copy(), dt, G.passes)
    print('RES %-6s %.1f с, проходов/раундов %d, достижимо %.1f%%' % (nm, dt, G.passes, 100 * (G.Vn < 1e3 / 2).mean()), flush=True)
Qa, Va = res['якоби'][:2]; Qb, Vb = res['вёдра'][:2]; fa = Qa < 500; fb = Qb < 500
print('RES совпадение достижимости Q: %d/%d; max|ΔQ| %.2e (p99 %.2e), max|ΔV| %.2e; ускорение ×%.1f' % ((fa == fb).sum(), len(fa), np.abs(Qa[fa & fb] - Qb[fa & fb]).max(), np.percentile(np.abs(Qa[fa & fb] - Qb[fa & fb]), 99), np.abs(Va[fa & fb] - Vb[fa & fb]).max(), res['якоби'][2] / res['вёдра'][2]))
d = (Qb - Qa)[fa & fb]; print('RES знак: вёдра ниже Якоби в %d узлах (>1e-4), выше в %d; сумма по ниже %.1f' % ((d < -1e-4).sum(), (d > 1e-4).sum(), d[d < -1e-4].sum()))
# неподвижная точка? один проход Якоби от решения вёдер
S.SOLVEB = 0; G.V = Qb.copy(); G.Vn = Vb.copy(); G._solve_q(1, 1e-12, False, warm=True); dq = np.abs(np.minimum(G.V, 1e3) - np.minimum(Qb, 1e3)); print('RES после 1 прохода Якоби от вёдер: max|ΔQ| %.2e, узлов >1e-4: %d' % (dq.max(), (dq > 1e-4).sum()))
S.SOLVEB = 0; G.V = Qa.copy(); G.Vn = Va.copy(); G._solve_q(1, 1e-12, False, warm=True); dq = np.abs(np.minimum(G.V, 1e3) - np.minimum(Qa, 1e3)); print('RES после 1 прохода Якоби от Якоби: max|ΔQ| %.2e, узлов >1e-4: %d' % (dq.max(), (dq > 1e-4).sum()))
