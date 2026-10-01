"""Тёплый старт OCP от пути коридора (stats_3down/3up.json worker'а: res[i].seq/dts) → T_warm ≤ T_corr (если IPOPT не ушёл хуже).
Запуск: PYTHONPATH=<casadi> G=0.3 DOWN=1|UP=1 WMP=50 python3 ref6d_warm.py stats.json i [N M]  → строка 'q<i+1> T_corr .. T_warm ..'"""
import sys, json, time
import numpy as np
import ref6d_ocp as R
res = json.load(open(sys.argv[1]))['res']; i = int(sys.argv[2]); N, M = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (60, 4)
o = res[i]; x0 = R.starts(len(res))[i]; t0 = time.time()
if not o.get('seq'): print('q%d нет пути коридора' % (i + 1)); sys.exit()
best = None
for sc in (1.0, 1.05):                                                # затравка T: как у коридора и +5% (запас допустимости)
    r = R.warm_start(x0, o['seq'], np.array(o['dts']) * sc, N, M)
    if r is not None and (best is None or r[0] < best): best = r[0]
print('q%d T_corr %.4f ok %s T_warm %s (%.0f с)' % (i + 1, o['T'], o['ok'], None if best is None else '%.4f' % best, time.time() - t0), flush=True)
