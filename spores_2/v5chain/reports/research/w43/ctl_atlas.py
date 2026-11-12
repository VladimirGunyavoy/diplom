"""w43 контроль (в): траектории атласа (m3gdg_spores_w43ctl.json, us/dts) через МОЙ numba-поток + okv/in_goal/зазор графа."""
import os, sys, json
sys.argv = ['x', '19', '0']; os.environ['NB'] = '1000'
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3gdg_graph as M
M.Rq, M.Rw = .301, .601
d = json.load(open(sys.argv[0] and os.path.expanduser('~/spore_v5/w43/reports/bench/data/m3gdg_spores_w43ctl.json'))); Tr = M.transitions(19)
for r in d['res']:
    k = r['k']; st, gl = Tr[k]; x = np.r_[st, 0., 0., 0.]; t = 0.; hit = None; mw = 0.; mc = 9.; ttot = float(sum(r['dts']))
    for u, dt in zip(r['us'], r['dts']):
        n = max(1, int(np.ceil(dt / .01))); h = dt / n; u = np.array(u, float)
        for _ in range(n):
            k1 = M.acc(x, u, M.A, M.Ai, M.Bm, M.Sm, M.gw); k2 = M.acc(x + h / 2 * k1, u, M.A, M.Ai, M.Bm, M.Sm, M.gw); k3 = M.acc(x + h / 2 * k2, u, M.A, M.Ai, M.Bm, M.Sm, M.gw); k4 = M.acc(x + h * k3, u, M.A, M.Ai, M.Bm, M.Sm, M.gw)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); x[:3] = (x[:3] + M.PI) % (2 * M.PI) - M.PI; t += h; mw = max(mw, np.abs(x[3:]).max()); mc = min(mc, float(M.G3.clearance(x[:3], M.OB)))
            if hit is None and M.in_goal(x, np.asarray(gl, float)): hit = t
    print('k=%d Tatlas %.3f | мой поток: попал в окно (.3/.6) %s, max|w| %.3f, мин зазор %.3f, граф-критерий (|w|≤3.05, зазор≥%.2f): %s' % (k, r['T'], None if hit is None else round(hit, 3), mw, mc, M.CL, 'OK' if (mw <= 3.05 and mc >= M.CL) else 'НАРУШЕНИЕ'))
