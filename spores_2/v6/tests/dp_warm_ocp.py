import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.corridor_nd import refine
from src.atlas6.manip3dyn import flow as _fl
GG = 2.0; Rq, Rw = .3, .5; UP = np.array([np.pi/2, 0, 0, 0]); wr = lambda x: (x + np.pi) % (2*np.pi) - np.pi
fl = lambda P, s, t: _fl(P, s, t, dt_max=.02, n=2, g=GG)
g = lambda x: np.array([Rq**2 - wr(x[0]-UP[0])**2, Rq**2 - wr(x[1])**2, Rw**2 - x[2]**2, Rw**2 - x[3]**2])
a = np.load('/tmp/claude-1000/refdp_G2_W3_N80.npy'); T = a[0]; u = a[1:].reshape(80, 2); h = T/80
x0 = np.array([-np.pi/2, 0, 0, 0.]); VAR = sys.argv[2] if len(sys.argv) > 2 else 'A'
path = []
for ui in u:
    s1, s2 = int(ui[0] > 0), int(ui[1] > 0)
    if abs(ui[0]) < .99 and VAR == 'B':                       # сингулярный шаг (u1=.72): чаттеринг ±1 с долей (1+u)/2
        f = (1 + ui[0]) / 2; path += [(1 | (s2 << 1), h * f), (0 | (s2 << 1), h * (1 - f))]
    else: path.append((s1 | (s2 << 1), h))
seq = None
from src.atlas6.corridor_nd import merge
print('path format', type(path[0]), 'сегментов после merge', len(merge(path)[0]), flush=True)
for WL in ((slice(2, 4), 3.0),):
    t0 = time.time(); r = refine(fl, x0, path, g, tries=int(sys.argv[1]) if len(sys.argv) > 1 else 5, wlim=WL)
    xe = x0.copy()
    for s_, d in zip(r[1], r[2]):
        for _ in range(8): xe = _fl(xe, s_, d/8, dt_max=.005, n=2, g=GG)
    print('wlim', WL, 'T', round(r[0], 3), 'ok', r[3], 'сегм', len(r[1]), 'g_min rk4', round(float(g(xe).min()), 4), 'сек', round(time.time()-t0), flush=True)
