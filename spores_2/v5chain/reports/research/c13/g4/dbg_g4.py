import sys, runpy, numpy as np
sys.argv = ['g4_map.py', 'di4', '2000', '.5']
g = runpy.run_path('g4_map.py', run_name='lib')
P, tree, EM, h, edges, adv, wrap, inside, diff, US, N = [g[k] for k in ('P','tree','EM','h','edges','adv','wrap','inside','diff','US','N')]
src = np.arange(40); ei, ey, et, ed = edges(P, tree, src, EM)
tt = np.linspace(0, 1., 2001)
n = 0
for i, y, t, dd in sorted(zip(ei, ey, et, ed))[:100000]:
    best = 9
    for u in US:
        tr = wrap(adv(P[i][None].repeat(len(tt), 0), u, tt)); okk = np.maximum.accumulate(~inside(tr)); tr = tr[~okk]
        best = min(best, np.linalg.norm(diff(P[y][None], tr), axis=-1).min())
    if best > EM + 1e-9 and n < 8: print('extra', i, y, 'map t=%.3f dist=%.4f  brute min=%.4f EM=%.4f' % (t, dd, best, EM)); n += 1
