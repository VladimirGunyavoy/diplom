"""ENV4 (2): классификация проигрыша на старт. Запуск (aida, bench/ как у spores_drone): EULER=1 RHOB=.3 python diag_chain.py 6000 400 5 [i ...]
Для плана corridor_query: конец по flow (Эйлер) vs по verify; мин. зазор вдоль verify; max|v|,|ω|; в какой момент касание; звено (прямое/стык/обратное) — по времени касания относительно длительностей прямой/обратной части не известны → по доле времени."""
import os, sys, json
BENCH = os.path.expanduser(os.environ.get('BENCH', '~/spore_v5/a31/spores_2/bench')); src = open(os.path.join(BENCH, 'spores_drone.py')).read(); cut = src.index("if __name__ == '__main__':")
ids = [int(a) for a in sys.argv[4:]]; sys.argv = sys.argv[:4]; g_ = {'__name__': 'sd', '__file__': os.path.join(BENCH, 'spores_drone.py')}; exec(compile(src[:cut], 'sd', 'exec'), g_)
import numpy as np
S, X0, flow, clr, verify, corridor_query, back, g, miss, K, NF, RHOF, PAD, US_ = [g_[k] for k in 'S X0 flow clr verify corridor_query back g miss K NF RHOF PAD US_'.split()]
def trace(x0, seq, dts, dt=.02):
    x, y, th, vx, vy, w = [float(v) for v in x0]; ends = np.cumsum(dts); T = 0.; k = 0; mc = 9; vm = wm = 0; tt = None
    while T < ends[-1] + 3.:
        while k < len(seq) - 1 and T >= ends[k]: k += 1
        u = US_[seq[k]] if T < ends[-1] else US_[seq[-1]]; c, s = np.cos(th), np.sin(th); sp = np.hypot(vx, vy)
        vx += (c * u[0] - s * u[1] - .05 * sp * vx) * dt; vy += (s * u[0] + c * u[1] - .05 * sp * vy - .5) * dt; w += .2 * u[1] / .1 * dt; x += vx * dt; y += vy * dt; th += w * dt; T += dt
        c_ = float(clr(np.array([x, y]))); 
        if c_ < mc: mc = c_; tt = T
        vm = max(vm, np.hypot(vx, vy), abs(vx), abs(vy)); wm = max(wm, abs(w))
        if y >= 4 and abs(x) <= .4: return dict(hit=True, T=T, minclr=mc, t_minclr=tt, vmax=vm, wmax=wm)
        if y > 4.3 or y < -.3: break
    return dict(hit=False, T=T, minclr=mc, t_minclr=tt, vmax=vm, wmax=wm, end=[x, y])
for i in ids:
    b = corridor_query(S, flow, X0[i], back, .4, NF, RHOF, g, miss, K=K, tries=3, kn=8, neigh=2, nbest=1, clear=lambda P: clr(P) - PAD, wlim=None)
    if b is None: print(i, 'no_path', flush=True); continue
    seq, dts = list(b[1]), list(b[2]); xe = X0[i].copy()
    for s_, t_ in zip(seq, dts): xe = flow(xe, s_, t_)
    print(i, 'Tplan %.2f nseg %d' % (b[0], len(seq)), 'конец по flow x,y,|v|,|w|', np.round([xe[0], xe[1], np.hypot(xe[3], xe[4]), abs(xe[5])], 2), 'flow minclr/PAD ok', 'trace', {k: (np.round(v, 2).tolist() if not isinstance(v, bool) else v) for k, v in trace(X0[i], seq, dts).items()}, 'dts', np.round(dts, 3).tolist(), flush=True)
