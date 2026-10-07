"""research-22, эксп. 12г: где V < LB (невозможное) на di4 L1600 — по удалению от цели (LB) и по близости к границе поля."""
import sys, os, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
R = .35
def tbox(x0, v0, K=81):
    s_ = np.linspace(-R, R, K); tx = np.r_[s_, s_, np.full(K, -R), np.full(K, R)]; tv = np.r_[np.full(K, -R), np.full(K, R), s_, s_]; out = np.full(len(x0), np.inf)
    for a in range(0, len(x0), 20000):
        x = x0[a:a + 20000, None]; v = v0[a:a + 20000, None]; best = np.full(x.shape[0], np.inf)
        for s in (1., -1.):
            q = s * (tx - x) + (v * v + tv * tv) / 2; vm = s * np.sqrt(np.maximum(q, 0)); t1 = s * (vm - v); t2 = s * (vm - tv); best = np.minimum(best, np.where((q >= 0) & (t1 >= -1e-12) & (t2 >= -1e-12), t1 + t2, np.inf).min(1))
        out[a:a + 20000] = best
    out[(np.abs(x0) <= R) & (np.abs(v0) <= R)] = 0.; return out
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); P = A.P; T1 = tbox(P[:, 0], P[:, 2]); T2 = tbox(P[:, 1], P[:, 3]); LB = np.maximum(T1, T2)
stop = np.abs(np.c_[P[:, 0] + P[:, 2] * np.abs(P[:, 2]) / 2, P[:, 1] + P[:, 3] * np.abs(P[:, 3]) / 2]).max(1); edge = np.abs(P).max(1)
for k in (1, 0):
    V = np.load('V_sblay%d.npy' % k); fin = V < G.BIG / 2; d = V - LB; print('SBLAY', k, '| конечных %.3f' % fin.mean())
    for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 3), (3, 4), (4, 9)):
        m = fin & (LB >= lo) & (LB < hi); print('  LB [%.1f, %.1f): узлов %7d | V − LB: мед. %+.3f p5 %+.3f p1 %+.3f | V < LB − .02: %.3f | V < LB − .1: %.3f | V < LB − .3: %.3f' % (lo, hi, m.sum(), np.median(d[m]), *np.percentile(d[m], [5, 1]), (d[m] < -.02).mean(), (d[m] < -.1).mean(), (d[m] < -.3).mean()))
    m = fin & (LB > .5); bad = d < -.1
    print('  среди V < LB − .1: доля с тормозным путём за полем (> 2.5): %.3f (у всех узлов %.3f) | с |коорд.| > 2.2: %.3f (у всех %.3f) | min(T1, T2)/LB мед. %.3f (у всех %.3f)' % ((stop[m & bad] > 2.5).mean(), (stop[m] > 2.5).mean(), (edge[m & bad] > 2.2).mean(), (edge[m] > 2.2).mean(), np.median(np.minimum(T1, T2)[m & bad] / LB[m & bad]), np.median(np.minimum(T1, T2)[m] / LB[m])))
