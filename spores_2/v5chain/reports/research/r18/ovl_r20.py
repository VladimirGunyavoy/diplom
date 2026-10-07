import pickle, sys, numpy as np
from matplotlib.path import Path
def contour(G): return np.r_[G[:, 0], G[-1, 1:-1], G[::-1, -1], G[0, -2:0:-1]]
rng = np.random.default_rng(0); X = np.c_[rng.uniform(-np.pi, np.pi, 40000), rng.uniform(-3.5, 3.5, 40000)]
for run in sys.argv[1:]:
    C = pickle.load(open(run + '/data/cells.pkl', 'rb')); out = []
    for u in sorted(set(round(c['u'], 3) for c in C)):
        cs = [c for c in C if round(c['u'], 3) == u]; cnt = np.zeros(len(X), int); own = []
        for c in cs:
            G = np.array(c['G'], float); G[..., 0] = np.unwrap(G[..., 0], axis=0); P = Path(contour(G)); ins = np.zeros(len(X), bool)
            for sh in (-2 * np.pi, 0., 2 * np.pi): ins |= P.contains_points(X + [sh, 0])
            cnt += ins; own.append(ins)
        cov = cnt > 0; fr = [float((cnt[o] > 1).mean()) for o in own if o.any()]
        out.append('u=%+.1f: ≥2 кл. %.0f%% площади, ≥3 %.0f%%, ср. слоёв %.2f; клеток >50%% под чужими: %d/%d' % (u, 100 * (cnt >= 2)[cov].mean(), 100 * (cnt >= 3)[cov].mean(), cnt[cov].mean(), sum(f > .5 for f in fr), len(fr)))
    print(run, '|', ' ; '.join(out))
