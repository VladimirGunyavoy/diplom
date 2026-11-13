"""c13 Г4.1 (И4, hyp_user_ideas_2026-11-13.md): карта переходов «узел → узел» из постоянных управлений, без цели.
Ребро x→y (цена t), если y в пределах ε от дуги flow(x,u,t), t∈[0,TMAX], u∈US. Описательно: средняя/медианная исходящая степень, доля узлов без выхода, доля крупнейшей сильно связной компоненты — для ε ∈ h·{1/8,1/4,1/3,1/2[,1]}, h=(V/N)^(1/n).
python g4_map.py di4|m3d N [EMAXF=.5] [NSRC=все] ; env: TMAX, SEED, WORKERS, CHECK=1 (контроль: перебор t по плотной сетке, N=2000).
Метрика — евклидова в «сырых» координатах (м3d: рад и рад/с вперемешку); периодика q — тор (cKDTree boxsize); дуга обрезается, где выходит из поля."""
import os, sys, time, json, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
env_name = sys.argv[1]; N = int(sys.argv[2]); EMAXF = float(sys.argv[3]) if len(sys.argv) > 3 else .5; NSRC = int(sys.argv[4]) if len(sys.argv) > 4 else N
SEED = int(os.environ.get('SEED', 0)); WK = int(os.environ.get('WORKERS', 8)); rng = np.random.default_rng(SEED)
if env_name == 'di4':
    d = 4; per = np.zeros(4); lim = np.array([2.5, 2.5, 2., 2.]); TMAX = float(os.environ.get('TMAX', 1.))
    US = np.array([[a, b] for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
    def rhs(X, u): return np.concatenate([X[..., 2:], np.broadcast_to(u, X[..., :2].shape)], -1)
    def adv(X, u, t): t = np.asarray(t, float)[..., None]; return np.concatenate([X[..., :2] + X[..., 2:] * t + u * t * t / 2, X[..., 2:] + u * t], -1)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import manip3d as M3
    d = 6; per = np.array([2 * np.pi] * 3 + [0] * 3); lim = np.array([np.pi] * 3 + [3.] * 3); TMAX = float(os.environ.get('TMAX', 1.)); US = M3.US
    def rhs(X, u): return M3.f(X, np.broadcast_to(u, X[..., :3].shape))
    def adv(X, u, t, nsub=None):
        n = max(1, int(np.ceil(float(np.max(t)) / .02))); h = np.asarray(t, float)[..., None] / n; Y = X.copy(); uu = np.broadcast_to(u, X[..., :3].shape)
        for _ in range(n):
            k1 = M3.f(Y, uu); k2 = M3.f(Y + h / 2 * k1, uu); k3 = M3.f(Y + h / 2 * k2, uu); k4 = M3.f(Y + h * k3, uu); Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return Y
P_ = per > 0; V = float(np.prod(2 * lim)); h = (V / N) ** (1 / d); EM = EMAXF * h
def wrap(X): X = np.array(X, float); X[..., P_] = (X[..., P_] + np.pi) % (2 * np.pi) - np.pi; return X
def diff(a, b): z = a - b; z[..., P_] = (z[..., P_] + np.pi) % (2 * np.pi) - np.pi; return z
def inside(X): return (np.abs(X[..., ~P_]) <= lim[~P_]).all(-1)
def embed(X): Z = np.array(X, float); Z[..., P_] = (Z[..., P_] + np.pi) % (2 * np.pi); Z[..., ~P_] += 500.; return Z       # тор: [0,2π); остальные +500 в бокс 1000
box = np.where(P_, 2 * np.pi, 1000.)
P = rng.uniform(-lim, lim, (N, d)); P = wrap(P); tree = cKDTree(embed(P), boxsize=box)
def edges(P, tree, src, EM, J=None):
    """все кандидаты (i, y, t, dist) с dist ≤ EM; i — из src."""
    X = P[src]; out = []; spd = float(np.max(np.linalg.norm(rhs(X[:2000], US[0]), axis=-1))) + 3.; 
    for ui_, u in enumerate(US):
        dtm = TMAX / (J or max(8, int(np.ceil(TMAX * spd / (h / 2)))))
        J_ = int(round(TMAX / dtm)); S = X.copy(); ok = np.ones(len(X), bool); I = np.arange(len(X))
        for j in range(J_ + 1):
            if j > 0: S = adv(S, u, np.full(len(S), dtm)) if env_name == 'di4' or True else S; S = wrap(S)
            ok &= inside(S); Fj = rhs(S, u); s_ = np.linalg.norm(Fj, axis=-1).max() * dtm
            idx = np.nonzero(ok)[0]
            if not len(idx): break
            dist, nb = tree.query(embed(S[idx]), k=8, distance_upper_bound=EM + s_ / 2, workers=WK)
            for c in range(8):
                m = np.isfinite(dist[:, c]); 
                if not m.any(): break
                ii = idx[m]; y = nb[m, c]; r = diff(P[y], S[ii]); F = Fj[ii]; ds = (r * F).sum(-1) / np.maximum((F * F).sum(-1), 1e-12); ds = np.clip(ds, max(-dtm, -j * dtm), min(dtm, TMAX - j * dtm))   # концы дуги: не выходить за [0, TMAX] (контроль CHECK нашёл 15 лишних рёбер на t=0/TMAX)
                t = np.clip(j * dtm + ds, 0, TMAX); res = np.linalg.norm(r - F * ds[:, None], axis=-1); k_ = (res <= EM) & (src[ii] != y)
                out.append((src[ii][k_], y[k_], t[k_], res[k_], np.full(int(k_.sum()), ui_, np.int8)))
    return [np.concatenate([o[q] for o in out]) for q in range(5)]
def reduce_eps(ei, ey, et, ed, eps, ui=None):
    m = ed <= eps; ei, ey, et = ei[m], ey[m], et[m]; uu = ui[m] if ui is not None else None; key = ei.astype(np.int64) * N + ey; o = np.lexsort((et, key)); key, ei, ey, et = key[o], ei[o], ey[o], et[o]; f = np.ones(len(key), bool); f[1:] = key[1:] != key[:-1]
    return (ei[f], ey[f], et[f]) if ui is None else (ei[f], ey[f], et[f], uu[o][f])
def stats(ei, ey, N, nsrc):
    deg = np.bincount(ei, minlength=N); A = csr_matrix((np.ones(len(ei)), (ei, ey)), shape=(N, N)); nc, lab = connected_components(A, directed=True, connection='strong'); cs = np.bincount(lab)
    return dict(edges=int(len(ei)), deg_mean=float(deg[:nsrc].mean()), deg_med=float(np.median(deg[:nsrc])), no_out=float((deg[:nsrc] == 0).mean()), scc_max=float(cs.max() / N), n_scc=int(nc))
