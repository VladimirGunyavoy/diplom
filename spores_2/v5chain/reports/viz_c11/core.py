"""c11: самодостаточный стенд «маятник» для визуализации метода (numpy/scipy; численно как tvg_*: θ̈=sinθ+u, |u|≤.3, τ=.2, цель ±.1).
Узлы/VI/Делоне-шов повторены по смыслу tvg_vt.py / tvg_dl.py (не копия кода; обратное дерево — упрощённая версия build_back)."""
import numpy as np, os
from scipy.spatial import Delaunay, cKDTree
PI = np.pi; UM = .3; ACT = np.array([-UM, 0., UM]); TAU = .2; GH = .1; OMAX = 3.5; SC = np.array([1., 1.])
def f(Z, u): return np.stack([Z[..., 1], np.sin(Z[..., 0]) + u], -1)
def rk4(Z, u, h):
    k1 = f(Z, u); k2 = f(Z + h / 2 * k1, u); k3 = f(Z + h / 2 * k2, u); k4 = f(Z + h * k3, u); return Z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def flow(Z, u, t, hmax=.02):
    n = max(1, int(np.ceil(abs(t) / hmax))); h = t / n
    for _ in range(n): Z = rk4(Z, u, h)
    return Z
def wrap(Z):
    Z = np.array(Z, float); Z[..., 0] = (Z[..., 0] + PI) % (2 * PI) - PI; return Z
def in_goal(Z): Z = wrap(Z); return (np.abs(Z) <= GH - .02).all(-1)
def okv(Z): return np.abs(np.asarray(Z)[..., 1]) <= OMAX
def edges(P, nsub=5):
    """концы дуг (n,3,2), OK (n,3), GL (n,3) — время попадания в цель по дуге (0 — не попала)"""
    n = len(P); E = np.zeros((n, 3, 2)); OK = np.zeros((n, 3), bool); GL = np.zeros((n, 3)); h = TAU / nsub
    for s, u in enumerate(ACT):
        Z = P.copy(); ok = np.ones(n, bool); hit = np.zeros(n)
        for k in range(nsub):
            Z = flow(Z, u, h); ok &= okv(Z); g = in_goal(Z) & (hit == 0); hit[g] = (k + 1) * h
        E[:, s] = Z; OK[:, s] = ok; GL[:, s] = hit
    return E, OK, GL
def arc(x, s, m=12):
    """полилиния дуги из x для действия s длиной TAU (без wrap)"""
    pts = [np.array(x, float)]
    for _ in range(m): pts.append(rk4(pts[-1], ACT[s], TAU / m))
    return np.array(pts)
def make_nodes(N=20000, seed=0, ngoal=150):
    rng = np.random.default_rng(seed); P = np.c_[rng.uniform(-PI, PI, N), rng.uniform(-OMAX, OMAX, N)]
    G = rng.uniform(-GH + .02, GH - .02, (ngoal, 2)); return np.vstack([G, P])
def emb(Z): Z = np.asarray(Z); return np.c_[np.cos(Z[..., 0]), np.sin(Z[..., 0]), Z[..., 1]]
class Interp:
    """Делоне по узлам + периодические образы θ; веса λ (как tvg_dl.Interp)"""
    def __init__(s, P, marg=.5):
        X, I = [P], [np.arange(len(P))]
        m = P[:, 0] > PI - marg; Y = P[m].copy(); Y[:, 0] -= 2 * PI; X.append(Y); I.append(np.flatnonzero(m))
        m = P[:, 0] < -PI + marg; Y = P[m].copy(); Y[:, 0] += 2 * PI; X.append(Y); I.append(np.flatnonzero(m))
        s.X = np.vstack(X); s.I = np.concatenate(I); s.tri = Delaunay(s.X)
    def weights(s, Q):
        Q = wrap(Q); sm = s.tri.find_simplex(Q); ok = sm >= 0; s0 = np.where(ok, sm, 0); Tm = s.tri.transform[s0]
        b = np.einsum('nij,nj->ni', Tm[:, :2], Q - Tm[:, 2]); lam = np.c_[b, 1 - b.sum(1)]; lam = np.clip(lam, 0, None); lam /= lam.sum(1, keepdims=True)
        return s.I[s.tri.simplices[s0]], lam, ok, s0
def vi_dl(J, LAM, OK, GL, ing, nit=400, big=1e3, keep_hist=0):
    n = len(ing); L = 3; W = np.full(n, big); W[ing] = 0; hist = [W.copy()]; Jf = J.reshape(n, L, 3); Lf = LAM.reshape(n, L, 3)
    for it in range(nit):
        Wj = W[Jf]; c = TAU + (Lf * Wj).sum(-1); c = np.where(GL > 0, GL, c); c = np.where(OK, c, big); Wn = np.minimum(c.min(1), big); Wn[ing] = 0
        dd = np.abs(Wn - W).max(); W = Wn
        if keep_hist and it % keep_hist == 0: hist.append(W.copy())
        if dd < 1e-5: break
    return W, hist, it
def vi_nn(P, E, OK, GL, ing, kv, K=9, nit=400, big=1e3, keep_hist=0):
    n = len(P); tree = cKDTree(emb(P)); dd, jj = tree.query(emb(wrap(E.reshape(-1, 2))), k=K); W = np.full(n, big); W[ing] = 0; hist = [W.copy()]
    jj = jj.reshape(n, 3, K); dd = dd.reshape(n, 3, K)
    for it in range(nit):
        m = (W[jj] + kv * dd).min(-1); c = np.where(GL > 0, GL, TAU + m); c = np.where(OK, c, big); Wn = np.minimum(c.min(1), big); Wn[ing] = 0
        d_ = np.abs(Wn - W).max(); W = Wn
        if keep_hist and it % keep_hist == 0: hist.append(W.copy())
        if d_ < 1e-5: break
    return W, hist, tree
def kv_auto(P, tree, W, k=9, q=.9):
    dd, jj = tree.query(emb(P), k=k); m = (dd > 1e-9) & (W[:, None] < 1e2) & (W[jj] < 1e2); r = (np.abs(W[:, None] - W[jj])[m] / dd[m]); return float(np.quantile(r, q))
def solve_all(N=20000, seed=0, hk=3, cache='/tmp/c11_solve.npz'):
    P = make_nodes(N, seed); E, OK, GL = edges(P); ing = in_goal(P)
    Ew = wrap(E.reshape(-1, 2)); ip = Interp(P); J, LAM, ins, _ = ip.weights(Ew); OKd = (OK & ins.reshape(-1, 3)) | (OK & (GL > 0))
    Wd, histd, itd = vi_dl(J, LAM, OKd, GL, ing, keep_hist=hk)
    kv = .05
    for _ in range(6):
        Wn, histn, tree = vi_nn(P, E, OK, GL, ing, kv, keep_hist=hk); k2 = kv_auto(P, tree, Wn)
        if abs(k2 - kv) < .05 * k2: break
        kv = k2
    return dict(P=P, E=E, OK=OK, GL=GL, ing=ing, ip=ip, Wd=Wd, histd=histd, Wn=Wn, histn=histn, tree=tree, kv=kv, itd=itd)
class Pol:
    """жадный агент: argmin_a [τ + V̂(конец дуги)]; mode 'nn' — ближайшие+KV·d, 'dl' — Делоне-барицентр"""
    def __init__(s, S, mode):
        s.S, s.mode = S, mode; s.W = S['Wn'] if mode == 'nn' else S['Wd']; s.tree = S['tree']
    def vh(s, Q):
        Q = wrap(np.atleast_2d(Q))
        if s.mode == 'nn':
            dd, jj = s.tree.query(emb(Q), k=9); return (s.W[jj] + s.S['kv'] * dd).min(1)
        J, lam, ok, _ = s.S['ip'].weights(Q); w = (lam * s.W[J]).sum(1); bad = ~ok | (s.W[J] >= 1e2).any(1)
        if bad.any():
            dd, jj = s.tree.query(emb(Q[bad]), k=9); w[bad] = (s.S['Wn'][jj] + s.S['kv'] * dd).min(1)
        return w
    def act(s, x):
        E, OK, GL = edges(np.asarray(x, float)[None]); E, OK, GL = E[0], OK[0], GL[0]; v = s.vh(E); c = np.where(GL > 0, GL, TAU + v); c = np.where(OK, c, 1e9)
        return int(np.argmin(c)), c, v, E
def rollout(S, mode, x0, tmax=40.):
    pol = Pol(S, mode); x = np.array(x0, float); tr = [x.copy()]; acts = []; info = []; t = 0.
    while t < tmax:
        s, c, v, E = pol.act(x); info.append(dict(x=x.copy(), s=s, c=c, v=v, E=E)); acts.append(s)
        done = False
        for _ in range(10):
            x = rk4(x, ACT[s], TAU / 10); t += TAU / 10
            if in_goal(x): done = True; break
        tr.append(x.copy())
        if done: break
    return np.array(tr), acts, info, t
