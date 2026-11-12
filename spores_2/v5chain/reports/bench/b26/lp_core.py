import numpy as np, numba as nb
# копия lp1/lp_batch (r36/tvg_lp.py) и vit_dl (r36/tvg_dl.py)
@nb.njit(cache=True)
def lp1(X, c, out_j, out_l):
    k, dd = X.shape; m = dd + 1; nv = k + m; Tb = np.zeros((m + 1, nv + 1)); basis = np.empty(m, np.int64); M = 1e4
    for i in range(dd):
        for j in range(k): Tb[i, j] = X[j, i]
    for j in range(k): Tb[dd, j] = 1.
    Tb[dd, nv] = 1.
    for i in range(m): Tb[i, k + i] = 1.; basis[i] = k + i
    for j in range(k): Tb[m, j] = c[j]
    for i in range(m): Tb[m, k + i] = M
    for i in range(m):
        for j in range(nv + 1): Tb[m, j] -= M * Tb[i, j]
    for it in range(200):
        e = -1
        for j in range(nv):
            if Tb[m, j] < -1e-10: e = j; break
        if e < 0: break
        r = -1; best = 1e300
        for i in range(m):
            if Tb[i, e] > 1e-12:
                q = Tb[i, nv] / Tb[i, e]
                if q < best - 1e-14 or (abs(q - best) <= 1e-14 and basis[i] < basis[r]): best = q; r = i
        if r < 0: return False
        p = Tb[r, e]
        for j in range(nv + 1): Tb[r, j] /= p
        for i in range(m + 1):
            if i != r and Tb[i, e] != 0.:
                f = Tb[i, e]
                for j in range(nv + 1): Tb[i, j] -= f * Tb[r, j]
        basis[r] = e
    s = 0.
    for i in range(m):
        if basis[i] >= k:
            if Tb[i, nv] > 1e-9: return False
            out_j[i] = 0; out_l[i] = 0.
        else: out_j[i] = basis[i]; out_l[i] = max(Tb[i, nv], 0.); s += out_l[i]
    if s <= 0: return False
    for i in range(m): out_l[i] /= s
    return True

@nb.njit(parallel=True, cache=True)
def lp_batch(NB_, Qs):
    n, k, dd = NB_.shape; m = dd + 1; J = np.zeros((n, m), np.int64); L = np.zeros((n, m)); OK = np.zeros(n, np.bool_)
    for t in nb.prange(n):
        X = NB_[t] - Qs[t]; c = np.empty(k)
        cm = 0.
        for j in range(k):
            s = 0.
            for i in range(dd): s += X[j, i] * X[j, i]
            c[j] = s; cm = max(cm, s)
        for j in range(k): c[j] /= (cm + 1e-300)
        jj = np.zeros(m, np.int64); ll = np.zeros(m); OK[t] = lp1(X, c, jj, ll); J[t] = jj; L[t] = ll
    return J, L, OK


@nb.njit(parallel=True)
def vit_dl(J, LAM, OK, GL, ING, nP, tau, nit, big):
    W = np.full(nP, big); L_ = OK.shape[1]; K_ = J.shape[1]
    for i in range(nP):
        if ING[i]: W[i] = 0.
    dd = 0.; it = 0
    for it in range(nit):
        Wn = np.empty(nP)
        for i in nb.prange(nP):
            if ING[i]: Wn[i] = 0.; continue
            best = big
            for a in range(L_):
                if not OK[i, a]: continue
                if GL[i, a] > 0: c = GL[i, a]
                else:
                    r = i * L_ + a; s = 0.; bad = False
                    for k in range(K_):
                        w = W[J[r, k]]
                        if w >= big and LAM[r, k] > 1e-12: bad = True; break
                        s += LAM[r, k] * w
                    if bad: continue
                    c = tau[a] + s
                if c < best: best = c
            Wn[i] = best
        dd = np.max(np.abs(Wn - W)); W = Wn
        if dd < 1e-5: break
    return W, it, dd



@nb.njit(parallel=True)
def vit_dx(J, LAM, EXT, OK, GL, ING, nP, tau, nit, big):
    """vit_dl + добавка EXT[r] к цене ребра (фолбэк «ближайший + KV·d» на рёбрах-дырах)"""
    W = np.full(nP, big); L_ = OK.shape[1]; K_ = J.shape[1]
    for i in range(nP):
        if ING[i]: W[i] = 0.
    dd = 0.; it = 0
    for it in range(nit):
        Wn = np.empty(nP)
        for i in nb.prange(nP):
            if ING[i]: Wn[i] = 0.; continue
            best = big
            for a in range(L_):
                if not OK[i, a]: continue
                if GL[i, a] > 0: c = GL[i, a]
                else:
                    r = i * L_ + a; s = 0.; bad = False
                    for k in range(K_):
                        w = W[J[r, k]]
                        if w >= big and LAM[r, k] > 1e-12: bad = True; break
                        s += LAM[r, k] * w
                    if bad: continue
                    c = tau[a] + s + EXT[r]
                if c < best: best = c
            Wn[i] = best
        dd = np.max(np.abs(Wn - W)); W = Wn
        if dd < 1e-5: break
    return W, it, dd
