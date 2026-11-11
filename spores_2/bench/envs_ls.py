"""b17: Б-LS — схема Б (пересчёт V области у дисков однопроходным label-setting, research-31 lsk2_r31.py) для агента А (LOOK=5).
Зона дисков и стенсилы строятся ОДИН раз (prep_zone, в родителе до fork); раз в k шагов refresh(): область = зона ∩ {V₀ ≤ V₀(аг), V₀(аг) − V₀ ≤ T_PR},
запрет — облако дисков m(τ)=p+wτ, раздутое κσ(τ), τ = V₀(аг) − V₀(узел); LS (pe .3) + 1 GS → A.V (V₀ вне области)."""
import os, time, heapq
import numpy as np, numba as nb

def prep_zone(G, A, C0, HR, eps=.03, smax=.25, tube=.5, extra=.3, cache=None):
    import os
    if cache and os.path.exists(cache):
        t0 = time.time(); d = np.load(cache); Z = {k: d[k] for k in d.files}; Z['nz'] = int(Z['nz']); Z['kap'] = float(Z['kap']); Z['t_prep'] = time.time() - t0; return Z
    Z = _prep_zone(G, A, C0, HR, eps, smax, tube, extra)
    if cache: np.savez(cache, **{k: v for k, v in Z.items()})
    return Z

def prep_rev(Z):
    t0 = time.time(); Z['cnt'], Z['rev_e'], Z['rev_j'] = build_rev(Z['loc'], Z['X'], Z['W'], Z['nz']); Z['ownz'] = np.repeat(Z['nd'], np.diff(Z['st'])).astype(np.int32)
    E = len(Z['X']); Z['bs'], Z['bsm'], Z['bmx'] = np.empty(E), np.empty(E), np.empty(E); Z['t_rev'] = time.time() - t0; return Z

def _prep_zone(G, A, C0, HR, eps=.03, smax=.25, tube=.5, extra=.3):
    t0 = time.time(); kap = np.sqrt(-2 * np.log(eps)); P = A.P
    import os
    if int(os.environ.get('ZFULL', 0)): zl = np.arange(len(P))   # b17: зона = весь атлас (оболочка без геометрии)
    else: dist = np.linalg.norm(P[:, None, :2] - C0[None], axis=2).min(1); zl = np.flatnonzero(dist <= HR + kap * smax + extra + tube)
    Pz = P[zl]; nz = len(zl)
    Is, Xs, Ws = [], [], []; Vg = np.full(nz, np.inf)
    for u in G.US:
        Vg = np.minimum(Vg, A.tgoal(Pz, u)); a, b, c = A.stencils(G.step(Pz, u)); Is.append(a); Xs.append(b.astype(np.int32)); Ws.append(c)
    I = np.concatenate(Is); X = np.concatenate(Xs); W = np.concatenate(Ws).astype(np.float32 if int(os.environ.get('ZFULL', 0)) else np.float64); del Is, Xs, Ws
    o = np.argsort(I, kind='stable'); I, X, W = I[o], X[o], W[o]; del o
    st = np.r_[0, np.flatnonzero(I[1:] != I[:-1]) + 1, len(I)].astype(np.int64); nd = I[st[:-1]].astype(np.int64)
    loc = np.full(len(P), -1, np.int64); loc[zl] = np.arange(nz)
    Z = dict(zl=zl, Pz=Pz, nz=nz, X=X, W=W, nd=nd, st=st, Vg_n=np.where(np.isfinite(Vg[nd]), Vg[nd], G.BIG), loc=loc, goal=np.asarray(A.goal)[zl].astype(bool), V0=A.V.copy(), V0z=A.V[zl].copy(), kap=kap, t_prep=time.time() - t0)
    return Z

@nb.njit(cache=True)
def build5(V0, loc, X, W, nd, st, inreg, BIG):
    """срезы рёбер области + обратный индекс области (вершина → рёбра) + суммы по вершинам вне области"""
    nz = len(inreg); nk = len(nd); K = X.shape[1]
    eo = np.full(nk, -1, np.int64); ne = 0
    for k in range(nk):
        if inreg[nd[k]]:
            eo[k] = ne; ne += st[k + 1] - st[k]
    s = np.zeros(ne); sm = np.zeros(ne); mx = np.full(ne, -1e30); ownz = np.zeros(ne, np.int64); cnt = np.zeros(nz + 1, np.int64)
    for k in range(nk):
        if eo[k] < 0: continue
        for e in range(st[k], st[k + 1]):
            ce = eo[k] + e - st[k]; ownz[ce] = nd[k]
            for j in range(K):
                w = W[e, j]
                if w <= 1e-12: continue
                g = X[e, j]; l = loc[g]
                if l >= 0 and inreg[l]: cnt[l + 1] += 1
                else:
                    v = V0[g]
                    if v < BIG / 2:
                        s[ce] += w * v; sm[ce] += w
                        if v > mx[ce]: mx[ce] = v
    for i in range(nz): cnt[i + 1] += cnt[i]
    tot = cnt[nz]; rev_e = np.empty(tot, np.int64); rev_w = np.empty(tot); pos = cnt[:nz].copy()
    for k in range(nk):
        if eo[k] < 0: continue
        for e in range(st[k], st[k + 1]):
            ce = eo[k] + e - st[k]
            for j in range(K):
                w = W[e, j]
                if w <= 1e-12: continue
                l = loc[X[e, j]]
                if l >= 0 and inreg[l]:
                    rev_e[pos[l]] = ce; rev_w[pos[l]] = w; pos[l] += 1
    return eo, s, sm, mx, ownz, cnt, rev_e, rev_w

@nb.njit(cache=True)
def run5(eo, s, sm, mx, ownz, cnt, rev_e, rev_w, nd, st, Vg_n, inreg, forb, goal, BIG, DTN, WTHR, pe):
    nz = len(inreg); nk = len(nd)
    tent = np.full(nz, BIG); acc = np.zeros(nz, np.bool_)
    for k in range(nk):
        if eo[k] < 0: continue
        i = nd[k]; t = Vg_n[k]
        for e in range(st[k], st[k + 1]):
            ce = eo[k] + e - st[k]
            if sm[ce] >= WTHR:
                val = DTN + s[ce] + (1 - sm[ce]) * (mx[ce] + pe)
                if val < t: t = val
        tent[i] = t
    h = [(0.0, np.int64(0))]; h.pop()
    for i in range(nz):
        if not inreg[i]: continue
        if goal[i]: tent[i] = 0.
        if not forb[i] and tent[i] < BIG / 2: h.append((tent[i], np.int64(i)))
    heapq.heapify(h); npop = 0
    while len(h) > 0:
        t, i = heapq.heappop(h)
        if acc[i] or t > tent[i]: continue
        acc[i] = True; npop += 1
        for r in range(cnt[i], cnt[i + 1]):
            ce = rev_e[r]; sm_ = rev_w[r]; s[ce] += sm_ * t; sm[ce] += sm_
            if t > mx[ce]: mx[ce] = t
            n = ownz[ce]
            if acc[n] or forb[n] or goal[n] or sm[ce] < WTHR: continue
            val = DTN + s[ce] + (1 - sm[ce]) * (mx[ce] + pe)
            if val < tent[n]:
                tent[n] = val; heapq.heappush(h, (val, n))
    out = np.full(nz, BIG)
    for i in range(nz):
        if acc[i]: out[i] = tent[i]
    return out, npop

@nb.njit(cache=True)
def build_rev(loc, X, W, nz):
    """b17: обратный индекс ВСЕЙ зоны, один раз: вершина (зон.-лок.) → (глобальное ребро e, позиция j в стенсиле)"""
    E, K = X.shape; cnt = np.zeros(nz + 1, np.int64)
    for e in range(E):
        for j in range(K):
            if W[e, j] <= 1e-12: continue
            l = loc[X[e, j]]
            if l >= 0: cnt[l + 1] += 1
    for i in range(nz): cnt[i + 1] += cnt[i]
    rev_e = np.empty(cnt[nz], np.int32); rev_j = np.empty(cnt[nz], np.int8); pos = cnt[:nz].copy()
    for e in range(E):
        for j in range(K):
            if W[e, j] <= 1e-12: continue
            l = loc[X[e, j]]
            if l >= 0:
                rev_e[pos[l]] = e; rev_j[pos[l]] = j; pos[l] += 1
    return cnt, rev_e, rev_j

@nb.njit(cache=True)
def run6(V0, loc, X, W, nd, st, ownz, cnt, rev_e, rev_j, Vg_n, inreg, forb, goal, s, sm, mx, BIG, DTN, WTHR, pe):
    """LS по готовому обратному индексу зоны; область = флаг inreg (вершины вне области — V₀), s/sm/mx — рабочие буферы длины E (инициализируются только рёбра области)"""
    nz = len(inreg); nk = len(nd); K = X.shape[1]
    tent = np.full(nz, BIG); acc = np.zeros(nz, np.bool_); nact = 0
    for k in range(nk):
        i = nd[k]
        if not inreg[i]: continue
        nact += 1; t = Vg_n[k]
        for e in range(st[k], st[k + 1]):
            s[e] = 0.; sm[e] = 0.; mx[e] = -1e30
            for j in range(K):
                w = W[e, j]
                if w <= 1e-12: continue
                g = X[e, j]; l = loc[g]
                if l >= 0 and inreg[l]: continue
                v = V0[g]
                if v < BIG / 2:
                    s[e] += w * v; sm[e] += w
                    if v > mx[e]: mx[e] = v
            if sm[e] >= WTHR:
                val = DTN + s[e] + (1 - sm[e]) * (mx[e] + pe)
                if val < t: t = val
        tent[i] = t
    h = [(0.0, np.int64(0))]; h.pop()
    for i in range(nz):
        if not inreg[i]: continue
        if goal[i]: tent[i] = 0.
        if not forb[i] and tent[i] < BIG / 2: h.append((tent[i], np.int64(i)))
    heapq.heapify(h); npop = 0
    while len(h) > 0:
        t, i = heapq.heappop(h)
        if acc[i] or t > tent[i]: continue
        acc[i] = True; npop += 1
        for r in range(cnt[i], cnt[i + 1]):
            e = np.int64(rev_e[r]); n = np.int64(ownz[e])
            if not inreg[n]: continue
            w = W[e, rev_j[r]]; s[e] += w * t; sm[e] += w
            if t > mx[e]: mx[e] = t
            if acc[n] or forb[n] or goal[n] or sm[e] < WTHR: continue
            val = DTN + s[e] + (1 - sm[e]) * (mx[e] + pe)
            if val < tent[n]:
                tent[n] = val; heapq.heappush(h, (val, n))
    out = np.full(nz, BIG)
    for i in range(nz):
        if acc[i]: out[i] = tent[i]
    return out, npop

@nb.njit(cache=True)
def run7(V0, loc, X, W, nd, st, ownz, cnt, rev_e, rev_j, Vg_n, inreg, forb, goal, S, BIG, DTN, WTHR, pe):
    """b17 (research-31, lsk5 ls2): то же, что run6, но суммы ребра (s, sm, mx) — одним массивом S[E,3] float32 (локальность памяти, −31% LS)"""
    nz = len(inreg); nk = len(nd); K = X.shape[1]
    tent = np.full(nz, BIG); acc = np.zeros(nz, np.bool_)
    for k in range(nk):
        i = nd[k]
        if not inreg[i]: continue
        t = Vg_n[k]
        for e in range(st[k], st[k + 1]):
            a0 = 0.; a1 = 0.; a2 = -1e30
            for j in range(K):
                w = W[e, j]
                if w <= 1e-12: continue
                g = X[e, j]; l = loc[g]
                if l >= 0 and inreg[l]: continue
                v = V0[g]
                if v < BIG / 2:
                    a0 += w * v; a1 += w
                    if v > a2: a2 = v
            S[e, 0] = a0; S[e, 1] = a1; S[e, 2] = a2
            if a1 >= WTHR:
                val = DTN + a0 + (1 - a1) * (a2 + pe)
                if val < t: t = val
        tent[i] = t
    h = [(0.0, np.int64(0))]; h.pop()
    for i in range(nz):
        if not inreg[i]: continue
        if goal[i]: tent[i] = 0.
        if not forb[i] and tent[i] < BIG / 2: h.append((tent[i], np.int64(i)))
    heapq.heapify(h); npop = 0
    while len(h) > 0:
        t, i = heapq.heappop(h)
        if acc[i] or t > tent[i]: continue
        acc[i] = True; npop += 1
        for r in range(cnt[i], cnt[i + 1]):
            e = np.int64(rev_e[r]); n = np.int64(ownz[e])
            if not inreg[n]: continue
            w = W[e, rev_j[r]]; a0 = S[e, 0] + w * t; a1 = S[e, 1] + w; a2 = S[e, 2]
            if t > a2: a2 = t
            S[e, 0] = a0; S[e, 1] = a1; S[e, 2] = a2
            if acc[n] or forb[n] or goal[n] or a1 < WTHR: continue
            val = DTN + a0 + (1 - a1) * (a2 + pe)
            if val < tent[n]:
                tent[n] = val; heapq.heappush(h, (val, n))
    out = np.full(nz, BIG)
    for i in range(nz):
        if acc[i]: out[i] = tent[i]
    return out, npop

@nb.njit(cache=True)
def gs5(V, zl, X, W, nd, st, Vg_n, forb, goal, order, BIG, DTN, WTHR, pe):
    K = X.shape[1]
    for oi in range(len(order)):
        k = order[oi]; i = nd[k]
        if forb[i] or goal[i]: continue
        best = Vg_n[k]
        for e in range(st[k], st[k + 1]):
            sm = 0.; s = 0.; mx = -1e30
            for j in range(K):
                w = W[e, j]
                if w <= 1e-12: continue
                v = V[X[e, j]]
                if v < BIG / 2:
                    s += w * v; sm += w
                    if v > mx: mx = v
            if sm < WTHR: continue
            val = DTN + s + (1 - sm) * (mx + pe)
            if val < best: best = val
        V[zl[i]] = best

def refresh(Z, G, A, y, p, w, q, sig0=0., pad=.1, HR=.7, T_PR=2., smax=.25, pe=.3, ngs=1, MARG=.3, install=True, tshift=0.):
    """пересчёт A.V для агента в состоянии y. Возврат dict: n (узлов области), npop, t по фазам, va до/после"""
    pc = time.perf_counter; T = {}; t00 = t0 = pc(); cur = A.V; V0 = Z['V0']; A.V = V0; va = float(A.vstar(np.asarray(y, float)[None])[0])
    if va >= G.BIG / 2: A.V = cur; return dict(n=0)
    V0z = Z['V0z']; inreg = (V0z <= va) & (va - V0z <= T_PR); r = np.flatnonzero(inreg); T['region'] = pc() - t0; t0 = pc()
    if len(r) == 0: A.V = cur; return dict(n=0)
    tau = va - V0z[r]; tau_s = tau + tshift if os.environ.get('LSB') != '2' else tau + tshift   # LSB=1: и σ, и положение на момент использования; LSB=2: положение p+wτ без сдвига, только σ(τ+t_LS)
    sg = np.minimum(np.sqrt(sig0 ** 2 + q * tau_s ** 3 / 3), smax)
    if os.environ.get('LSB') == '2': tau = tau   # положение — без сдвига
    else: tau = tau_s
    m = p[None] + w[None] * tau[:, None, None]; d = np.linalg.norm(Z['Pz'][r, None, :2] - m, axis=2) - HR - pad - Z['kap'] * sg[:, None]
    dm = d.min(1); forb = np.zeros(Z['nz'], bool); forb[r[dm < 0]] = True
    if MARG is not None:   # b17: область по текущему облаку (m(τ)+R+pad+κσ+MARG), без трубки хода
        r = r[dm < MARG]; inreg = np.zeros(Z['nz'], bool); inreg[r] = True
    T['forb'] = pc() - t0; t0 = pc()
    if 'rev_e' not in Z:
        Z['cnt'], Z['rev_e'], Z['rev_j'] = build_rev(Z['loc'], Z['X'], Z['W'], Z['nz']); Z['ownz'] = np.repeat(Z['nd'], np.diff(Z['st'])).astype(np.int32)
        E = len(Z['X']); Z['bs'], Z['bsm'], Z['bmx'] = np.empty(E), np.empty(E), np.empty(E); T['rev_once'] = pc() - t0; t0 = pc()
    if os.environ.get('LS32') == '1':
        if 'bS' not in Z: Z['bS'] = np.empty((len(Z['X']), 3), np.float32)
        Vr, npop = run7(V0, Z['loc'], Z['X'], Z['W'], Z['nd'], Z['st'], Z['ownz'], Z['cnt'], Z['rev_e'], Z['rev_j'], Z['Vg_n'], inreg, forb, Z['goal'], Z['bS'], G.BIG, G.DTN, G.WTHR, pe); T['ls'] = pc() - t0; t0 = pc()
    else: Vr, npop = run6(V0, Z['loc'], Z['X'], Z['W'], Z['nd'], Z['st'], Z['ownz'], Z['cnt'], Z['rev_e'], Z['rev_j'], Z['Vg_n'], inreg, forb, Z['goal'], Z['bs'], Z['bsm'], Z['bmx'], G.BIG, G.DTN, G.WTHR, pe); T['ls'] = pc() - t0; t0 = pc()
    V = V0.copy(); V[Z['zl'][r]] = Vr[r]; T['write'] = pc() - t0; t0 = pc()
    if ngs:
        ka = np.flatnonzero(inreg[Z['nd']]); order = ka[np.argsort(V[Z['zl'][Z['nd'][ka]]], kind='stable')]; T['gs_sort'] = pc() - t0; t0 = pc()
        for _ in range(ngs): gs5(V, Z['zl'], Z['X'], Z['W'], Z['nd'], Z['st'], Z['Vg_n'], forb, Z['goal'], order, G.BIG, G.DTN, G.WTHR, 1.)
        T['gs'] = pc() - t0
    A.V = V; va2 = float(A.vstar(np.asarray(y, float)[None])[0])
    if not install: A.V = cur
    T['tot'] = pc() - t00
    return dict(n=len(r), nf=int(forb.sum()), npop=int(npop), va=va, va2=va2, _V=V, **T)
