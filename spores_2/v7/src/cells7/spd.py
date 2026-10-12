"""SPD1 (b9): V на новую цель без пересборки слоёв и стенсилов (growN, manip/любая N-мерная система).
Рёбра атласа (I, IDX, W: стенсилы SONE/SBCAUS) от цели НЕ зависят (профиль 2026-10-12: solve g300so 361 с = стенсилы ~270 + Якоби ~75 + tgoal 13) —
строятся один раз (growN.Atlas.solve с SAVEE=префикс, либо edges_from_atlas), дальше смена цели = tgoal у границы цели + GPU-Якоби.
    S = GoalSolver(A, prefix)             # рёбра на GPU
    S.retarget(gc, tol=1e-9, lcut=None)   # gc — центр цели (N чисел, как growN.GC); lcut — остановка, когда V<lcut устоялась
Результат: A.V, G.GC = gc (rollout/ingoal работают с новой целью). Якоби — тот же оператор, что в growN.solve (SOLVEGPU, PESS, NOLATCH)."""
import time, numpy as np, torch
import __main__ as _m
import growN as G


class GoalSolver:
    def __init__(s, A, prefix, dev='cuda', compact=False, sc_dtype='float32'):
        """compact=True (SPD4): стенсил = (владелец int32, base int32, 4 координаты sc в ячейке); вершины base + смещения, вес = Π(sc | 1−sc) — считаются на лету.
        Рёбра не-аффинные (столбцовые SBCAUS) хранятся как раньше. sc_dtype: float32 (≡ W до 1e-7) или float16 (≈7e-4)."""
        s.A, s.dev, s.compact = A, dev, compact
        I = np.load(prefix + '_I.npy', mmap_mode='r'); IDX = np.load(prefix + '_IDX.npy', mmap_mode='r'); W = np.load(prefix + '_W.npy', mmap_mode='r'); s.ne = len(I); s.parts = []   # части: ('full', I, X, W) | ('cmp', I, B, SC)
        if not compact:
            s.parts.append(('full', torch.as_tensor(np.asarray(I).astype(np.int64), device=dev), torch.as_tensor(np.asarray(IDX), device=dev), torch.as_tensor(np.asarray(W), device=dev)))
        else:
            VOFF = G.VOFF; off = VOFF @ np.array([G.M ** (G.m_ - k) for k in range(G.N)]); ch = 1 << 21; Ia, Ba, Sa, If, Xf, Wf = [], [], [], [], [], []
            for a in range(0, s.ne, ch):
                x = np.asarray(IDX[a:a + ch]).astype(np.int64); w = np.asarray(W[a:a + ch]); i_ = np.asarray(I[a:a + ch]); aff = (x == x[:, :1] + off[None]).all(1)
                sc = np.stack([w[:, VOFF[:, k] == 1].sum(1, dtype=np.float64) for k in range(G.N)], 1)
                Ia.append(i_[aff].astype(np.int32)); Ba.append(x[aff, 0].astype(np.int32)); Sa.append(sc[aff].astype(sc_dtype)); If.append(i_[~aff]); Xf.append(x[~aff].astype(np.int32)); Wf.append(w[~aff])
            s.parts.append(('cmp', torch.as_tensor(np.concatenate(Ia), device=dev), torch.as_tensor(np.concatenate(Ba), device=dev), torch.as_tensor(np.concatenate(Sa), device=dev)))
            s.parts.append(('full', torch.as_tensor(np.concatenate(If).astype(np.int64), device=dev), torch.as_tensor(np.concatenate(Xf), device=dev), torch.as_tensor(np.concatenate(Wf), device=dev)))
        s.hasE = torch.zeros(A.N, dtype=torch.bool, device=dev)
        for p in s.parts: s.hasE[p[1].long()] = True
        s.vo = torch.as_tensor(G.VOFF, device=dev); s.off = torch.as_tensor(G.VOFF @ np.array([G.M ** (G.m_ - k) for k in range(G.N)]), device=dev)
        s.P = A.P; s.fmax = None

    def save_compact(s, out):
        """SPD4 (b10): компакт-кэш рёбер на диск — части parts как есть (cmp: I int32, base int32, sc; full: столбцовые рёбра) → <out>_C<k>_<имя>.npy (~0.9 ГБ против 4 ГБ у E_*_{I,IDX,W})."""
        assert s.compact, 'save_compact только для compact=True'
        for k, p in enumerate(s.parts):
            for nm, t in zip(('I', 'B', 'S'), p[1:]): np.save('%s_C%d_%s.npy' % (out, k, nm), t.cpu().numpy())
        np.save(out + '_Ckinds.npy', np.array([p[0] for p in s.parts])); np.save(out + '_Cne.npy', np.array([s.ne]))
    @classmethod
    def load_compact(cls, A, out, dev='cuda'):
        """GoalSolver из компакт-кэша (save_compact): исходные E_*_{I,IDX,W} не читаются."""
        s = cls.__new__(cls); s.A, s.dev, s.compact = A, dev, True; s.ne = int(np.load(out + '_Cne.npy')[0]); kinds = np.load(out + '_Ckinds.npy'); s.parts = []
        for k, kd in enumerate(kinds):
            I, B, S = [torch.as_tensor(np.load('%s_C%d_%s.npy' % (out, k, nm)), device=dev) for nm in ('I', 'B', 'S')]; s.parts.append((str(kd), I, B, S))
        s.hasE = torch.zeros(A.N, dtype=torch.bool, device=dev)
        for p in s.parts: s.hasE[p[1].long()] = True
        s.vo = torch.as_tensor(G.VOFF, device=dev); s.off = torch.as_tensor(G.VOFF @ np.array([G.M ** (G.m_ - k) for k in range(G.N)]), device=dev); s.P = A.P; s.fmax = None
        return s

    def nbytes(s): return sum(t.numel() * t.element_size() for p in s.parts for t in p[1:])

    def tgoal_min(s, gc):
        """Vg = min по u времени входа в цель за шаг DTN (growN.Atlas.tgoal), считается только у границы цели."""
        A = s.A; G.GC = np.asarray(gc, float); d = np.abs(G.wrapg(A.P)) - G.RHOV; dist = (np.maximum(d, 0.) ** 2).sum(-1) ** .5
        Vg = np.full(A.N, np.inf); nn = np.zeros(A.N, bool)
        for u in G.US:                                                                                          # узел может войти в цель за DTN, только если до коробки ближе 3·DTN·|f| (локальная скорость)
            fu = np.linalg.norm(G.f(A.P, u), axis=-1); idx = np.flatnonzero(dist <= 3. * G.DTN * fu); nn[idx] = True
            if len(idx): Vg[idx] = np.minimum(Vg[idx], A.tgoal(A.P[idx], u))
        return Vg, int(nn.sum())

    def retarget(s, gc, tol=1e-9, lcut=None, maxit=20000, log=print, ch=1 << 22, fp32=False):
        A, dev = s.A, s.dev; t0 = time.time(); Vg, nn = s.tgoal_min(gc); A.goal = G.ingoal(A.P); t1 = time.time()
        V = np.full(A.N, G.BIG); V[A.goal] = 0.; V = np.minimum(V, Vg)
        dt = torch.float32 if fp32 else torch.float64; Vt = torch.as_tensor(V, device=dev).to(dt); Vgt = torch.as_tensor(Vg, device=dev).to(dt); goal = torch.as_tensor(A.goal, device=dev); pe = G.PESS if G.PESS >= 0 else 0.
        n = 0; d = 1.
        while n < maxit:
            new = Vt.clone()
            if G.NOLATCH: new[s.hasE] = Vgt[s.hasE]
            for kind, gI, gX, gW in s.parts:
                for a in range(0, len(gI), ch):
                    if kind == 'full': vi = Vt[gX[a:a + ch].long()]; w_ = gW[a:a + ch].to(dt)
                    else:
                        sc = gW[a:a + ch].to(dt); vi = Vt[gX[a:a + ch].long()[:, None] + s.off[None]]; w_ = torch.ones(len(sc), 2 ** G.N, dtype=dt, device=dev)
                        for k in range(G.N): w_ = w_ * torch.where(s.vo[None, :, k] == 1, sc[:, k:k + 1], 1 - sc[:, k:k + 1])
                    ok = vi < G.BIG / 2; w = w_ * ok; sm = w.sum(1); mx = torch.where(ok, vi, torch.full_like(vi, -float('inf'))).max(1).values
                    v = (w * torch.where(ok, vi, torch.zeros_like(vi))).sum(1) + torch.nan_to_num((1 - sm) * (mx + pe), nan=0., posinf=0., neginf=0.); v[sm < G.WTHR] = G.BIG
                    new.scatter_reduce_(0, gI[a:a + ch].long(), G.DTN + v, 'amin')
            new[goal] = 0.; df = (new - Vt).abs()
            if lcut is not None: df = torch.where(new < lcut, df, torch.zeros_like(df))
            d = float(df.max()); Vt = new; n += 1
            if d < tol: break
        A.V = Vt.double().cpu().numpy(); A.n_it = n; torch.cuda.synchronize(); t2 = time.time()
        log('retarget: gc %s | у цели %d узлов | tgoal %.1f с | Якоби %d проходов %.1f с | всего %.1f с' % (np.round(gc, 3).tolist(), nn, t1 - t0, n, t2 - t1, t2 - t0)); return A


def _stencils_on(idx, O, Y):
    """как Atlas.stencils, но по произвольному HexIdx idx и базам узлов O его клеток (SPD1b: запрос только по новым клеткам)."""
    Y = G.wrapy(Y); pi, hid, sc = G._pquery(idx, Y)
    if not len(pi): return np.zeros(0, np.int64), np.zeros((0, 2 ** G.N), np.int64), np.zeros((0, 2 ** G.N), np.float32)
    stv = np.array([G.M ** (G.m_ - k) for k in range(G.N)]); off = G.VOFF @ stv; n_ = len(pi); IDX = np.empty((n_, 2 ** G.N), np.int64); W = np.empty((n_, 2 ** G.N), np.float32); CH = 1 << 21
    for a in range(0, n_, CH):
        b = min(a + CH, n_); ii = idx.I[hid[a:b]].astype(np.int64); base = O[ii[:, 0]] + (ii[:, 1:] * stv).sum(1); IDX[a:b] = base[:, None] + off[None]
        sb = sc[a:b]; W[a:b] = np.prod(np.where(G.VOFF[None] == 1, sb[:, None, :], 1 - sb[:, None, :]), 2)
    return pi, IDX, W


def _dedup_one(a, b, c_, lay):
    """по одному (самому центральному = max наибольшего веса) стенсилу на (узел, слой приземления) — как SONE."""
    key = a.astype(np.int64) * 16 + lay; o2 = np.lexsort((c_.max(1), key)); key = key[o2]; f = o2[np.r_[True, key[1:] != key[:-1]]] if len(o2) else o2
    return a[f], b[f], c_[f]


def grow_layers(gc, maxc=500, seed=777):
    """рост новых слоёв вокруг цели gc (fork-пул 4 процесса). Вызывать ДО инициализации CUDA (fork после torch.cuda не работает) или в отдельном процессе."""
    from multiprocessing import Pool
    G.GC = np.asarray(gc, float); mc0 = G.MAXC; G.MAXC = maxc
    with Pool(len(G.US)) as pool: lays = pool.map(G._layer, [(u, seed + k) for k, u in enumerate(G.US)])
    G.MAXC = mc0; return lays


def reseed(S, gc, lays, log=print, t_grow=0.):
    """SPD1b (b9): посев цели gc — GS-семена вокруг неё, новые клетки (по слою на каждое u, до maxc клеток) дописываются в готовый атлас A и кэш рёбер S
    (старые клетки/узлы/рёбра не пересчитываются; к ним добавляются рёбра старых узлов в новые клетки и рёбра новых узлов в атлас). Потом S.retarget(gc, …)."""
    A, t0 = S.A, time.time(); A.stencils(np.zeros((0, G.N))); cum_old = np.cumsum([len(l) for l in A.layers]); G.GC = np.asarray(gc, float); t1 = time.time(); nl0 = len(A.layers); N0 = A.N; mm = G.M ** G.m_
    new = [c for l in lays for c in l]; n_ = N0; ctrl = []
    for ui, l in enumerate(lays):
        for c in l: c.o = n_; n_ += c.G.shape[0] * mm; ctrl.append(ui)
    ctrl = np.array(ctrl); Pn = np.concatenate([c.G.reshape(-1, G.N) for c in new]); nN = len(Pn); lay_new = np.concatenate([[ui] * (c.G.shape[0] * mm) for ui, c in zip(ctrl, new)])
    # индекс новых клеток отдельно (для старых узлов) и общий qx (для агента и новых узлов)
    idxN = G.HexIdx(); [idxN.add(c) for c in new]; ON = np.array([c.o for c in new])
    if hasattr(A, 'qx'):
        for c in new: A.qx.add(c)
        A.O = np.concatenate([A.O, ON])
    A.layers += lays; A.cells += new; A.P = np.concatenate([A.P, Pn]); A.N = n_; A.V = np.concatenate([A.V, np.full(nN, G.BIG)]); A.goal = G.ingoal(A.P)
    cellof_new = np.repeat(np.arange(len(new)), [c.G.shape[0] * mm for c in new])
    Ie, Xe, We = [], [], []; idt = np.int32
    for ui, u in enumerate(G.US):
        # (а) старые узлы → новые клетки
        for s0 in range(0, N0, 400000):
            a, b, c_ = _stencils_on(idxN, ON, G.step(A.P[s0:s0 + 400000], u)); a = a + s0
            if len(a): vc = np.searchsorted(ON, b[:, 0], 'right') - 1; a, b, c_ = _dedup_one(a, b, c_, ctrl[vc]); Ie.append(a); Xe.append(b.astype(idt)); We.append(c_)
        # (б) новые узлы → весь атлас (старые + новые клетки); приземление в свою клетку при том же управлении — точным ребром по столбцу
        Y = G.step(Pn, u); a, b, c_ = [], [], []
        for s0 in range(0, nN, 400000):
            aa, bb, cc = A.stencils(Y[s0:s0 + 400000]); a.append(aa + s0); b.append(bb); c_.append(cc)
        a = np.concatenate(a); b = np.concatenate(b); c_ = np.concatenate(c_); an = a + N0
        Oall = A.O; ncold = len(Oall) - len(ON); vc = np.searchsorted(Oall, b[:, 0], 'right') - 1
        own = ncold + np.searchsorted(ON, an, 'right') - 1
        keep = np.where(lay_new[a] == ui, vc != own, True); an, b, c_, vc = an[keep], b[keep], c_[keep], vc[keep]
        lid = np.where(vc >= ncold, 4 + ctrl[np.clip(vc - ncold, 0, len(new) - 1)], np.searchsorted(cum_old, vc, 'right').clip(0, 3))
        an, b, c_ = _dedup_one(an, b, c_, lid)
        Ie.append(an); Xe.append(b.astype(idt)); We.append(c_)
    ea = np.concatenate([c.o + np.arange((c.G.shape[0] - 1) * mm) for c in new]); ex = np.zeros((len(ea), 2 ** G.N), np.float32); ex[:, 0] = 1.; eI = np.zeros((len(ea), 2 ** G.N), idt); eI[:] = (ea + mm)[:, None]
    Ie.append(ea); Xe.append(eI); We.append(ex)
    I = np.concatenate(Ie); X = np.concatenate(Xe); W = np.concatenate(We); dev = S.dev
    S.parts.append(('full', torch.as_tensor(I.astype(np.int64), device=dev), torch.as_tensor(X, device=dev), torch.as_tensor(W, device=dev)))
    hasE = torch.zeros(A.N, dtype=torch.bool, device=dev); hasE[:S.hasE.numel()] = S.hasE; hasE[S.parts[-1][1]] = True; S.hasE = hasE; S.ne += len(I)
    log('reseed: gc %s | новых клеток %s, узлов %d (было %d) | рост %.1f с, дописывание: рёбра +%d, %.1f с' % (np.round(gc, 3).tolist(), [len(l) for l in lays], nN, N0, t_grow, len(I), time.time() - t0)); return S
