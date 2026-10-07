"""research-21: solve по ведрам V (упорядоченная коррекция меток, Dial/Δ-stepping) против Якоби в grow_cells2d.
Гипотеза: Якоби/ГЗ делают ≈ Vmax/DTN проходов по ВСЕМ рёбрам (фронт V идёт на шаг DTN за проход, b4/w23);
если обрабатывать узлы в порядке V (ведро ширины D), каждое ребро считается ~1–3 раза ⇒ работа ~E·k вместо E·n_it.
Точность: коррекция меток (изменённый узел снова в очереди) — та же неподвижная точка, что у Якоби (оператор монотонный).
Запуск (aida, из experiments/pend): env <конфиг> python3 ~/spore_v5/r18/solve_bucket.py"""
import numpy as np, os, sys, time, json
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.getcwd(), '../../src/cells7')); import grow_cells2d as G

def edges(s):
    """те же рёбра, что в Atlas.solve: (узел, стоимость, вершины K=4, веса) + V0"""
    E = []; Vg = np.full(s.N, np.inf)
    HT = float(os.environ.get('SBTAU', 0)); CC = []                                       # research-21 SBTAU=h: медленный поток — шаг τ = max(DTN, h/|f|) ≤ 1 с (конец уходит из своей ячейки)
    for u in G.US:
        Vg = np.minimum(Vg, s.tgoal(s.P, u))
        if HT > 0:
            tau = np.clip(HT / np.maximum(np.linalg.norm(G.f(s.P, u), axis=1), 1e-9), G.DTN, 1.); n_ = 8; y = s.P.copy(); h_ = (tau / n_)[:, None]
            for _ in range(n_):
                k1 = G.f(y, u); k2 = G.f(y + h_ / 2 * k1, u); k3 = G.f(y + h_ / 2 * k2, u); k4 = G.f(y + h_ * k3, u); y = y + h_ / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            e = s.stencils(y); E.append(e); CC.append(tau[e[0]])
        else: e = s.stencils(G.step(s.P, u)); E.append(e); CC.append(np.full(len(e[0]), G.DTN))
    I = np.concatenate([e[0] for e in E]); IDX = np.concatenate([e[1] for e in E]); W = np.concatenate([e[2] for e in E]); C = np.concatenate(CC)
    if int(os.environ.get('SBCAUS', 0)):                                                   # research-21: стенсил СВОЕГО u в СВОЮ клетку (не последняя строка) — лишний: столбец клетки = траектория u,
        own = np.empty(s.N, np.int64); last = np.zeros(s.N, bool); lay = np.empty(s.N, np.int64)   # точный переход (i,j)→(i+1,j) уже есть ребром NFEDGE; этот стенсил даёт петли с весом ~1
        for ci, c in enumerate(s.cells):
            nt, m = c.G.shape[:2]; own[c.o:c.o + nt * m] = ci; last[c.o + (nt - 1) * m:c.o + nt * m] = True; lay[c.o:c.o + nt * m] = list(G.US).index(c.u)
        ul = np.concatenate([np.full(len(e[0]), k) for k, e in enumerate(E)]); hasc = np.array([getattr(c, 'TT', None) is not None for c in s.cells])
        drop = (ul == lay[I]) & (own[IDX[:, 0]] == own[I]) & ~last[I] & hasc[own[I]] & bool(G.NFEDGE)
        if int(os.environ.get('SBLAY', 0)): drop |= lay[IDX[:, 0]] != ul   # research-21 (мысль пользователя «узел — часть траектории»): переключение на u2 — V только из слоя u2
        print('SBCAUS: убрано стенсилов своего u в свою клетку', int(drop.sum()), 'из', len(I), flush=True)
        I, IDX, W, C = I[~drop], IDX[~drop], W[~drop], C[~drop]
        sl = (IDX == I[:, None]).any(1) & (W.max(1) > 1e-6); n_ = I[sl]; hm = np.array([getattr(c, 'nxt', None) is not None for c in s.cells])
        print('петли на себя после SBCAUS:', int(sl.sum()), '| последняя строка', int(last[n_].sum()), '| клетка без TT', int((~hasc[own[n_]]).sum()), '| клетка с MROW-продолжением', int(hm[own[n_]].sum()),
              '| свой слой', int((ul[~drop][sl] == lay[n_]).sum()), '| вес петли мед. %.2f' % float(np.median((W[sl] * (IDX[sl] == n_[:, None])).sum(1))), flush=True)
    ea, eb, ec = [], [], []
    for c in s.cells:
        if getattr(c, 'TT', None) is None or not G.NFEDGE: continue
        nt, m = c.G.shape[:2]; a = c.o + np.arange((nt - 1) * m); ea.append(a); eb.append(a + m); ec.append(np.maximum(np.diff(c.TT, axis=0).ravel(), 1e-6))
    za, zb, z2 = [], [], []
    for c in s.cells:
        nx = getattr(c, 'nxt', None)
        if nx is None: continue
        ntA, mA = c.G.shape[:2]
        for j in range(mA):
            jf = j * c.stc; q_ = jf // nx.stc; w_ = (jf - q_ * nx.stc) / nx.stc
            if w_ == 0: za.append(c.o + (ntA - 1) * mA + j); zb.append(nx.o + q_)
            else: z2.append((c.o + (ntA - 1) * mA + j, nx.o + q_, w_))
    if za: ea.append(np.array(za)); eb.append(np.array(zb)); ec.append(np.zeros(len(za)))
    if ea:
        ea, eb, ec = map(np.concatenate, (ea, eb, ec)); k = len(ea)
        I = np.r_[I, ea]; C = np.r_[C, ec]; IDX = np.r_[IDX, np.c_[eb, np.zeros((k, 3), int)]]; W = np.r_[W, np.c_[np.ones(k), np.zeros((k, 3))]]
    if z2:
        z2 = np.array(z2); a_, b_, w_ = z2[:, 0].astype(int), z2[:, 1].astype(int), z2[:, 2]; k = len(a_)
        I = np.r_[I, a_]; C = np.r_[C, np.zeros(k)]; IDX = np.r_[IDX, np.c_[b_, b_ + 1, np.zeros((k, 2), int)]]; W = np.r_[W, np.c_[1 - w_, w_, np.zeros((k, 2))]]
    V0 = np.minimum(s.V, Vg); dead = getattr(s, 'dead', np.zeros(s.N, bool)); V0[dead] = G.BIG; V0[s.goal] = 0.
    return I, C, IDX, W, V0, dead

def solve_bucket(s, D=None, tol=None):
    tol = float(os.environ.get('SBTOL', 1e-9)) if tol is None else tol                         # порог рассылки улучшения
    t0 = time.time(); I, C, IDX, W, V, dead = edges(s); te = time.time() - t0; D = D or G.DTN; BIG = G.BIG
    SELF = int(os.environ.get('SBSELF', 0)); nself = 0
    if SELF:                                                                                # research-21: петля на себя (вершина = сам узел, вес s) — решить точно: V = (c + Σ_{j≠i} w_j V_j) / (1 − s)
        sm = (IDX == I[:, None]) & (W > 1e-6); s_ = (W * sm).sum(1); k = (s_ > 0) & (s_ < 1 - 1e-9); nself = int(k.sum())
        W = np.where(sm, 0., W); W[k] = W[k] / (1 - s_[k])[:, None]; C = C.copy(); C[k] = C[k] / (1 - s_[k]); dead_e = s_ >= 1 - 1e-9; C[dead_e] = BIG
    M = W > 1e-6; ev = np.repeat(np.arange(len(I)), 4)[M.ravel()]; vv = IDX.ravel()[M.ravel()]   # обратные рёбра: вершина → рёбра, где она с весом
    o = np.argsort(vv, kind='stable'); ev = ev[o]; ptr = np.searchsorted(vv[o], np.arange(s.N + 1))
    fixed = s.goal | dead; pend = (V < BIG / 2); nev = 0; nb = 0; th = 0.
    vmax = float(V[V < BIG / 2].max()) if (V < BIG / 2).any() else 1.; bar = tqdm(total=round(vmax + 30, 1), desc='solve вёдра (V)', unit='с V', mininterval=float(os.environ.get('TQDM_MI', 10)), leave=False)
    while pend.any():
        th = max(th, V[pend].min()); thr = th + D; nb += 1; bar.n = round(min(th, bar.total), 1); bar.set_postfix(вёдер=nb, на_ребро='%.1f' % (nev / max(len(I), 1))); bar.refresh()
        bt = np.flatnonzero(pend & (V < thr))
        while len(bt):
            pend[bt] = False; lo, hi = ptr[bt], ptr[bt + 1]; n_ = hi - lo; tot = n_.sum()
            if not tot: break
            e = ev[np.repeat(lo - np.r_[0, np.cumsum(n_)[:-1]], n_) + np.arange(tot)]; e = np.unique(e); nev += len(e)
            val = C[e] + G.Atlas.interp(W[e], V[IDX[e]], not G.JAG)
            nd = I[e]; oo = np.argsort(nd, kind='stable'); nd, val = nd[oo], val[oo]; st = np.flatnonzero(np.r_[True, nd[1:] != nd[:-1]])
            un = nd[st]; mv = np.minimum.reduceat(val, st); imp = (mv < V[un] - tol) & ~fixed[un]
            un, mv = un[imp], mv[imp]; V[un] = mv; pend[un] = True; bt = un[mv < thr]
        th = thr
    bar.close(); s.V_b = V; return dict(nself=nself, t_edges=round(te, 2), t_total=round(time.time() - t0, 2), buckets=nb, edges=len(I), evals=int(nev), evals_per_edge=round(nev / len(I), 2))

if __name__ == '__main__':
    t0 = time.time(); A = G.Atlas(); tb = time.time() - t0
    t1 = time.time(); A.solve(); tj = time.time() - t1; Vj = A.V.copy()
    out = dict(cells=len(A.cells), nodes=int(A.N), t_build=round(tb, 1), jacobi=dict(t=round(tj, 2), iters=int(A.n_it), evals=int(A.n_it * A.edges)))
    for D in [float(x) for x in os.environ.get('BD', '%g' % G.DTN).split(',')]:
        r = solve_bucket(A, D); fin = (Vj < G.BIG / 2) | (A.V_b < G.BIG / 2); dv = np.abs(A.V_b - Vj)[fin]
        r.update(D=D, dV_max=float(dv.max()) if len(dv) else 0., dV_mean=float(dv.mean()) if len(dv) else 0., big_j=int((Vj >= G.BIG / 2).sum()), big_b=int((A.V_b >= G.BIG / 2).sum())); out['bucket_%g' % D] = r
    print(json.dumps(out), flush=True)
