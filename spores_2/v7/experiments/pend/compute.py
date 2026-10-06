"""«Трейн» для 2D систем на общем ядре v5chain/reports/research/grow_cells2d.py: считает и пишет данные в <эксперимент>/data/ (картинку рисует plot.py).
Запуск из этой папки: [SYS=pend] [TMAX=1.5] [OVL=.05] ... python3 compute.py 01_имя-эксперимента"""
import time; _T0R19 = time.time()
import numpy as np, sys, os, json, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '../../src/cells7') if os.environ.get('GCORE') != 'research' else os.path.join(HERE, '../../../v5chain/reports/research'))   # GCORE=research — прототип research
import grow_cells2d as G
tag = sys.argv[1]; D = os.path.join(HERE, tag, 'data'); os.makedirs(D, exist_ok=True); t0 = time.time()
def dump(name, obj, js=True):
    tmp = os.path.join(D, name + '.tmp'); (json.dump(obj, open(tmp, 'w')) if js else pickle.dump(obj, open(tmp, 'wb'))); os.replace(tmp, os.path.join(D, name))
PAR = dict(SYS=G.SYS, US=list(G.US), TMAX=G.TMAX, RMAX=G.RMAX, OVL=G.OVL, RHO=G.RHO, ADAPT=G.ADAPT, NORM=G.NORM, DELTA=G.DELTA, SEL=G.SEL, CORE=G.CORE, JUMP=G.JUMP, JMODE=G.JMODE, JAG=G.JAG, JDIR=G.JDIR, GROW=G.GROW, OVH=G.OVH, FRAC=G.FRAC, GOALB=int(os.environ.get('GOALB', 1)), DEPTH=getattr(G, 'DEPTH', 0), M=G.M, DTN=G.DTN, XL=G.XL, WL=G.WL, PER=G.PER)
def status(stage, **kw): dump('status.json', dict(stage=stage, sec=round(time.time() - t0, 1), params=PAR, **kw))
def cells_dump(cells): dump('cells.pkl', [dict(u=float(c.u), c=c.c, r=float(c.r), n=c.n, G=c.G.astype(np.float32)) for c in cells], js=False)
done = []
def log(u, cells):
    if len(cells) % 5 == 0: cells_dump(done + cells); status('строю атлас u = %+g' % u, cells=len(done) + len(cells))
for fn in ('cells.pkl', 'value.npz', 'agent.npz', 'paths.pkl'):
    if os.path.exists(os.path.join(D, fn)): os.remove(os.path.join(D, fn))
CUT = float(os.environ.get('CUT', 1. if G.SYS == 'pend' else .5)); REFINE = int(os.environ.get('REFINE', 0)) or (1 if CUT > 0 else 0); RTOL = float(os.environ.get('RTOL', .05)); RFT = int(os.environ.get('RFT', 1)); HB = .1     # REFINE (research-14): проходы измельчения по невязке Беллмана
NX, NW = int(np.ceil(2 * G.XL / HB)), int(np.ceil(2 * G.WL / HB)); LEV = np.zeros((NX, NW), int)
def binof(Y): Y = np.atleast_2d(Y); return np.clip(((G.wrap(Y[:, 0]) + G.XL) / HB).astype(int), 0, NX - 1), np.clip(((Y[:, 1] + G.WL) / HB).astype(int), 0, NW - 1)
def limits(p, u): l = LEV[binof(p)][0]; return max(G.RMAX / 2 ** l, .02), (G.TMAX / 2 ** l if RFT else G.TMAX)
G.LIMITS = limits; hist = []
GOALB = int(os.environ.get('GOALB', 1)); GB = None                                           # GOALB=1 (research-14): край цели ±RHO — стена с самого начала (клетка не лежит поперёк края цели)
if GOALB:
    from scipy.spatial import cKDTree; e_ = np.linspace(-G.RHO, G.RHO, 41); GB = np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]]; G.BARRIER = cKDTree(GB)
def cut_rollout(A, pa, pb, NB=int(os.environ.get('CUTRN', 6))):
    """пары узлов со скачком V > CUT (проход 0): время агента из концов; скачок не подтвердился (|Ta − Tb| ≤ CUT) — шум, стены нет;
    иначе бисекция отрезка NB раз: середина — к тому концу, на чьё время похоже её время; стена — середина последнего отрезка (обрывки разных клеток сходятся в одну линию)."""
    TH = float(os.environ.get('CUTRT', 2.))                                                     # short horizon + V* of the atlas (profile research-15: long rollouts = 57% of run time)
    def T_(Y):
        lk = G.LOOK; G.LOOK = int(os.environ.get('CUTRLOOK', 0)); T, _, P_ = A.rollout(Y, tmax=TH); G.LOOK = lk   # research-16: прокатка LOOK — только итоговому агенту (в бисекции CUTR она ×4 ко времени)
        return np.where(np.isfinite(T), T, np.minimum(TH + A.vstar(P_[-1]), 1e3))
    pb = pa + np.c_[G.wrap(pb[:, 0] - pa[:, 0]), pb[:, 1] - pa[:, 1]]; Ta, Tb = T_(pa), T_(pb); real = np.abs(Ta - Tb) > CUT; a, b, ta, tb = pa[real], pb[real], Ta[real], Tb[real]
    for _ in range(NB):
        m = (a + b) / 2; tm = T_(m); la = np.abs(tm - ta) < np.abs(tm - tb); a = np.where(la[:, None], m, a); ta = np.where(la, tm, ta); b = np.where(la[:, None], b, m); tb = np.where(la, tb, tm)
    W = (a + b) / 2; W[:, 0] = G.wrap(W[:, 0]); print('CUTR: пар', len(pa), 'подтверждено', int(real.sum()), flush=True); return W
def cut_adaptive(A, K=int(os.environ.get('CUTAK', 16)), ND=8):
    """разрыв V как край изображения, прямо по узлам (любая размерность — та же схема): k ближайших; размах V > CUT → кандидат;
    у кандидата — разрез окрестности прямой через узел по ND направлениям, по плоскости с каждой стороны; скачок J = разность плоскостей в узле,
    качество q = 1 − SSE_разрез / SSE_одна; стена — где J > CUT и J·q максимально поперёк разреза; точки стены — отрезок ±h/2 вдоль разреза."""
    from scipy.spatial import cKDTree
    fin = A.V < G.BIG / 2; P = A.P[fin].astype(float); V = A.V[fin]; P[:, 0] = G.wrap(P[:, 0]); per = G.PER is not None
    Q = np.c_[(P[:, 0] + G.XL) % (2 * G.XL), P[:, 1] + G.WL + 5]; tr = cKDTree(Q, boxsize=[2 * G.XL, 2 * G.WL + 10] if per else None)
    dist, nb = tr.query(Q, K); D = P[nb] - P[:, None]; D[..., 0] = G.wrap(D[..., 0]) if per else D[..., 0]; Vn = V[nb]
    def fit(Dm, Vm, W):                                                                          # взвешенная плоскость по маске W: возвращает a (значение в узле) и SSE
        X = np.concatenate([np.ones(Dm.shape[:2] + (1,)), Dm], -1); XtX = np.einsum('nk,nki,nkj->nij', W, X, X) + 1e-9 * np.eye(3); Xty = np.einsum('nk,nki,nk->ni', W, X, Vm)
        c = np.linalg.solve(XtX, Xty[..., None])[..., 0]; r = (np.einsum('nki,ni->nk', X, c) - Vm) * W; return c[:, 0], (r * r).sum(1), c
    a1, sse1, _ = fit(D, Vn, np.ones(Vn.shape)); cand = np.flatnonzero(Vn.max(1) - Vn.min(1) > CUT)                     # размах V в окрестности > CUT (невязка RMS не годится: узел со скачком обычно один из K)
    if not len(cand): return np.zeros((0, 2))
    Dc, Vc = D[cand], Vn[cand]; best = np.full(len(cand), -1.); J = np.zeros(len(cand)); Nn = np.zeros((len(cand), 2))
    for th in np.linspace(0, np.pi, ND, endpoint=False):
        nv = np.array([np.cos(th), np.sin(th)]); sd = Dc @ nv; L = (sd < 0).astype(float); R = (sd > 0).astype(float)
        ok = (L.sum(1) >= 3) & (R.sum(1) >= 3); aL, sL, _ = fit(Dc, Vc, L); aR, sR, _ = fit(Dc, Vc, R)
        q = 1 - (sL + sR) / np.maximum(sse1[cand], 1e-12); j = np.abs(aL - aR); sc = np.where(ok & (j > CUT), j * q, -1.); m = sc > best
        best[m], J[m], Nn[m] = sc[m], j[m], nv
    keep = best > 0; ci = cand[keep]; sc = best[keep]; Nk = Nn[keep]
    if not len(ci): return np.zeros((0, 2))
    S = np.full(len(P), -1.); S[ci] = sc; Dk = D[ci]; across = np.abs(np.einsum('nkj,nj->nk', Dk, Nk)) > np.abs(np.einsum('nkj,nj->nk', Dk, np.c_[-Nk[:, 1], Nk[:, 0]]))
    nmax = sc >= np.where(across, S[nb[ci]], -1.).max(1); ci, Nk = ci[nmax], Nk[nmax]; h = np.median(dist[ci, 1:4], 1)
    T = np.c_[-Nk[:, 1], Nk[:, 0]]; ns = np.maximum(2, np.ceil(h / .005).astype(int)); pts = [P[i] + np.linspace(-h_ / 2, h_ / 2, n_)[:, None] * t for i, h_, n_, t in zip(ci, h, ns, T)]
    print('CUTA: кандидатов', len(cand), 'стена', len(ci), 'узлов', flush=True); return np.concatenate(pts)
NF_FINAL = G.NORMFRONT
for ps in range(REFINE + 1):
    if int(os.environ.get('NF0', 1)) == 0: G.NORMFRONT = NF_FINAL if ps == REFINE else 0   # research-16 NF0=0: проход 0 (поиск стен CUT) — без NORMFRONT; NF только в итоговом проходе
    status('старт' if not ps else 'проход %d' % ps, refine=hist); rng = np.random.default_rng(int(os.environ.get('SEED', 0))); A = G.Atlas.__new__(G.Atlas); A.layers = []; A.idx = []; done = []
    for u in G.US: l, ix = G.build_layer(u, rng, log); A.layers.append(l); A.idx.append(ix); done += l; cells_dump(done)
    if ps == REFINE: break
    A.finish(); A.solve(); Y, Vo = [], []
    if CUT > 0:                                                                                # CUT: середины пар соседних узлов (поперёк и вдоль) с перепадом V > CUT — барьер для прохода 1
        from scipy.spatial import cKDTree; DB = []; PA, PB = [], []
        for c in A.cells:
            g = c.G; v = A.V[c.o:c.o + len(g) * c.m].reshape(len(g), c.m)
            for a_, b_, va, vb in ((g[:, :-1], g[:, 1:], v[:, :-1], v[:, 1:]), (g[:-1], g[1:], v[:-1], v[1:])):
                k = (np.abs(va - vb) > CUT) & (va < G.BIG / 2) & (vb < G.BIG / 2); DB.append(((a_ + b_) / 2)[k]); PA.append(a_[k]); PB.append(b_[k])
        CUTG = float(os.environ.get('CUTG', 0))                                                  # CUTG = h > 0 (research-15): стена по V* (min по всем клеткам) на равномерной сетке шага h — одна согласованная стена вместо гребёнки обрывков от каждой клетки
        if CUTG > 0:
            gx_ = np.arange(-G.XL, G.XL, CUTG); gw_ = np.arange(-G.WL, G.WL + 1e-9, CUTG); GX_, GW_ = np.meshgrid(gx_, gw_, indexing='ij'); VG = A.vstar(np.c_[GX_.ravel(), GW_.ravel()]).reshape(GX_.shape); PG = np.stack([GX_, GW_], -1); DB = []
            K_ = int(os.environ.get('CUTK', 5)); per = G.PER is not None                         # скачок > CUT на отрезке K_ шагов → стена в ОДНОЙ точке — где шаг V самый крутой (тонкая стена)
            for ax_ in (0, 1):
                V_ = np.moveaxis(VG, ax_, 0); P_ = np.moveaxis(PG, ax_, 0); n_ = V_.shape[0]; wr = per and ax_ == 0; ok_ = V_ < G.BIG / 2
                idx_ = np.arange(n_) if wr else np.arange(n_ - K_)
                for i0 in idx_:
                    ii = (i0 + np.arange(K_ + 1)) % n_ if wr else i0 + np.arange(K_ + 1)
                    seg = V_[ii]; good = ok_[ii].all(0); jump = (np.abs(seg[-1] - seg[0]) > CUT) & good
                    if not jump.any(): continue
                    d1 = np.abs(np.diff(seg, axis=0)); kk = d1.argmax(0); cols = np.flatnonzero(jump)
                    a_ = P_[ii[kk[cols]], cols]; b_ = P_[ii[kk[cols] + 1], cols]
                    if wr: b_ = np.where((b_[:, :1] < a_[:, :1]), b_ + np.array([2 * G.XL, 0.]), b_)
                    DB.append((a_ + b_) / 2)
            DB = [np.unique(np.round(np.concatenate(DB), 6), axis=0)] if DB else []
        if int(os.environ.get('CUTR', 0)): DB = [cut_rollout(A, np.concatenate(PA), np.concatenate(PB))]   # CUTR=1 (research-15): место разрыва — бисекцией по времени реальных траекторий агента
        if int(os.environ.get('CUTA', 0)): DB = [cut_adaptive(A)]                              # CUTA=1 (research-15, идея пользователя): разрыв V по окрестности k узлов — плоскость → разрез на две плоскости → тонкая стена
        DB = np.concatenate(DB + ([GB] if GB is not None else [])); DB[:, 0] = G.wrap(DB[:, 0]); G.BARRIER = cKDTree(DB); np.save(os.path.join(D, 'barrier.npy'), DB); hist.append(dict(cells=len(done), nodes=int(A.N), barrier_pts=len(DB))); print('проход', ps, hist[-1], flush=True); continue                                                       # невязка: V в центре четырёхугольника (среднее 4 узлов) против шага Беллмана из этой точки
    for c in A.cells:
        g = c.G; nt = len(g); v = A.V[c.o:c.o + nt * c.m].reshape(nt, c.m); Y.append(((g[:-1, :-1] + g[1:, :-1] + g[:-1, 1:] + g[1:, 1:]) / 4).reshape(-1, 2)); Vo.append(((v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:]) / 4).ravel())
    Y, Vo = np.concatenate(Y), np.concatenate(Vo); rhs = np.full(len(Y), np.inf)
    for u in G.US: tg = A.tgoal(Y, u); rhs = np.minimum(rhs, np.where(np.isfinite(tg), tg, G.DTN + A.vstar(G.step(Y, u))))
    ok = (Vo < G.BIG / 2) & (rhs < G.BIG / 2); res = np.abs(Vo - rhs); bad = ok & (res > RTOL); M_ = np.zeros_like(LEV, bool); M_[binof(Y[bad])] = True
    M_ = M_ | np.roll(M_, 1, 0) | np.roll(M_, -1, 0); M_ = M_ | np.roll(M_, 1, 1) | np.roll(M_, -1, 1); LEV[M_] += 1
    hist.append(dict(cells=len(done), nodes=int(A.N), res_med=round(float(np.median(res[ok])), 4), res_p95=round(float(np.quantile(res[ok], .95)), 4), bad=round(float(bad.mean()), 3), lev_max=int(LEV.max()), lev_area=round(float((LEV > 0).mean()), 3)))
    print('проход', ps, hist[-1], flush=True); np.save(os.path.join(D, 'lev.npy'), LEV)
A.finish(); con = [G.contact_stats(l, ix) for l, ix in zip(A.layers, A.idx)]; X = np.c_[rng.uniform(-G.XL, G.XL, 20000), rng.uniform(-G.WL, G.WL, 20000)]; cov = [round(float(ix.covered(X).mean()), 3) for ix in A.idx]
import time as _tm; _tb = _tm.time() - _T0R19; status('считаю цену V', cells=len(done), cells_by_layer=[len(l) for l in A.layers], nodes=int(A.N), contact=con, cover=cov); A.solve(); _ts = _tm.time() - _T0R19 - _tb   # research-19 (слово пользователя): время построения атласа и расчёта V
gx, gw = np.linspace(-G.XL, G.XL, 161), np.linspace(-G.WL, G.WL, 141); GX, GW = np.meshgrid(gx, gw); VV = A.vstar(np.c_[GX.ravel(), GW.ravel()]).reshape(GX.shape)
np.savez(os.path.join(D, 'value.tmp.npz'), gx=gx, gw=gw, V=VV); os.replace(os.path.join(D, 'value.tmp.npz'), os.path.join(D, 'value.npz'))
if os.environ.get('PROBE'):                                                                    # research-16: разбор заниженной V — какая клетка даёт минимум в точке
    import pickle; PQ = np.array(json.loads(os.environ['PROBE'])); I, IDX, W = A.stencils(PQ); v = A.interp(W, A.V[IDX]); cid = np.searchsorted(np.array([c.o for c in A.cells]), IDX[:, 0], 'right') - 1; rep = []
    for q in range(len(PQ)):
        k = np.flatnonzero(I == q); k = k[np.argsort(v[k])][:4]
        rep.append([dict(v=float(v[t]), cell=int(cid[t]), u=float(A.cells[cid[t]].u), VI=A.V[IDX[t]].round(3).tolist(), W=W[t].round(3).tolist(), P=A.P[IDX[t]].round(3).tolist()) for t in k])
    pickle.dump(dict(rep=rep, cells=[dict(G=A.cells[i].G, u=A.cells[i].u, V=A.V[A.cells[i].o:A.cells[i].o + A.cells[i].G.shape[0] * A.cells[i].m].reshape(A.cells[i].G.shape[0], -1)) for i in sorted({r_['cell'] for rr in rep for r_ in rr})]), open(os.path.join(D, 'probe.pkl'), 'wb'))
    for q, rr in enumerate(rep): print('PROBE', PQ[q].tolist(), json.dumps(rr[:2]), flush=True)
    Tq, _, pq = A.rollout(PQ)
    for q in range(len(PQ)):
        pp = pq[:, q]; vv = A.vstar(pp); dv = vv[:-1] - vv[1:]                                 # вдоль пути V должна падать на DTN за шаг; падение больше — «короткий путь»
        print('TRACE', q, 'T', round(float(Tq[q]), 3), 'V0', round(float(vv[0]), 3), flush=True)
        print('  path', [(round(k * G.DTN, 2), pp[k].round(2).tolist(), round(float(vv[k]), 2)) for k in range(0, len(pp), 5) if k * G.DTN <= Tq[q] + .3], flush=True)
        for k in np.flatnonzero(dv > 2.5 * G.DTN)[:8]:
            I_, IDX_, W_ = A.stencils(pp[k:k + 1]); v_ = A.interp(W_, A.V[IDX_]); t_ = int(np.argmin(v_)) if len(v_) else -1; cid_ = (np.searchsorted(np.array([c.o for c in A.cells]), IDX_[t_, 0], 'right') - 1) if t_ >= 0 else -1
            print('  JUMP k', int(k), 'x', pp[k].round(3).tolist(), '->', pp[k + 1].round(3).tolist(), 'V', round(float(vv[k]), 3), '->', round(float(vv[k + 1]), 3), 'cell', int(cid_), 'VI', A.V[IDX_[t_]].round(3).tolist() if t_ >= 0 else None, 'TT', None if cid_ < 0 or getattr(A.cells[cid_], 'TT', None) is None else A.cells[cid_].TT[:, 2].round(3).tolist()[:40], flush=True)
base = dict(refine=hist, cells=len(done), cells_by_layer=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), contact=con, cover=cov, big_nodes=round(float((A.V >= G.BIG / 2).mean()), 3)); status('агент', **base)
Q, ref = G.starts_ref(); ok = ref > .05; Q, ref = Q[ok], ref[ok]; _t1 = _tm.time(); T, sw, path = A.rollout(Q); _ta = _tm.time() - _t1
_qt = []                                                                                      # research-19: время ОДНОГО запроса (старт → траектория до цели) — по одному старту, распределение
from tqdm import tqdm as _tq
for _i in _tq(range(min(len(Q), int(os.environ.get('QTIME', 20)))), desc='замер времени запроса (по одному старту)', mininterval=5): _t1 = _tm.time(); A.rollout(Q[_i:_i + 1]); _qt.append(_tm.time() - _t1)
_qt = np.array(_qt); np.save(os.path.join(D, 'qtime.npy'), _qt); base.update(gstat={k_: int(v_) for k_, v_ in G.GSTAT.items()}, mg5=[round(float(q_), 3) for q_ in np.quantile(G.MGD, [.1, .5, .9, 1.])] if getattr(G, 'MGD', None) else None, t_build=round(_tb, 1), t_solve=round(_ts, 1), t_agent_batch=round(_ta, 1), q_ms=dict(n=len(_qt), med=round(float(np.median(_qt)) * 1e3, 1), p90=round(float(np.quantile(_qt, .9)) * 1e3, 1), max=round(float(_qt.max()) * 1e3, 1), mean=round(float(_qt.mean()) * 1e3, 1)) if len(_qt) else None)
np.savez(os.path.join(D, 'agent.tmp.npz'), Q=Q, T=T, ref=ref, sw=sw); os.replace(os.path.join(D, 'agent.tmp.npz'), os.path.join(D, 'agent.npz'))
fz = np.isfinite(T); r = T[fz] / ref[fz]
if G.SYS == 'pend':                                                                           # пути для картинки: 7 стартов + их зеркала (−φ, −ω) — видно закрутку в обе стороны
    S0 = np.array([(2.2, 0.), (2.8, .8), (1.5, 1.2), (.8, -1.5), (2.5, -1.), (3.0, .3), (1.9, -.6)]); S = np.r_[S0, -S0]
    PN = int(os.environ.get('PATHN', 0))                                                     # PATHN > 0 (слово пользователя 2026-10-06): старты путей — равномерная сетка по всему полю
    PR = int(os.environ.get('PATHR', 0))                                                     # PATHR > 0 (слово пользователя 2026-10-06): PATHR стартов путей равномерно-случайно по полю
    if PR: S = np.c_[np.random.default_rng(int(os.environ.get('PSEED', 0))).uniform(-G.XL, G.XL, PR), np.random.default_rng(int(os.environ.get('PSEED', 0)) + 1).uniform(-G.WL, G.WL, PR)]
    elif PN: nx = max(2, int(round(np.sqrt(PN * 2 * G.XL / (2 * G.WL))))); nw = max(2, PN // nx); S = np.stack(np.meshgrid(-G.XL + (np.arange(nx) + .5) * 2 * G.XL / nx, -G.WL + (np.arange(nw) + .5) * 2 * G.WL / nw), -1).reshape(-1, 2)
else: S = np.array([[-1.3, -.6], [1.4, 1.0], [-.4, 1.3], [.9, -1.2], [-1.4, .9], [1.3, .6], [-1.4, -1.0], [.4, -1.3]])
T2, _, p2 = A.rollout(S); paths = []
for i in range(len(S)):
    pp = p2[:, i]; st_ = np.flatnonzero(np.all(np.abs(pp[1:] - pp[:-1]) < 1e-12, axis=1)); paths.append(dict(p=pp[:(st_[0] + 1 if len(st_) else len(pp))].astype(np.float32), T=float(T2[i]), ref=None))
dump('paths.pkl', paths, js=False)
status('готово', reach=round(float(fz.mean()), 3), T_mean=round(float(r.mean()), 4), T_med=round(float(np.median(r)), 4), T_max=round(float(r.max()), 3), T_min=round(float(r.min()), 3), sw_med=float(np.median(sw[fz])), **base)
print(open(os.path.join(D, 'status.json')).read())
